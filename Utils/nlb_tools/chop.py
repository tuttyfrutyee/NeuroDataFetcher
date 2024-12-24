import os
import logging
from os import path

# If you actually need HDF5 file handling, keep h5py. Otherwise, remove.
import h5py  

import pandas as pd
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

# --------------------------------------------------
#   Utility Functions
# --------------------------------------------------

def chop_data(data: np.ndarray, 
              overlap: int, 
              window: int, 
              offset: int = 0) -> np.ndarray:
    """
    Rearranges an array of continuous data into overlapping segments.

    Parameters
    ----------
    data : np.ndarray
        A TxN numpy array of N features measured across T time points.
    overlap : int
        The number of points to overlap between subsequent segments.
    window : int
        The number of time points in each segment.
    offset : int, optional
        Number of points to skip from the start before chopping, 
        by default 0

    Returns
    -------
    np.ndarray
        An SxTxN numpy array of S overlapping segments spanning 
        T time points with N features.
    """
    offset_data = data[offset:]
    # Number of chops
    n_chops = int((offset_data.shape[0] - window) / (window - overlap)) + 1

    # Shape for the as_strided output
    shape = (n_chops, window, offset_data.shape[-1])
    # Strides for stepping along the time axis
    strides = (offset_data.strides[0] * (window - overlap), 
               offset_data.strides[0], 
               offset_data.strides[1])

    chopped = np.lib.stride_tricks.as_strided(
        offset_data, shape=shape, strides=strides
    ).copy().astype(np.float32)

    return chopped


def merge_chops(data: np.ndarray, 
                overlap: int, 
                orig_len: int = None, 
                smooth_pwr: float = 2.0) -> np.ndarray:
    """
    Merges an array of overlapping segments back into continuous data.

    Parameters
    ----------
    data : np.ndarray
        An SxTxN numpy array of S overlapping segments spanning 
        T time points with N features.
    overlap : int
        The number of overlapping points between subsequent segments.
    orig_len : int, optional
        The original length of the continuous data, by default None
    smooth_pwr : float, optional
        The power of smoothing used in blending overlaps.
        - Use np.inf to keep only ends of chops.
        - Use 1 for linear blending.
        - >1 increasingly prefers ends; <1 increasingly prefers beginnings.

    Returns
    -------
    np.ndarray
        A TxN numpy array of N features measured across T time points.
    """
    if smooth_pwr < 1:
        logger.warning(
            "Using `smooth_pwr` < 1 for merging chops is not recommended."
        )

    # The middle portion of each chop that has no overlap
    full_weight_len = data.shape[1] - 2 * overlap

    # Create x-values for the overlap ramp
    # (avoid dividing by zero if overlap=0)
    x = np.linspace(1 / overlap, 1 - 1 / overlap, overlap) if overlap > 0 else []
    x = np.array(x, dtype=np.float32)

    # Power-function ramp
    # ramp: shape (overlap, 1)
    ramp = 1 - (x ** smooth_pwr)
    ramp = np.expand_dims(ramp, axis=-1)

    merged_segments = []
    split_ixs = np.cumsum([overlap, full_weight_len])

    for i in range(len(data)):
        # Split chop into: first-overlap, middle, last-overlap
        first, middle, last = np.split(data[i], split_ixs, axis=0)

        if i == 0:
            # The first chop: only ramp out the last portion
            last *= ramp
        elif i == len(data) - 1:
            # The last chop: ramp out the first portion and keep last portion fully
            first = first * (1 - ramp) + merged_segments.pop()
        else:
            # Middle chop: ramp out the first portion, ramp out the last portion
            first = first * (1 - ramp) + merged_segments.pop()
            last *= ramp

        merged_segments.extend([first, middle, last])

    merged = np.concatenate(merged_segments, axis=0)

    # Pad with NaNs if the original length was shorter
    if orig_len is not None and len(merged) < orig_len:
        n_to_pad = orig_len - len(merged)
        nans = np.full((n_to_pad, merged.shape[1]), np.nan, dtype=np.float32)
        merged = np.concatenate([merged, nans], axis=0)

    return merged


# --------------------------------------------------
#   Classes
# --------------------------------------------------

class SegmentRecord:
    """
    Stores information needed to reconstruct a segment from chops.
    """

    def __init__(self, 
                 seg_id: int, 
                 clock_time: pd.Series, 
                 offset: int, 
                 n_chops: int, 
                 overlap: int):
        """
        Parameters
        ----------
        seg_id : int
            The ID of this segment.
        clock_time : pd.Series
            The TimeDeltaIndex of the original data from this segment.
        offset : int
            The offset of the chops from the start of the segment.
        n_chops : int
            The number of chops that make up this segment.
        overlap : int
            The number of bins of overlap between adjacent chops.
        """
        self.seg_id = seg_id
        self.clock_time = clock_time
        self.offset = offset
        self.n_chops = n_chops
        self.overlap = overlap

    def rebuild_segment(self, 
                        chops: np.ndarray, 
                        smooth_pwr: float = 2.0) -> pd.DataFrame:
        """
        Reassembles a segment from its chops.

        Parameters
        ----------
        chops : np.ndarray
            A 3D numpy array of shape (n_chops, seg_len, data_dim)
            that holds the data from all of the chops in this segment.
        smooth_pwr : float, optional
            The power to use for smoothing in `merge_chops`.
            Default is 2.0.

        Returns
        -------
        pd.DataFrame
            A DataFrame of reconstructed segment data, indexed by the 
            original `clock_time`.
        """
        try:
            # Merge the chops for this segment
            merged_array = merge_chops(
                chops, 
                overlap=self.overlap,
                orig_len=(len(self.clock_time) - self.offset),
                smooth_pwr=smooth_pwr
            )

            # Add NaNs for points that were not modeled due to offset
            data_dim = merged_array.shape[1]
            offset_nans = np.full((self.offset, data_dim), np.nan, dtype=np.float32)
            merged_array = np.concatenate([offset_nans, merged_array], axis=0)

            # Recreate DataFrame with the appropriate `clock_time`
            segment_df = pd.DataFrame(merged_array, index=self.clock_time)
            return segment_df

        except Exception as e:
            logger.error(f"Error rebuilding segment {self.seg_id}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on failure


class ChopInterface:
    """
    Chops data from NWBDatasets or other DataFrames into segments 
    with fixed overlap.
    """

    def __init__(self,
                 window: int,
                 overlap: int,
                 max_offset: int = 0,
                 chop_margins: int = 0,
                 random_seed: int = None):
        """
        Initializes a ChopInterface.

        Parameters
        ----------
        window : int
            The length of chopped segments in ms.
        overlap : int
            The overlap between chopped segments in ms.
        max_offset : int, optional
            The maximum offset of the first chop from the beginning of 
            each segment in ms. The actual offset will be chosen 
            randomly. Default is 0 (no offset).
        chop_margins : int, optional
            The size of extra margins to add to either end of each chop 
            in bins (for use with temporal_shift, etc.). Default is 0.
        random_seed : int, optional
            The random seed for generating the dataset. Default is None
            (no fixed seed).
        """

        def to_timedelta(ms):
            return pd.to_timedelta(ms, unit='ms')

        self.window = to_timedelta(window)
        self.overlap = to_timedelta(overlap)
        self.max_offset = to_timedelta(max_offset)
        self.chop_margins = chop_margins

        # Using a dedicated NumPy Generator is preferred in modern code:
        self.rng = np.random.default_rng(seed=random_seed)

        # Will be populated by calls to chop(...)
        self.segment_records = []

    def chop(self, 
             neural_df: pd.DataFrame, 
             chop_fields) -> dict[str, np.ndarray]:
        """
        Chops a trialized or continuous DataFrame into overlapping segments.

        Parameters
        ----------
        neural_df : pd.DataFrame
            A continuous or trialized DataFrame (e.g. from NWB or RDS).
        chop_fields : str or list of str
            Column(s) in `neural_df` to chop.

        Returns
        -------
        dict of str -> np.ndarray
            A data_dict of the chopped data. Each entry is a 3D numpy 
            array (samples x time x features).
        """
        if isinstance(chop_fields, str):
            chop_fields = [chop_fields]

        # Prepare dimension splitting
        data_fields = sorted(chop_fields)

        def get_field_dim(field: str) -> int:
            """
            Returns the dimensionality of the field.
            """
            arr = neural_df[field]
            # If 1D, shape is (T,); make it 2D for uniform handling
            return arr.shape[1] if len(arr.shape) > 1 else 1

        data_dims = [get_field_dim(f) for f in data_fields]
        data_splits = np.cumsum(data_dims[:-1])  # where to split the cat array

        logger.info(f"Chopping data field(s) {data_fields} with "
                    f"dimension(s) {data_dims}.")

        # Determine bin_width and segment grouping
        if 'trial_id' in neural_df.columns:
            # Trialized data
            bin_width = neural_df.clock_time.iloc[1] - neural_df.clock_time.iloc[0]
            segments = neural_df.groupby('trial_id')
        else:
            # Continuous data
            bin_width = neural_df.index[1] - neural_df.index[0]

            # For continuous data, check if any columns have NaNs.
            # If so, we attempt to split them out.
            # We do a combined check across all fields in chop_fields.
            any_nan = neural_df[chop_fields].isna()
            # If multiple fields, use `.any(axis=1)`.
            if isinstance(any_nan, pd.DataFrame):
                any_nan = any_nan.any(axis=1)

            if any_nan.any():
                changes = any_nan.diff()
                # We'll look for the boundary points
                splits = np.where(changes)[0].tolist() + [len(neural_df)]
                # We'll group these in pairs: [start_of_nan, end_of_nan, ...]
                # This logic depends on how you want to handle NaN-blocks, 
                # so adapt if needed.
                segments = {}
                idx = 0
                segment_id = 1
                while idx < (len(splits) - 1):
                    start = splits[idx]
                    end = splits[idx + 1]
                    segments[segment_id] = neural_df.iloc[start:end].reset_index()
                    segment_id += 1
                    idx += 2
                segments = segments.items()
            else:
                segments = {1: neural_df.reset_index()}.items()

        # Convert timedeltas to integer bins
        window_bins = int(self.window / bin_width)
        overlap_bins = int(self.overlap / bin_width)

        # Convert chop_margins to an actual timedelta if needed
        chop_margins_td = pd.to_timedelta(self.chop_margins * bin_width, unit='ms')

        # Offsets
        if 'trial_id' in neural_df.columns:
            # For trialized data, we can use the user-specified max_offset
            max_offset_bins = int(self.max_offset / bin_width)
            max_offset_td = self.max_offset

            # Use our dedicated RNG
            def get_offset():
                return self.rng.integers(low=0, high=max_offset_bins + 1)

        else:
            # For continuous data, offset usage is discouraged
            max_offset_bins = 0
            max_offset_td = pd.to_timedelta(0)
            logger.info("Ignoring offset for continuous data.")
            def get_offset():
                return 0

        def to_ms(td: pd.Timedelta) -> int:
            return int(td.total_seconds() * 1000)

        chop_msg = " - ".join([
            "Chopping data",
            f"Window: {window_bins} bins, {to_ms(self.window)} ms",
            f"Overlap: {overlap_bins} bins, {to_ms(self.overlap)} ms",
            f"Max offset: {max_offset_bins} bins, {to_ms(max_offset_td)} ms",
            f"Chop margins: {self.chop_margins} bins, {to_ms(chop_margins_td)} ms",
        ])
        logger.info(chop_msg)

        # Perform the actual chopping
        data_dict = defaultdict(list)
        segment_records_local = []

        for segment_id, segment_df in segments:
            # Combine the selected fields
            data_arrays = []
            for f in data_fields:
                arr = segment_df[f]
                if len(arr.shape) == 1:
                    arr = arr.to_numpy()[:, None]
                else:
                    arr = arr.to_numpy()
                data_arrays.append(arr)

            segment_array = np.concatenate(data_arrays, axis=1)

            # If margins are used, pad the data
            if self.chop_margins > 0:
                seg_dim = segment_array.shape[1]
                pad = np.full((self.chop_margins, seg_dim), 0.0001, dtype=np.float32)
                segment_array = np.concatenate([pad, segment_array, pad], axis=0)

            # Sample an offset (for trialized data only)
            offset_bins = get_offset()

            chops = chop_data(
                data=segment_array,
                overlap=overlap_bins + 2 * self.chop_margins,
                window=window_bins + 2 * self.chop_margins,
                offset=offset_bins
            )

            # Split chops into original fields
            data_chops_split = np.split(chops, data_splits, axis=2)

            # Populate data_dict
            for field, dchop in zip(data_fields, data_chops_split):
                data_dict[field].append(dchop)

            # Create a record to help reconstruct segments
            seg_rec = SegmentRecord(
                seg_id=segment_id,
                clock_time=segment_df.clock_time 
                    if 'clock_time' in segment_df 
                    else pd.Series(segment_df.index), 
                offset=offset_bins,
                n_chops=len(chops),
                overlap=overlap_bins
            )
            segment_records_local.append(seg_rec)

        # Store them on self so merge(...) can access
        self.segment_records = segment_records_local

        # Merge all segment chops into a single array for each field
        data_dict = {
            name: np.concatenate(chop_list, axis=0)
            for name, chop_list in data_dict.items()
        }

        n_chops_total = next(iter(data_dict.values())).shape[0] if data_dict else 0
        n_segments = len(segment_records_local)
        logger.info(f"Created {n_chops_total} chops from {n_segments} segment(s).")

        return data_dict

    def merge(self, 
              chopped_data: dict[str, np.ndarray], 
              smooth_pwr: float = 2.0) -> pd.DataFrame:
        """
        Merges chopped data to reconstruct the original input sequence.

        Parameters
        ----------
        chopped_data : dict of str -> np.ndarray
            The chopped data, each value is a 3D array: (S, T, F).
        smooth_pwr : float, optional
            The power to use for smoothing in `merge_chops`. 
            Default is 2.0.

        Returns
        -------
        pd.DataFrame
            A merged DataFrame indexed by the clock time of the original 
            chops. Columns are a MultiIndex with the field name and 
            feature index.
        """
        if not self.segment_records:
            logger.warning("No segment records found. Did you run chop() first?")
            return pd.DataFrame()

        output_fields = sorted(chopped_data.keys())
        output_arrays = [chopped_data[f] for f in output_fields]
        output_dims = [arr.shape[-1] for arr in output_arrays]

        # Concatenate across the feature dimension
        # shape: (S, T, sum_of_dims)
        output_full = np.concatenate(output_arrays, axis=-1)

        # Split into chops belonging to each segment
        seg_split_ixs = np.cumsum([rec.n_chops for rec in self.segment_records])[:-1]
        seg_chops = np.split(output_full, seg_split_ixs, axis=0)

        # Rebuild each segment
        segment_dfs = []
        for record, chops in zip(self.segment_records, seg_chops):
            df_seg = record.rebuild_segment(chops, smooth_pwr=smooth_pwr)
            segment_dfs.append(df_seg)

        # Concatenate all segments
        merged_df = pd.concat(segment_dfs)

        # Build a multi-index for columns
        midx_tuples = []
        for sig, dim in zip(output_fields, output_dims):
            for i in range(dim):
                midx_tuples.append((sig, f"{i:04}"))

        merged_df.columns = pd.MultiIndex.from_tuples(midx_tuples, names=["signal", "feature"])

        return merged_df
