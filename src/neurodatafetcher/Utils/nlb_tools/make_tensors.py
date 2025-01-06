import logging
import os
from itertools import product

import h5py
import numpy as np
import pandas as pd

from .nwb_interface import NWBDataset
from .chop import ChopInterface, chop_data, merge_chops

logger = logging.getLogger(__name__)

PARAMS = {
    'mc_maze': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'behavior_source': 'data',
        'behavior_field': 'hand_vel',
        'lag': 100,
        'make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'eval_make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['trial_type', 'trial_version'],
            'make_params': {
                'align_field': 'move_onset_time',
                'align_range': (-250, 450),
            },
            'kern_sd': 70,
        },
    },
    'mc_rtt': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'rate_field': 'rates',
        'horate_field': 'heldout_rates',
        'behavior_source': 'data',
        'behavior_field': 'finger_vel',
        'lag': 140,
        'make_params': {
            'align_field': 'start_time',
            'align_range': (0, 600),
            'allow_overlap': True,
        },
        'eval_make_params': {
            'align_field': 'start_time',
            'align_range': (0, 600),
            'allow_overlap': True,
        },
        'fp_len': 200,
    },
    'area2_bump': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'rate_field': 'rates',
        'horate_field': 'heldout_rates',
        'behavior_source': 'data',
        'behavior_field': 'hand_vel',
        'decode_masks': lambda x: np.stack([x.ctr_hold_bump == 0, x.ctr_hold_bump == 1]).T,
        'lag': -20,
        'make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-100, 500),
        },
        'eval_make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-100, 500),
            'allow_overlap': True,
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['cond_dir', 'ctr_hold_bump'],
            'make_params': {
                    'align_field': 'move_onset_time',
                    'align_range': (-100, 500),
            },
            'kern_sd': 40,
        },
    },
    'dmfc_rsg': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'behavior_source': 'trial_info',
        'behavior_mask': lambda x: x.is_outlier == 0,
        'behavior_field': ['is_eye', 'theta', 'is_short', 'ts', 'tp'],
        'jitter': lambda x: np.stack([
            np.zeros(len(x)),
            np.where(
                x.split == 'test',
                np.zeros(len(x)),
                np.clip(1500.0 - x.get('tp', pd.Series(np.nan)).to_numpy(), 0.0, 300.0)
            )
        ]).T,
        'make_params': {
            'align_field': 'go_time',
            'align_range': (-1500, 0),
            'allow_overlap': True,
        },
        'eval_make_params': {
            'start_field': 'set_time',
            'end_field': 'go_time',
            'align_field': 'go_time',
        },
        'eval_tensor_params': {
            'seg_len': 1500,
            'pad': 'front'
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['is_eye', 'theta', 'is_short', 'ts'],
            'make_params': {
                'start_field': 'set_time',
                'end_field': 'go_time',
                'align_field': 'go_time',
            },
            'kern_sd': 70,
            'pad': 'front',
            'seg_len': 1500,
            'skip_mask': lambda x: x.is_outlier == 1,
        },
    },
    'mc_maze_large': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'rate_field': 'rates',
        'horate_field': 'heldout_rates',
        'behavior_source': 'data',
        'behavior_field': 'hand_vel',
        'lag': 120,
        'make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'eval_make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['trial_type', 'trial_version'],
            'make_params': {
                'align_field': 'move_onset_time',
                'align_range': (-250, 450),
            },
            'kern_sd': 50,
        },
    },
    'mc_maze_medium': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'rate_field': 'rates',
        'horate_field': 'heldout_rates',
        'behavior_source': 'data',
        'behavior_field': 'hand_vel',
        'lag': 120,
        'make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'eval_make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['trial_type', 'trial_version'],
            'make_params': {
                'align_field': 'move_onset_time',
                'align_range': (-250, 450),
            },
            'kern_sd': 50,
        },
    },
    'mc_maze_small': {
        'spk_field': 'spikes',
        'hospk_field': 'heldout_spikes',
        'rate_field': 'rates',
        'horate_field': 'heldout_rates',
        'behavior_source': 'data',
        'behavior_field': 'hand_vel',
        'lag': 120,
        'make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'eval_make_params': {
            'align_field': 'move_onset_time',
            'align_range': (-250, 450),
        },
        'fp_len': 200,
        'psth_params': {
            'cond_fields': ['trial_type', 'trial_version'],
            'make_params': {
                'align_field': 'move_onset_time',
                'align_range': (-250, 450),
            },
            'kern_sd': 50,
        },
    },
}


def make_train_input_tensors(dataset, dataset_name,
                             trial_split='train',
                             update_params=None,
                             save_file=True,
                             return_dict=True,
                             save_path="train_input.h5",
                             include_behavior=False,
                             include_forward_pred=False,
                             seed=0):
    """
    Makes model training input tensors and optionally saves them as an .h5 file.

    Parameters
    ----------
    dataset : NWBDataset
        An instance of NWBDataset to make tensors from
    dataset_name : {'mc_maze', 'mc_rtt', 'area2_bum', 'dmfc_rsg',
                    'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
        Name of dataset. Used to select default parameters from PARAMS
    trial_split : {'train', 'val'}, array-like, or list, optional
        The selection of trials to make the tensors with.
        By default 'train'
    update_params : dict, optional
        New parameters with which to update default dict from PARAMS
    save_file : bool, optional
        Whether to save the reshaped data to an HDF5 file, by default True
    return_dict : bool, optional
        Whether to return the reshaped data in a dict, by default True
    save_path : str, optional
        Path to where the HDF5 output file should be saved, by default "train_input.h5"
    include_behavior : bool, optional
        Whether to include behavioral data in the returned tensors, by default False
    include_forward_pred : bool, optional
        Whether to include forward-prediction spiking data in the returned tensors, by default False
    seed : int, optional
        Seed for random generator used for jitter

    Returns
    -------
    dict of np.array
        A dict containing 3D numpy arrays of spiking data for the indicated trials
        and possibly additional data based on provided arguments.
    """
    assert isinstance(dataset, NWBDataset), "`dataset` must be an instance of NWBDataset"
    assert dataset_name in PARAMS.keys(), f"`dataset_name` must be one of {list(PARAMS.keys())}"
    is_valid_split = isinstance(trial_split, (pd.Series, np.ndarray, list)) or trial_split in ['train', 'val']
    assert is_valid_split, "Invalid `trial_split` argument."

    # Fetch and update params
    params = PARAMS[dataset_name].copy()
    if update_params is not None:
        params.update(update_params)

    # Add filename extension if necessary
    if not save_path.endswith('.h5'):
        save_path += '.h5'

    # Unpack params
    spk_field = params['spk_field']
    hospk_field = params['hospk_field']
    make_params = params['make_params'].copy()
    jitter = params.get('jitter', None)

    # Prep mask
    trial_mask = _prep_mask(dataset, trial_split)

    # Prep jitter if necessary
    if jitter is not None:
        np.random.seed(seed)
        jitter_vals = _prep_jitter(dataset, trial_mask, jitter)
        align_field = make_params.get('align_field', make_params.get('start_field', 'start_time'))
        align_vals = dataset.trial_info[trial_mask][align_field]
        align_jit = align_vals + pd.to_timedelta(jitter_vals, unit='ms')
        align_jit.name = align_field.replace('_time', '_jitter_time')
        dataset.trial_info = pd.concat([dataset.trial_info, align_jit], axis=1)
        if 'align_field' in make_params:
            make_params['align_field'] = align_jit.name
        else:
            make_params['start_field'] = align_jit.name

    # Make output spiking arrays and put into data_dict
    train_dict = make_stacked_array(dataset, [spk_field, hospk_field], make_params, trial_mask)
    data_dict = {
        'train_spikes_heldin': train_dict[spk_field],
        'train_spikes_heldout': train_dict[hospk_field],
    }

    # Add behavior data if necessary
    if include_behavior:
        behavior_source = params['behavior_source']
        behavior_field = params['behavior_field']
        behavior_make_params = _prep_behavior(dataset, params.get('lag', None), make_params)

        if behavior_source == 'data':
            train_behavior = make_jagged_array(
                dataset, [behavior_field], behavior_make_params, trial_mask
            )[0][behavior_field]
        else:
            train_behavior = (
                dataset.trial_info[trial_mask][behavior_field]
                .apply(lambda x: x.dt.total_seconds() if hasattr(x, "dt") else x)
                .to_numpy()
                .astype('float')
            )

        # Filter out behavior on certain trials if necessary
        if 'behavior_mask' in params:
            if callable(params['behavior_mask']):
                behavior_mask = params['behavior_mask'](dataset.trial_info[trial_mask])
            else:
                behavior_mask, _ = params['behavior_mask']
            train_behavior[~behavior_mask] = np.nan

        data_dict['train_behavior'] = train_behavior

    # Add forward prediction data if necessary
    if include_forward_pred:
        fp_len = params['fp_len']
        fp_steps = fp_len / dataset.bin_width
        fp_make_params = _prep_fp(make_params, fp_steps, dataset.bin_width)
        fp_dict = make_stacked_array(dataset, [spk_field, hospk_field], fp_make_params, trial_mask)
        data_dict['train_spikes_heldin_forward'] = fp_dict[spk_field]
        data_dict['train_spikes_heldout_forward'] = fp_dict[hospk_field]

    # Delete jitter column
    if jitter is not None:
        dataset.trial_info.drop(align_jit.name, axis=1, inplace=True)

    # Save and return data
    if save_file:
        save_to_h5(data_dict, save_path, overwrite=True)
    if return_dict:
        return data_dict


def make_eval_input_tensors(dataset, dataset_name,
                            trial_split='val',
                            update_params=None,
                            save_file=True,
                            return_dict=True,
                            save_path="eval_input.h5",
                            seed=0):
    """
    Makes model evaluation input tensors and optionally saves them as an .h5 file.

    Parameters
    ----------
    dataset : NWBDataset
        An instance of NWBDataset to make tensors from
    dataset_name : {'mc_maze', 'mc_rtt', 'area2_bum', 'dmfc_rsg',
                    'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
        Name of dataset. Used to select default parameters from PARAMS
    trial_split : {'train', 'val', 'test'}, array-like, or list, optional
        The selection of trials to make the tensors with.
        By default 'val'
    update_params : dict, optional
        New parameters with which to update default dict from PARAMS
    save_file : bool, optional
        Whether to save the reshaped data to an HDF5 file, by default True
    return_dict : bool, optional
        Whether to return the reshaped data in a dict, by default True
    save_path : str, optional
        Path to where the HDF5 output file should be saved
    seed : int, optional
        Seed for random generator used for jitter

    Returns
    -------
    dict of np.array
        A dict containing 3D numpy arrays of spiking data for the indicated trials.
    """
    assert isinstance(dataset, NWBDataset), "`dataset` must be an instance of NWBDataset"
    assert dataset_name in PARAMS.keys(), f"`dataset_name` must be one of {list(PARAMS.keys())}"
    is_valid_split = isinstance(trial_split, (pd.Series, np.ndarray, list)) or trial_split in ['train', 'val', 'test']
    assert is_valid_split, "Invalid `trial_split` argument."

    # Fetch and update params
    params = PARAMS[dataset_name].copy()
    if update_params is not None:
        params.update(update_params)

    # Add filename extension if necessary
    if not save_path.endswith('.h5'):
        save_path += '.h5'

    # Unpack params
    spk_field = params['spk_field']
    hospk_field = params['hospk_field']
    make_params = params['make_params'].copy()
    make_params['allow_nans'] = True
    jitter = params.get('jitter', None)

    # Prep mask
    trial_mask = _prep_mask(dataset, trial_split)

    # Prep jitter if necessary
    if jitter is not None:
        np.random.seed(seed)
        jitter_vals = _prep_jitter(dataset, trial_mask, jitter)
        align_field = make_params.get('align_field', make_params.get('start_field', 'start_time'))
        align_vals = dataset.trial_info[trial_mask][align_field]
        align_jit = align_vals + pd.to_timedelta(jitter_vals, unit='ms')
        align_jit.name = align_field.replace('_time', '_jitter_time')
        dataset.trial_info = pd.concat([dataset.trial_info, align_jit], axis=1)
        if 'align_field' in make_params:
            make_params['align_field'] = align_jit.name
        else:
            make_params['start_field'] = align_jit.name

    # Make output spiking arrays and put into data_dict
    if not np.any(dataset.trial_info[trial_mask].split == 'test'):
        eval_dict = make_stacked_array(dataset, [spk_field, hospk_field], make_params, trial_mask)
        data_dict = {
            'eval_spikes_heldin': eval_dict[spk_field],
            'eval_spikes_heldout': eval_dict[hospk_field],
        }
    else:
        # 'test' split has no heldout data
        eval_dict = make_stacked_array(dataset, [spk_field], make_params, trial_mask)
        data_dict = {
            'eval_spikes_heldin': eval_dict[spk_field],
        }

    # Delete jitter column
    if jitter is not None:
        dataset.trial_info.drop(align_jit.name, axis=1, inplace=True)

    # Save and return data
    if save_file:
        save_to_h5(data_dict, save_path, overwrite=True)
    if return_dict:
        return data_dict


def make_eval_target_tensors(dataset, dataset_name,
                             train_trial_split='train',
                             eval_trial_split='val',
                             update_params=None,
                             save_file=True,
                             return_dict=True,
                             save_path="target_data.h5",
                             include_psth=False,
                             seed=0):
    """
    Makes tensors containing target data used to evaluate model predictions.
    Creates 3D arrays containing true heldout spiking data for eval trials and 
    other arrays for model evaluation.

    Parameters
    ----------
    dataset : NWBDataset
        An instance of NWBDataset to make tensors from
    dataset_name : {'mc_maze', 'mc_rtt', 'area2_bum', 'dmfc_rsg',
                    'mc_maze_large', 'mc_maze_medium', 'mc_maze_small'}
        Name of dataset. Used to select default parameters from PARAMS
    train_trial_split : {'train', 'val'}, array-like, or list, optional
        The selection of trials used for training; default 'train'
    eval_trial_split : {'train', 'val'}, array-like, or list, optional
        The selection of trials used for evaluation; default 'val'
    update_params : dict, optional
        New parameters with which to update default dict from PARAMS
    save_file : bool, optional
        Whether to save the reshaped data to an HDF5 file, by default True
    return_dict : bool, optional
        Whether to return the reshaped data in a data dict, by default True
    save_path : str, optional
        Path to where the HDF5 output file should be saved
    include_psth : bool, optional
        Whether to make PSTHs for evaluation, by default False
    seed : int, optional
        Seed for random generator used for jitter

    Returns
    -------
    nested dict of np.array
        Dict containing data for evaluation, including held-out spiking activity
        for eval trials and behavioral correlates.
    """
    assert isinstance(dataset, NWBDataset), "`dataset` must be an instance of NWBDataset"
    assert dataset_name in PARAMS.keys(), f"`dataset_name` must be one of {list(PARAMS.keys())}"
    valid_train_split = isinstance(train_trial_split, (pd.Series, np.ndarray, list)) or train_trial_split in ['train', 'val', 'test']
    valid_eval_split = isinstance(eval_trial_split, (pd.Series, np.ndarray, list)) or eval_trial_split in ['train', 'val', 'test']
    assert valid_train_split, "Invalid `train_trial_split` argument."
    assert valid_eval_split, "Invalid `eval_trial_split` argument."

    # Fetch and update params
    params = PARAMS[dataset_name].copy()
    if update_params is not None:
        params.update(update_params)

    # Add filename extension if necessary
    if not save_path.endswith('.h5'):
        save_path += '.h5'

    # Unpack params
    spk_field = params['spk_field']
    hospk_field = params['hospk_field']
    make_params = params['eval_make_params'].copy()
    behavior_source = params['behavior_source']
    behavior_field = params['behavior_field']
    jitter = params.get('jitter', None)
    eval_tensor_params = params.get('eval_tensor_params', {}).copy()
    fp_len = params['fp_len']
    fp_steps = fp_len / dataset.bin_width
    suf = '' if (dataset.bin_width == 5) else f'_{dataset.bin_width}'

    # Prep masks
    train_mask = _prep_mask(dataset, train_trial_split)
    eval_mask = _prep_mask(dataset, eval_trial_split)
    if isinstance(eval_trial_split, str) and eval_trial_split == 'test':
        # In some older setups, 'none' might be used for ignoring
        ignore_mask = dataset.trial_info.split == 'none'
    else:
        ignore_mask = ~(train_mask | eval_mask)

    # Prep jitter if necessary
    if jitter is not None:
        align_field = make_params.get('align_field', make_params.get('start_field', 'start_time'))

        np.random.seed(seed)
        train_jitter_vals = _prep_jitter(dataset, train_mask, jitter)
        train_align_vals = dataset.trial_info[train_mask][align_field]
        train_align_jit = train_align_vals + pd.to_timedelta(train_jitter_vals, unit='ms')
        train_align_jit.name = align_field.replace('_time', '_jitter_time')
        dataset.trial_info = pd.concat([dataset.trial_info, train_align_jit], axis=1)

        np.random.seed(seed)
        eval_jitter_vals = _prep_jitter(dataset, eval_mask, jitter)
        eval_align_vals = dataset.trial_info[eval_mask][align_field]
        eval_align_jit = eval_align_vals + pd.to_timedelta(eval_jitter_vals, unit='ms')
        eval_align_jit.name = align_field.replace('_time', '_jitter_time')
        dataset.trial_info.loc[eval_mask, eval_align_jit.name] = eval_align_jit
        if 'align_field' in make_params:
            make_params['align_field'] = eval_align_jit.name
        else:
            make_params['start_field'] = eval_align_jit.name
    else:
        train_jitter_vals = None
        eval_jitter_vals = None

    behavior_make_params = _prep_behavior(dataset, params.get('lag', None), make_params)

    # Make spiking arrays for eval trials
    eval_dict = make_stacked_array(dataset, [hospk_field], make_params, eval_mask)

    if behavior_source == 'data':
        # Use make_jagged_array in case some data is cut short at edges
        btrain_dict = make_jagged_array(dataset, [behavior_field], behavior_make_params, train_mask)[0]
        beval_dict = make_jagged_array(dataset, [behavior_field], behavior_make_params, eval_mask)[0]
    else:
        btrain_dict = {}
        beval_dict = {}

    # Retrieve behavioral data
    if behavior_source == 'trial_info':
        train_behavior = (
            dataset.trial_info[train_mask][behavior_field]
            .apply(lambda x: x.dt.total_seconds() if hasattr(x, "dt") else x)
            .to_numpy()
            .astype('float')
        )
        eval_behavior = (
            dataset.trial_info[eval_mask][behavior_field]
            .apply(lambda x: x.dt.total_seconds() if hasattr(x, "dt") else x)
            .to_numpy()
            .astype('float')
        )
    else:
        train_behavior = btrain_dict.get(behavior_field, None)
        eval_behavior = beval_dict.get(behavior_field, None)

    # Mask some behavioral data if desired
    if 'behavior_mask' in params:
        if callable(params['behavior_mask']):
            train_behavior_mask = params['behavior_mask'](dataset.trial_info[train_mask])
            eval_behavior_mask = params['behavior_mask'](dataset.trial_info[eval_mask])
        else:
            train_behavior_mask, eval_behavior_mask = params['behavior_mask']
        train_behavior[~train_behavior_mask] = np.nan
        eval_behavior[~eval_behavior_mask] = np.nan

    # Prepare forward prediction spiking data
    fp_make_params = _prep_fp(make_params, fp_steps, dataset.bin_width)
    fp_dict = make_stacked_array(dataset, [spk_field, hospk_field], fp_make_params, eval_mask)

    # Construct data dict
    data_dict = {
        dataset_name + suf: {
            'eval_spikes_heldout': eval_dict[hospk_field],
            'train_behavior': train_behavior,
            'eval_behavior': eval_behavior,
            'eval_spikes_heldin_forward': fp_dict[spk_field],
            'eval_spikes_heldout_forward': fp_dict[hospk_field],
        }
    }

    # Include decode masks if desired
    if 'decode_masks' in params:
        if callable(params['decode_masks']):
            train_decode_mask = params['decode_masks'](dataset.trial_info[train_mask])
            eval_decode_mask = params['decode_masks'](dataset.trial_info[eval_mask])
        else:
            train_decode_mask, eval_decode_mask = params['decode_masks']
        data_dict[dataset_name + suf]['train_decode_mask'] = train_decode_mask
        data_dict[dataset_name + suf]['eval_decode_mask'] = eval_decode_mask

    # Calculate PSTHs if desired
    if include_psth:
        psth_params = params.get('psth_params', None)
        if psth_params is None:
            logger.warning("PSTHs are not supported for this dataset, skipping...")
        else:
            (train_cond_idx, eval_cond_idx), psths, comb = _make_psth(
                dataset, train_mask, eval_mask, ignore_mask, **psth_params
            )

            data_dict[dataset_name + suf]['eval_cond_idx'] = eval_cond_idx
            data_dict[dataset_name + suf]['train_cond_idx'] = train_cond_idx
            data_dict[dataset_name + suf]['psth'] = psths

            if jitter is not None:
                data_dict[dataset_name + suf]['eval_jitter'] = (eval_jitter_vals / dataset.bin_width).round().astype(int)
                data_dict[dataset_name + suf]['train_jitter'] = (train_jitter_vals / dataset.bin_width).round().astype(int)

    # Delete jitter column(s)
    if jitter is not None:
        dataset.trial_info.drop(eval_align_jit.name, axis=1, inplace=True)

    # Save and return data
    if save_file:
        save_to_h5(data_dict, save_path, overwrite=True)
    if return_dict:
        return data_dict


''' Array creation helper functions '''
def make_stacked_array(dataset, fields, make_params, include_trials):
    """
    Generates 3D trial x time x channel arrays for each given field.

    Parameters
    ----------
    dataset : NWBDataset
        An NWBDataset object to extract data from
    fields : str or list of str
        Field name(s) to extract data for
    make_params : dict
        Arguments for `NWBDataset.make_trial_data`
    include_trials : array-like
        Boolean array to select trials to extract

    Returns
    -------
    dict of np.array
        Dict mapping each field in `fields`
        to a 3D trial x time x channel numpy array
    """
    if 'ignored_trials' in make_params:
        logger.warning(
            "`ignored_trials` found in `make_params`. Deleting and overriding with `include_trials`"
        )
        make_params.pop('ignored_trials')

    if not isinstance(fields, list):
        fields = [fields]

    trial_data = dataset.make_trial_data(ignored_trials=~include_trials, **make_params)
    grouped = list(trial_data.groupby('trial_id', sort=False))

    array_dict = {}
    for field in fields:
        array_dict[field] = np.stack([trial[field].to_numpy() for _, trial in grouped])
    return array_dict


def make_jagged_array(dataset,
                      fields,
                      make_params,
                      include_trials,
                      jitter=None,
                      pad='back',
                      seg_len=None):
    """
    Generates 3D trial x time x channel arrays for each field, accommodating uneven trial lengths.

    Parameters
    ----------
    dataset : NWBDataset
        An NWBDataset object to extract data from
    fields : str or list of str
        Field name(s) to extract data for
    make_params : dict
        Arguments for `NWBDataset.make_trial_data`
    include_trials : array-like
        Boolean array to select trials to extract
    jitter : np.ndarray or list, optional
        Jitter values for each trial (if any), by default None
    pad : {'back', 'front'}, optional
        Whether to pad shorter trials with NaNs
        at the end ('back') or beginning ('front'), by default 'back'
    seg_len : int, optional
        Trial length to limit arrays to in ms, by default None

    Returns
    -------
    list of dict
        Each dict in the returned list maps fields to 
        a 3D trial x time x channel numpy array.
    """
    if 'ignored_trials' in make_params:
        logger.warning(
            "`ignored_trials` found in `make_params`. Overriding with `include_trials`"
        )
        make_params.pop('ignored_trials')

    if not isinstance(fields, list):
        fields = [fields]

    if not isinstance(include_trials, list):
        include_trials = [include_trials]

    if jitter is None:
        jitter = [np.zeros(it.sum()) for it in include_trials]
    elif not isinstance(jitter, list):
        jitter = [jitter]

    trial_data = dataset.make_trial_data(ignored_trials=~np.any(include_trials, axis=0),
                                         **make_params)
    grouped = dict(list(trial_data.groupby('trial_id', sort=False)))
    max_len = np.max([trial.shape[0] for _, trial in grouped.items()]) \
        if seg_len is None else int(round(seg_len / dataset.bin_width))

    dict_list = []
    for trial_sel, jitter_vals in zip(include_trials, jitter):
        trial_ixs = dataset.trial_info[trial_sel].index.to_numpy()
        array_dict = {}

        for field in fields:
            arr = np.full((len(trial_ixs), max_len, dataset.data[field].shape[1]), np.nan)
            for i in range(len(trial_ixs)):
                jit = int(round(jitter_vals[i] / dataset.bin_width))
                data = grouped[trial_ixs[i]][field].to_numpy()

                if pad == 'front':
                    if jit == 0:
                        data = data[-max_len:]
                        arr[i, -data.shape[0]:, :] = data
                    elif jit > 0:
                        data = data[-(max_len - jit):]
                        arr[i, -(data.shape[0] + jit):-jit, :] = data
                    else:
                        data = data[-(max_len - jit):jit]
                        arr[i, -data.shape[0]:, :] = data
                else:  # pad == 'back'
                    if jit == 0:
                        data = data[:max_len]
                        arr[i, :data.shape[0], :] = data
                    elif jit > 0:
                        data = data[jit:(max_len + jit)]
                        arr[i, :data.shape[0], :] = data
                    else:
                        data = data[:(max_len - jit)]
                        arr[i, -jit:(data.shape[0] - jit)] = data

            array_dict[field] = arr
        dict_list.append(array_dict)
    return dict_list


def make_seg_chopped_array(dataset, fields, make_params, chop_params, include_trials):
    """
    Generates chopped 3D arrays from trial segments using ChopInterface for given fields.

    Note: This function is less frequently used and may need more robust testing.

    Parameters
    ----------
    dataset : NWBDataset
        An NWBDataset object to extract data from
    fields : str or list of str
        Field name(s) to extract data for
    make_params : dict
        Arguments for `NWBDataset.make_trial_data`
    chop_params : dict
        Arguments for `ChopInterface` initialization
    include_trials : array-like
        Boolean array to select trials to extract

    Returns
    -------
    tuple
        (array_dict, ChopInterface)
        array_dict : dict mapping each field to a 3D segment x time x channel numpy array
        ChopInterface : the interface that contains chop metadata
    """
    if 'ignored_trials' in make_params:
        logger.warning(
            "`ignored_trials` found in `make_params`. Overriding with `include_trials`"
        )
        make_params.pop('ignored_trials')

    if not isinstance(fields, list):
        fields = [fields]

    trial_data = dataset.make_trial_data(ignored_trials=~include_trials, **make_params)
    ci = ChopInterface(**chop_params)
    array_dict = ci.chop(trial_data, fields)
    return array_dict, ci


def make_cont_chopped_array(dataset, fields, chop_params, lag=0):
    """
    Generates 3D chopped arrays from continuous data using ChopInterface for given fields.
    Note: This function is less frequently used and may need more robust testing.

    Parameters
    ----------
    dataset : NWBDataset
        An NWBDataset object to extract data from
    fields : str or list of str
        Field name(s) to extract data for
    chop_params : dict
        Arguments for `ChopInterface` initialization
    lag : int, optional
        Amount of initial offset (ms) for continuous data
        before chopping, by default 0

    Returns
    -------
    tuple
        (array_dict, ChopInterface)
    """
    if not isinstance(fields, list):
        fields = [fields]

    ci = ChopInterface(**chop_params)

    if lag > 0:
        # Create a new DataFrame with NaNs to append
        lag_df = pd.DataFrame(
            np.full((lag, dataset.data.shape[1]), np.nan),
            index=(dataset.data.index[-lag:] + pd.to_timedelta(lag, unit='ms'))
        )
        data = pd.concat([dataset.data.iloc[lag:], lag_df], axis=0)
    else:
        data = dataset.data

    array_dict = ci.chop(data, fields)
    return array_dict, ci


''' Chop merging helper functions '''
def merge_seg_chops_to_df(dataset, data_dicts, cis):
    """
    Merges segment-chopped 3D arrays back into the main continuous DataFrame.

    Parameters
    ----------
    dataset : NWBDataset
        NWBDataset from which chopped data was generated
    data_dicts : dict of np.array or list of dict
        Dict (or list of dicts) mapping field names to 
        3D segment x time x channel chopped arrays
    cis : ChopInterface or list of ChopInterface
        Corresponding ChopInterface(s) needed for merging
    """
    if not isinstance(data_dicts, list):
        data_dicts = [data_dicts]
    if not isinstance(cis, list):
        cis = [cis]
    assert len(data_dicts) == len(cis), "`data_dicts` and `cis` must be the same length"

    fields = list(data_dicts[0].keys())
    logger.info(f"Merging {fields} into dataframe")

    merged_list = []
    for data_dict, ci in zip(data_dicts, cis):
        merged_df = ci.merge(data_dict)
        merged_list.append(merged_df)
    merged = pd.concat(merged_list, axis=0).reset_index()

    if merged.clock_time.duplicated().sum() != 0:
        logger.warning("Duplicate time indices found. Merging by averaging.")
        merged = merged.groupby('clock_time', sort=False).mean().reset_index()

    dataset.data = pd.concat([dataset.data, merged.set_index('clock_time')], axis=1)


def merge_cont_chops_to_df(dataset, data_dicts, ci, masks):
    """
    Merges continuous-chopped 3D arrays back into the main continuous DataFrame.

    Parameters
    ----------
    dataset : NWBDataset
        NWBDataset from which chopped data was generated
    data_dicts : dict of np.array or list of dict
        Dict (or list of dicts) mapping field names to 
        3D segment x time x channel arrays
    ci : ChopInterface
        The ChopInterface used to chop continuous data
    masks : array-like or list of array-like
        Boolean masks indicating which chops were not dropped
    """
    if not isinstance(data_dicts, list):
        data_dicts = [data_dicts]
    if not isinstance(masks, list):
        masks = [masks]
    assert isinstance(ci, ChopInterface), "`ci` should be a single ChopInterface for merging continuous chops."
    assert len(data_dicts) == len(masks), "`data_dicts` and `masks` must be the same length"

    fields = list(data_dicts[0].keys())
    logger.info(f"Merging {fields} into dataframe")

    num_chops = len(masks[0])
    chop_len = data_dicts[0][fields[0]].shape[1]
    full_dict = {}

    for field in fields:
        num_chan = data_dicts[0][field].shape[2]
        full_arr = np.full((num_chops, chop_len, num_chan), np.nan)
        for data_dict, mask in zip(data_dicts, masks):
            full_arr[mask] = data_dict[field]
        full_dict[field] = full_arr

    merged = ci.merge(full_dict).reset_index()
    if merged.clock_time.duplicated().sum() != 0:
        logger.warning("Duplicate time indices found. Merging by averaging.")
        merged = merged.groupby('clock_time', sort=False).mean().reset_index()

    dataset.data = pd.concat([dataset.data, merged.set_index('clock_time')], axis=1)


''' Miscellaneous helper functions '''
def combine_train_eval(dataset, train_dict, eval_dict, train_split, eval_split):
    """
    Combines two dicts of tensors (from 'train' and 'eval') into one tensor dict
    while preserving the original order of trials.

    Parameters
    ----------
    dataset : NWBDataset
        NWBDataset from which data was extracted
    train_dict : dict
        Dict containing tensors from the `train_split`
    eval_dict : dict
        Dict containing tensors from the `eval_split`
    train_split : str or list
        The trial split(s) in `train_dict`
    eval_split : str or list
        The trial split(s) in `eval_dict`

    Returns
    -------
    dict of np.array
        Merged dict of np.array with arrays containing data from both input dicts.
    """
    train_mask = _prep_mask(dataset, train_split)
    eval_mask = _prep_mask(dataset, eval_split)

    # Check that there's no overlap
    if np.any(train_mask & eval_mask):
        raise ValueError("Duplicate trial(s) found in both `train_split` and `eval_split`. Unable to merge...")

    tolist = lambda x: x if isinstance(x, list) else [x]
    train_eval_split = tolist(train_split) + tolist(eval_split)
    train_eval_mask = _prep_mask(dataset, train_eval_split)
    num_tot = train_eval_mask.sum()
    train_idx = np.arange(num_tot)[train_mask[train_eval_mask]]
    eval_idx = np.arange(num_tot)[eval_mask[train_eval_mask]]

    return _combine_dict(train_dict, eval_dict, train_idx, eval_idx)


def _combine_dict(train_dict, eval_dict, train_idx, eval_idx):
    """
    Recursive helper function that combines dict of tensors from two splits
    into one tensor using provided indices.
    """
    combine_dict_ = {}
    for key, val in train_dict.items():
        if isinstance(val, dict):
            if key not in eval_dict:
                logger.warning(f'{key} not found in `eval_dict`, skipping...')
                continue
            combine_dict_[key] = _combine_dict(
                train_dict[key],
                eval_dict[key],
                train_idx,
                eval_idx
            )
        else:
            eval_key = key.replace('train', 'eval')
            if eval_key not in eval_dict:
                logger.warning(f"{eval_key} not found in `eval_dict`, skipping...")
                continue
            train_arr = val
            eval_arr = eval_dict[eval_key]

            if train_arr.shape[1] != eval_arr.shape[1]:
                raise ValueError(f"Trial lengths for {key} and {eval_key} don't match")
            if train_arr.shape[2] != eval_arr.shape[2]:
                raise ValueError(f"Number of channels for {key} and {eval_key} don't match")

            full_arr = np.empty(
                (train_arr.shape[0] + eval_arr.shape[0], train_arr.shape[1], train_arr.shape[2])
            )
            full_arr[train_idx] = train_arr
            full_arr[eval_idx] = eval_arr
            combine_dict_[key] = full_arr
    return combine_dict_


def _prep_mask(dataset, trial_split):
    """
    Converts a trial split specification (string, boolean mask, or list) 
    into a boolean array. Also combines multiple splits if a list is provided.
    """
    def split_to_mask(s):
        if isinstance(s, str):
            return dataset.trial_info.split == s
        return s

    if isinstance(trial_split, list):
        masks = [split_to_mask(s) for s in trial_split]
        trial_mask = np.any(masks, axis=0)
    else:
        trial_mask = split_to_mask(trial_split)
    return trial_mask


def _prep_behavior(dataset, lag, make_params):
    """
    Helper function that returns new make_params for behavioral data,
    optionally shifting align_range by `lag`.
    """
    behavior_make_params = make_params.copy()
    if lag is not None:
        behavior_make_params['allow_nans'] = True
        if 'align_range' in behavior_make_params:
            behavior_make_params['align_range'] = tuple(t + lag for t in make_params['align_range'])
        else:
            # If align_range wasn't specified, define a trivial range just to get data
            behavior_make_params['align_range'] = (lag, lag)
    else:
        behavior_make_params = None
    return behavior_make_params


def _prep_fp(make_params, fp_steps, bin_width_ms):
    """
    Helper function that returns new make_params for forward-prediction data,
    appending extra time after the trial end.
    """
    align_point = make_params.get('align_field', make_params.get('end_field', 'end_time'))
    align_start = make_params.get('align_range', (0, 0))[1]
    align_window = (align_start, align_start + fp_steps * bin_width_ms)

    fp_make_params = {
        'align_field': align_point,
        'align_range': align_window,
        'allow_overlap': True,
    }
    return fp_make_params


def _prep_jitter(dataset, trial_mask, jitter):
    """
    Helper function that randomly chooses jitter values for each trial
    (in ms) given a 2-column array or callable.
    """
    trial_info = dataset.trial_info[trial_mask]

    if callable(jitter):
        jitter_range = jitter(trial_info)
    elif isinstance(jitter, (list, tuple)) and len(jitter) == 2:
        jitter_range = np.tile(np.array(jitter), (len(trial_info), 1))
    elif isinstance(jitter, np.ndarray):
        assert jitter.shape == (len(trial_info), 2), (
            f"Error: `jitter` array shape is incorrect; "
            f"provided shape: {jitter.shape}, expected shape: ({len(trial_info)}, 2)"
        )
        jitter_range = jitter
    else:
        logger.error("Unrecognized type for argument `jitter`")
        return np.zeros(len(trial_info))

    # Convert ms values into bins, then sample
    jitter_range = np.floor((jitter_range / dataset.bin_width).round(4))

    def sample(x):
        return np.random.random() * (x[1] - x[0]) + x[0]

    jitter_vals = np.apply_along_axis(sample, 1, jitter_range).round()
    return jitter_vals * dataset.bin_width


def _make_psth(dataset, train_mask, eval_mask, ignore_mask,
               make_params, cond_fields, kern_sd, pad='back',
               psth_len=None, seg_len=None, skip_mask=None):
    """
    Computes PSTHs for each condition in `cond_fields`.
    This function can be slow and memory-intensive for large data.
    """
    bin_width = dataset.bin_width
    ti = dataset.trial_info
    neur = dataset.data[['spikes', 'heldout_spikes']].columns

    # Reload minimal dataset with only spikes so we can smooth & resample
    dataset = NWBDataset(dataset.fpath, dataset.prefix,
                         skip_fields=[
                             'force', 'hand_pos', 'hand_vel',
                             'finger_pos', 'finger_vel', 'eye_pos',
                             'cursor_pos', 'muscle_len', 'muscle_vel',
                             'joint_ang', 'joint_vel'
                         ])
    dataset.trial_info = ti
    dataset.data = dataset.data.loc[:, neur]

    dataset.smooth_spk(kern_sd, signal_type=['spikes', 'heldout_spikes'],
                       overwrite=True, ignore_nans=True)
    if bin_width != 1:
        dataset.resample(bin_width)

    if skip_mask is not None:
        if callable(skip_mask):
            skip_mask = skip_mask(dataset.trial_info)
    else:
        skip_mask = np.full(len(dataset.trial_info), False)

    if isinstance(cond_fields, str):
        cond_fields = [cond_fields]

    # Unique condition combos
    combos = (
        dataset.trial_info[~ignore_mask][cond_fields]
        .dropna()
        .set_index(cond_fields).index.unique().tolist()
    )
    combos = sorted(combos)

    # Trial IDs for train/eval
    align_key = make_params.get('align_field', 'start_time')
    train_trial_ids = (
        dataset.trial_info[train_mask][['trial_id', align_key]]
        .dropna().trial_id.to_numpy()
    )
    eval_trial_ids = (
        dataset.trial_info[eval_mask][['trial_id', align_key]]
        .dropna().trial_id.to_numpy()
    )

    psth_list = []
    train_ids_list = []
    eval_ids_list = []
    remove_combs = []

    for comb in combos:
        # Find trials in this combination
        mask = np.all(dataset.trial_info[cond_fields] == comb, axis=1)
        if not np.any(mask & (~ignore_mask) & (~skip_mask)):
            logger.warning(f"No matching trials found for {comb}. Dropping.")
            remove_combs.append(comb)
            continue

        # Make trial data
        trial_data = dataset.make_trial_data(
            ignored_trials=(~mask | ignore_mask | skip_mask),
            allow_nans=True,
            **make_params
        )

        # Remove extremely short or outlier trials (heuristic)
        mean_len = np.mean([g.shape[0] for _, g in trial_data.groupby('trial_id')])
        tlens = {tid: grp.shape[0] for tid, grp in trial_data.groupby('trial_id')}
        bad_tid = [tid for tid in tlens if tlens[tid] < (0.8 * mean_len)]
        trial_data = trial_data[~np.isin(trial_data.trial_id, bad_tid)]

        # Compute PSTH by grouping over time
        min_len = np.min([g.shape[0] for _, g in trial_data.groupby('trial_id')])
        psth = trial_data.groupby('align_time')[['spikes', 'heldout_spikes']].mean().to_numpy()

        if pad == 'back':
            psth = psth[:min_len]
        else:  # pad == 'front'
            psth = psth[-min_len:]

        # Indices of train/eval trials in this condition
        curr_ids = trial_data.trial_id.unique()
        train_ids = np.sort(np.where(np.isin(train_trial_ids, curr_ids))[0])
        eval_ids = np.sort(np.where(np.isin(eval_trial_ids, curr_ids))[0])

        psth_list.append(psth)
        train_ids_list.append(train_ids)
        eval_ids_list.append(eval_ids)

    # Stack PSTHs to 3D array
    max_len = np.max([p.shape[0] for p in psth_list]) \
        if seg_len is None else int(round(seg_len / dataset.bin_width))

    def pad_psth(p):
        if p.shape[0] == max_len:
            return p
        elif p.shape[0] > max_len:
            return p[:max_len] if pad == 'back' else p[-max_len:]
        else:
            if pad == 'back':
                return np.vstack([p, np.full((max_len - p.shape[0], p.shape[1]), np.nan)])
            else:
                return np.vstack([np.full((max_len - p.shape[0], p.shape[1]), np.nan), p])

    psth_list = [pad_psth(p) for p in psth_list]

    # Filter out combos with no trials
    good_combos = [c for c in combos if c not in remove_combs]
    psth_stack = np.stack(psth_list)
    train_ids_list = np.array(train_ids_list, dtype='object')
    eval_ids_list = np.array(eval_ids_list, dtype='object')

    return (train_ids_list, eval_ids_list), psth_stack, good_combos


''' Tensor saving functions '''
def save_to_h5(data_dict, save_path, overwrite=False, dlen=32, compression="gzip"):
    """
    Saves a nested dictionary of arrays to an HDF5 file.

    Parameters
    ----------
    data_dict : dict
        Dict containing data to be saved
    save_path : str
        Path to the HDF5 output file
    overwrite : bool, optional
        Whether to overwrite existing datasets/groups in the file, by default False
    dlen : int, optional
        Byte length of numeric data format, by default 32
    compression : str, optional
        Compression to use when writing to HDF5, by default "gzip"
    """
    with h5py.File(save_path, 'a') as h5file:
        good, dup_list = _check_h5_r(data_dict, h5file, overwrite)
        if good:
            if len(dup_list) > 0:
                logger.warning(f"{dup_list} already found in {save_path}. Overwriting...")
            _save_h5_r(data_dict, h5file, dlen, compression)
            logger.info(f"Saved data to {save_path}")
        else:
            logger.warning(
                f"{dup_list} already found in {save_path}. "
                "Save canceled. Set `overwrite=True` or use a different path."
            )


def _check_h5_r(data_dict, h5obj, overwrite):
    """
    Recursive helper that checks for duplicate keys in h5obj and deletes them if `overwrite == True`.
    Returns (good, dup_list) where:
        good : bool indicating if we can safely write data
        dup_list : any duplicate paths found
    """
    dup_list = []
    good = True
    for key in data_dict.keys():
        if key in h5obj.keys():
            if isinstance(h5obj[key], h5py.Group) and isinstance(data_dict[key], dict):
                rgood, rdup_list = _check_h5_r(data_dict[key], h5obj[key], overwrite)
                good = good and rgood
                dup_list += list(zip([key] * len(rdup_list), rdup_list))
            else:
                dup_list.append(key)
                if overwrite:
                    del h5obj[key]
                else:
                    good = False
    return good, dup_list


def _save_h5_r(data_dict, h5obj, dlen, compression="gzip"):
    """
    Recursive helper that writes all items in a dict to an h5py.File or h5py.Group object.
    """
    for key, val in data_dict.items():
        if isinstance(val, dict):
            if key in h5obj.keys():
                h5group = h5obj[key]
            else:
                h5group = h5obj.create_group(key)
            _save_h5_r(val, h5group, dlen, compression)
        else:
            # Handle dtype
            if val.dtype == 'object':
                sub_dtype = val[0].dtype
                if dlen is not None:
                    if np.issubdtype(sub_dtype, np.floating):
                        sub_dtype = f'float{dlen}'
                    elif np.issubdtype(sub_dtype, np.integer):
                        sub_dtype = f'int{dlen}'
                dtype = h5py.vlen_dtype(sub_dtype)
            else:
                dtype = val.dtype
                if dlen is not None:
                    if np.issubdtype(dtype, np.floating):
                        dtype = f'float{dlen}'
                    elif np.issubdtype(dtype, np.integer):
                        dtype = f'int{dlen}'

            h5obj.create_dataset(key, data=val, dtype=dtype, compression=compression)


def h5_to_dict(h5obj):
    """
    Recursive function that reads an HDF5 file or group into a nested Python dict.
    """
    data_dict = {}
    for key in h5obj.keys():
        if isinstance(h5obj[key], h5py.Group):
            data_dict[key] = h5_to_dict(h5obj[key])
        else:
            data_dict[key] = h5obj[key][()]
    return data_dict


def combine_h5(file_paths, save_path=None):
    """
    Combines multiple HDF5 files into one, merging top-level keys
    and overwriting if needed.

    Parameters
    ----------
    file_paths : list
        List of paths to HDF5 files to combine
    save_path : str, optional
        Path to save combined results. By default, overwrites first file in `file_paths`.
    """
    assert len(file_paths) > 1, "Must provide at least 2 files to combine"
    if save_path is None:
        save_path = file_paths[0]

    for fpath in file_paths:
        if fpath == save_path:
            continue
        with h5py.File(fpath, 'r') as h5file:
            data_dict = h5_to_dict(h5file)
        save_to_h5(data_dict, save_path)


''' Tensor chopping convenience functions '''
def chop_tensors(fpath, window, overlap, chop_fields=None, save_path=None):
    """
    Chops a tensor .h5 file directly without loading into NWBDataset.

    Parameters
    ----------
    fpath : str
        Path to HDF5 file containing data to chop
    window : int
        Length of chop window
    overlap : int
        Overlap shared between chop windows
    chop_fields : list of str, optional
        Fields to chop. By default None chops all fields
    save_path : str, optional
        Path to save chopped data to. By default None
        creates a new file with "_chopped" in the name.
    """
    with h5py.File(fpath, 'r') as h5file:
        if chop_fields is None:
            chop_fields = sorted(list(h5file.keys()))

        data_list = []
        for field in chop_fields:
            arr = h5file[field][()]
            if len(arr.shape) < 3:
                # At least trial x time x channel
                if len(arr.shape) == 2:
                    arr = arr[:, :, None]
                else:
                    arr = arr[:, None, None]
            data_list.append(arr)

    # Default save path
    if save_path is None:
        base, ext = os.path.splitext(fpath)
        save_path = f"{base}_chopped.h5"
    elif not save_path.endswith('.h5'):
        save_path += '.h5'

    # Chop
    splits = np.cumsum([arr.shape[2] for arr in data_list[:-1]])
    data = np.dstack(data_list)

    chop_list = []
    stride = window - overlap
    discard = (data.shape[1] - window) % stride
    if discard != 0:
        logger.warning(
            f"With window={window} and overlap={overlap}, {discard} samples will be discarded per trial."
        )

    for i in range(data.shape[0]):
        seg = data[i, :, :]
        chop_list.append(chop_data(seg, overlap, window))

    chopped_data = np.vstack(chop_list)
    chopped_data = np.split(chopped_data, splits, axis=2)
    data_dict = {chop_fields[i]: chopped_data[i] for i in range(len(chop_fields))}

    save_to_h5(data_dict, save_path, overwrite=True)


def merge_tensors(fpath, window, overlap, orig_len, merge_fields=None, save_path=None):
    """
    Merges a chopped tensor .h5 file back to its original shape (assuming a fixed segment length).

    Parameters
    ----------
    fpath : str
        Path to HDF5 file containing chopped data
    window : int
        Length of chop window
    overlap : int
        Overlap shared between chop windows
    orig_len : int
        Original length of trial segments before chopping
    merge_fields : list of str, optional
        Fields to merge. By default None merges all fields
    save_path : str, optional
        Path to save merged data to. By default None
        creates a new file with "_merged" in the name.
    """
    with h5py.File(fpath, 'r') as h5file:
        if merge_fields is None:
            merge_fields = sorted(list(h5file.keys()))

        data_list = []
        for field in merge_fields:
            arr = h5file[field][()]
            data_list.append(arr)

    # Default save path
    if save_path is None:
        base, ext = os.path.splitext(fpath)
        save_path = f"{base}_merged.h5"
    elif not save_path.endswith('.h5'):
        save_path += '.h5'

    data = np.dstack(data_list)
    # How many chops per segment?
    stride = window - overlap
    chop_per_seg = int(round((orig_len - overlap) / stride))

    splits = np.cumsum([arr.shape[2] for arr in data_list[:-1]])
    merge_list = []

    for i in range(0, data.shape[0], chop_per_seg):
        chopped_seg = data[i:(i + chop_per_seg), :, :]
        merge_list.append(merge_chops(chopped_seg, overlap, orig_len))

    merge_data = np.stack(merge_list)
    merge_data = np.split(merge_data, splits, axis=2)

    data_dict = {merge_fields[i]: merge_data[i] for i in range(len(merge_fields))}
    save_to_h5(data_dict, save_path, overwrite=True)
