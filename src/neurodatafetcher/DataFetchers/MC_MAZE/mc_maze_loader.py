from neurodatafetcher.Utils.nlb_tools.nwb_interface import NWBDataset
from neurodatafetcher.Utils.nlb_tools.make_tensors import (
    make_train_input_tensors, make_eval_input_tensors, make_eval_target_tensors, save_to_h5
)
import numpy as np

from neurodatafetcher.Utils.directory_manager import DirectoryManager

def fetch_mc_maze(dm: DirectoryManager, dataset_name: str):
    """
    Fetch MC Maze dataset.

    Args:
        dm (DirectoryManager): Directory manager containing dataset paths
        dataset_name (str): Name of dataset, one of: "standard", "small", 
                          "medium", "large"
    """

    datasetName = f"mc_maze_{dataset_name}" if dataset_name != "standard" else "mc_maze"

    dataset_path_train = dm.get_paths(datasetName, "train")
    dataset_path_test = dm.get_paths(datasetName, "test")


    dataset_train = NWBDataset(dataset_path_train, split_heldout=True)

    dataset_test = NWBDataset(dataset_path_test, split_heldout=True)

    dataset_train.resample(5)
    dataset_test.resample(5)

    optimize_train_dict = make_train_input_tensors(
        dataset_train, dataset_name=datasetName, trial_split=["train"], save_file=False,
        include_behavior=True,
        include_forward_pred=True,
    )

    eval_dict = make_eval_input_tensors(
        dataset_train,
        dataset_name=datasetName,
        trial_split="val",
        save_file=False)

    target_eval_dict = make_eval_target_tensors(
        dataset_train, dataset_name=datasetName, train_trial_split="train", eval_trial_split="val", save_file=False, )

    final_train_dict = make_train_input_tensors(
        dataset_train, dataset_name=datasetName, trial_split=["train", "val"], save_file=False,
        include_behavior=True,
        include_forward_pred=True,
    )

    test_dict = make_eval_input_tensors(
        dataset_test,
        dataset_name=datasetName,
        trial_split="test",
        save_file=False)    


    return optimize_train_dict, eval_dict, target_eval_dict, final_train_dict, test_dict

