import sys
sys.path.append("../../")


if __name__ == "__main__":

    from Utils.directory_manager import DirectoryManager

    dataset_paths = {
        'mc_maze_small': {
            'train': '/my_data/NEURAL_LATENTS/MC_MAZE/small/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb',
            'test': '/my_data/NEURAL_LATENTS/MC_MAZE/small/000140/sub-Jenkins/sub-Jenkins_ses-small_desc-test_ecephys.nwb'
        }
    }

    dm = DirectoryManager(dataset_paths)

    from DataFetchers.MC_MAZE.mc_maze_loader import fetch_mc_maze

    train_data, eval_data, target_eval_data, final_train_data, test_data = fetch_mc_maze(dm, "small")

    print("train_data.keys() = ", train_data.keys())

