class DirectoryManager:
    # Define available datasets
    AVAILABLE_DATASETS = {
        'mc_maze': {'train': '', 'test': ''},
        'mc_maze_small': {'train': '', 'test': ''},
        'mc_maze_medium': {'train': '', 'test': ''},
        'mc_maze_large': {'train': '', 'test': ''},
        # to be added
    }

    def __init__(self, dataset_paths=None):
        """
        Initialize DirectoryManager with dataset paths.
        
        Args:
            dataset_paths (dict): Dictionary mapping dataset names to their required paths
                Example: {
                    'imagenet': {
                        'train': '/path/to/train',
                        'test': '/path/to/test'
                    }
                }
        """
        self.dataset_paths = {}
        if dataset_paths:
            self.set_directories(dataset_paths)

    def set_directories(self, dataset_paths):
        """
        Set or update the paths for specified datasets.
        
        Args:
            dataset_paths (dict): Dictionary mapping dataset names to their required paths
        """
        for dataset_name, paths in dataset_paths.items():
            if dataset_name not in self.AVAILABLE_DATASETS:
                raise ValueError(f"Dataset '{dataset_name}' is not supported. "
                              f"Available datasets: {list(self.AVAILABLE_DATASETS.keys())}")
            
            required_paths = self.AVAILABLE_DATASETS[dataset_name]
            missing_paths = set(required_paths) - set(paths.keys())
            
            if missing_paths:
                raise ValueError(f"Missing required paths for {dataset_name}: {missing_paths}")
            
            self.dataset_paths[dataset_name] = paths

    def get_paths(self, dataset_name, split):
        """
        Get the paths for a specific dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            dict: Dictionary containing the paths for the specified dataset
        """
        if dataset_name not in self.dataset_paths:
            raise ValueError(f"Paths for dataset '{dataset_name}' have not been set")
        return self.dataset_paths[dataset_name][split]