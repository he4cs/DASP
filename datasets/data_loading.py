import os
import logging
import random
import numpy as np
import torch

from conf import complete_data_dir_path
from datasets.corruption_datasets import create_audiosetc_dataset


logger = logging.getLogger(__name__)


def get_test_loader(setting: str, adaptation: str, dataset_name: str, data_root_dir: str, domain_name: str, 
                    domain_names_all: list, severity: int, num_examples: int,rng_seed: int, batch_size: int = 128, 
                    shuffle: bool = False, workers: int = 4):
    """
    Create the test data loader
    Input:
        setting: Name of the considered setting
        adaptation: Name of the adaptation method
        dataset_name: Name of the dataset
        data_root_dir: Path of the data root directory
        domain_name: Name of the current domain
        domain_names_all: List containing all domains
        severity: Severity level in case of corrupted data
        num_examples: Number of test samples for the current domain
        rng_seed: A seed number
        batch_size: The number of samples to process in each iteration
        shuffle: Whether to shuffle the data. Will destroy pre-defined settings
        workers: Number of workers used for data loading
    Returns:
        test_loader: The test data loader
    """

    # Fix seed again to ensure that the test sequence is the same for all methods
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    # create the test dataset
    if dataset_name in ["vggsound", "ks50"]:
        data_dir = complete_data_dir_path(data_root_dir, dataset_name)
        test_dataset = create_audiosetc_dataset(dataset_name=dataset_name,
                                                data_dir=data_dir,
                                                corruption=domain_name,
                                                corruption_seq=domain_names_all,
                                                severity=severity,
                                                setting=setting)
    
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported!")

    try:
        # shuffle the test sequence; deterministic behavior for a fixed random seed
        random.shuffle(test_dataset.samples)

        # randomly subsample the dataset if num_examples is specified
        if num_examples != -1:
            num_samples_orig = len(test_dataset)
            # logger.info(f"Changing the number of test samples from {num_samples_orig} to {num_examples}...")
            test_dataset.samples = random.sample(test_dataset.samples, k=min(num_examples, num_samples_orig))
    
    except AttributeError:
        logger.warning("Attribute 'samples' is missing. Continuing without shuffling or subsampling the files...")
    
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True, drop_last=False)