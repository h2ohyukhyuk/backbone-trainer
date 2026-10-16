"""Dataset loaders for classification experiments."""

import importlib


SUPPORTED_DATASETS = ('image_net_1K', 'cifar10')


def get_dataset_module(name):
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f'Unknown dataset {name!r}; choose from {SUPPORTED_DATASETS}')
    return importlib.import_module(f'dataset.{name}')
