"""CIFAR-10 training and validation loaders using the project's YAML config.

The official test split is used for validation, as in the previous framework.
Set download: true to fetch missing CIFAR-10 files into data. In DDP, download
and integrity checks run one rank at a time to avoid simultaneous extraction
on shared storage; this also supports separate dataset directories per node.
"""

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset

from dataset.common import distributed_context, loader_options


NUM_CLASSES = 10


def get_train_loader(conf):
    dataset = _build_dataset(conf, train=True)
    distributed, rank, world_size = distributed_context(conf)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, seed=conf['seed'] if conf['seed'] is not None else 0)
    return DataLoader(dataset, shuffle=sampler is None, sampler=sampler,
                      **loader_options(conf, distributed))


def get_valid_loader(conf):
    dataset = _build_dataset(conf, train=False)
    distributed, rank, world_size = distributed_context(conf)
    if distributed:
        dataset = Subset(dataset, range(rank, len(dataset), world_size))
    return DataLoader(dataset, shuffle=False, **loader_options(conf, distributed))


def _build_dataset(conf, train):
    from torchvision import datasets, transforms

    size = conf['input_size']
    if conf['dummy']:
        return datasets.FakeData(50000 if train else 10000, (3, size, size),
                                 NUM_CLASSES, transforms.ToTensor())
    augmentation = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
    transform = transforms.Compose(augmentation + [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # Keep the normalization used by the previous CIFAR-10 loader.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if conf['download'] and dist.is_available() and dist.is_initialized():
        dataset = None
        for rank in range(dist.get_world_size()):
            if rank == dist.get_rank():
                dataset = datasets.CIFAR10(conf['data'], train=train, download=True, transform=transform)
            dist.barrier()
        return dataset
    return datasets.CIFAR10(conf['data'], train=train, download=conf['download'], transform=transform)
