"""ImageNet-1K loaders using the project's flat YAML configuration.

Both public functions take a configuration dictionary and return a DataLoader.
The dataset root must contain train/<class>/* and val/<class>/*.
main.py supplies distributed, global_rank, total_processes and processes_per_node at runtime.
batch_size and workers are per-node settings and are split among local DDP
processes without changing the configuration dictionary.
"""

from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler, Subset

from dataset.augmentation import build_train_transform
from dataset.common import distributed_context, loader_options


NUM_CLASSES = 1000


def get_train_loader(conf):
    """Build the training loader with ImageNet augmentation and a DDP sampler."""
    from torchvision import datasets, transforms

    size = conf['input_size']
    if conf['dummy']:
        dataset = datasets.FakeData(1281167, (3, size, size), NUM_CLASSES, transforms.ToTensor())
    else:
        dataset = datasets.ImageFolder(Path(conf['data']) / 'train', build_train_transform(conf))
    distributed, global_rank, total_processes = distributed_context(conf)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=total_processes, rank=global_rank,
                                     shuffle=True, seed=conf['seed'] if conf['seed'] is not None else 0)
    return DataLoader(dataset, shuffle=sampler is None, sampler=sampler,
                      **loader_options(conf, distributed))


def get_valid_loader(conf):
    """Build the validation loader; DDP shards contain no duplicated samples."""
    from torchvision import datasets, transforms

    size = conf['input_size']
    if conf['dummy']:
        dataset = datasets.FakeData(50000, (3, size, size), NUM_CLASSES, transforms.ToTensor())
    else:
        dataset = datasets.ImageFolder(Path(conf['data']) / 'val', transforms.Compose([
            transforms.Resize(int(size * 256 / 224)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]))
    distributed, global_rank, total_processes = distributed_context(conf)
    if distributed:
        dataset = Subset(dataset, range(global_rank, len(dataset), total_processes))
    return DataLoader(dataset, shuffle=False, **loader_options(conf, distributed))
