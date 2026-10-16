"""Shared DataLoader options and distributed settings."""

import random

import torch
import torch.distributed as dist


def distributed_context(conf):
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_rank(), dist.get_world_size()
    if conf.get('distributed', False):
        rank, world_size = conf['rank'], conf['world_size']
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError('Distributed loaders require world_size > 0 and 0 <= rank < world_size')
        return True, rank, world_size
    return False, 0, 1


def loader_options(conf, distributed):
    batch_size, workers = conf['batch_size'], conf['workers']
    if distributed:
        local_world_size = conf.get('local_world_size', 1)
        if local_world_size <= 0 or batch_size % local_world_size:
            raise ValueError('batch_size must be divisible by the number of local processes')
        batch_size //= local_world_size
        workers = (workers + local_world_size - 1) // local_world_size
    return dict(batch_size=batch_size, num_workers=workers,
                pin_memory=not conf['no_accel'] and torch.cuda.is_available(),
                worker_init_fn=seed_worker if conf['seed'] is not None else None)


def seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2 ** 32))
    import numpy as np
    np.random.seed(torch.initial_seed() % (2 ** 32))
