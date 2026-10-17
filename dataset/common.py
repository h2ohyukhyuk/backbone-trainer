"""Shared DataLoader options and distributed settings."""

import random

import torch
import torch.distributed as dist


def distributed_context(conf):
    if dist.is_available() and dist.is_initialized():
        return True, dist.get_rank(), dist.get_world_size()
    if conf.get('distributed', False):
        global_rank, total_processes = conf['global_rank'], conf['total_processes']
        if total_processes <= 0 or not 0 <= global_rank < total_processes:
            raise ValueError('Distributed loaders require total_processes > 0 and 0 <= global_rank < total_processes')
        return True, global_rank, total_processes
    return False, 0, 1


def loader_options(conf, distributed):
    batch_size, workers = conf['batch_size'], conf['workers']
    if distributed:
        processes_per_node = conf.get('processes_per_node', 1)
        if processes_per_node <= 0 or batch_size % processes_per_node:
            raise ValueError('batch_size must be divisible by the number of local processes')
        batch_size //= processes_per_node
        workers = (workers + processes_per_node - 1) // processes_per_node
    return dict(batch_size=batch_size, num_workers=workers,
                pin_memory=not conf['no_accel'] and torch.cuda.is_available(),
                worker_init_fn=seed_worker if conf['seed'] is not None else None)


def seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2 ** 32))
    import numpy as np
    np.random.seed(torch.initial_seed() % (2 ** 32))
