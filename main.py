"""Classification training configured through config_default.yaml.

Examples (from the project directory):
    python main.py
    python main.py --config_path experiments/APAGNet.yml
    python main.py --config_path experiments/APAGNet_cifar10.yml
    torchrun --standalone --nproc_per_node=2 main.py --config_path experiments/APAGNet.yml

All training settings live in YAML; only the config file is selected on the CLI.
An experiment file overrides experiments/config_default.yaml in this project.
Relative data/save/resume paths are relative to the current working directory.
"""

import argparse
import importlib
import logging
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim.lr_scheduler import StepLR
import yaml

from dataset import SUPPORTED_DATASETS, get_dataset_module
from train_val.classification import Trainer, Validationer, unwrap_model


def read_config(path):
    with Path(path).open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not all(isinstance(key, str) for key in config):
        raise ValueError(f'{path}: YAML must contain a mapping with string keys')
    return config


def validate_config(config):
    """Validate settings without defining any default values in Python."""
    types = {
        str: ('dataset', 'data', 'arch', 'save', 'resume', 'dist_url', 'dist_backend'),
        int: ('workers', 'epochs', 'start_epoch', 'batch_size', 'print_freq',
              'world_size', 'rank', 'seed', 'gpu', 'lr_step_size', 'input_size'),
        float: ('lr', 'momentum', 'weight_decay', 'lr_gamma'),
        bool: ('evaluate', 'pretrained', 'no_accel', 'multiprocessing_distributed', 'dummy', 'download'),
        dict: ('model_kwargs',),
    }
    expected_keys = {name for names in types.values() for name in names}
    missing, unknown = expected_keys - config.keys(), config.keys() - expected_keys
    if missing or unknown:
        raise ValueError(f'Configuration keys: missing={sorted(missing)}, unknown={sorted(unknown)}')
    for expected_type, names in types.items():
        for name in names:
            value = config[name]
            if name in ('seed', 'gpu') and value is None:
                continue
            if expected_type is float:
                if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                    raise ValueError(f'{name} must be a number')
                try:
                    config[name] = float(value)
                except ValueError as exc:
                    raise ValueError(f'{name} must be a number') from exc
            elif type(value) is not expected_type:
                raise ValueError(f'{name} must be {expected_type.__name__}')
    if not all(isinstance(key, str) for key in config['model_kwargs']):
        raise ValueError('model_kwargs must contain string keys')
    if config['dataset'] not in SUPPORTED_DATASETS:
        raise ValueError(f'dataset must be one of {SUPPORTED_DATASETS}')
    for name in ('batch_size', 'print_freq', 'lr_step_size', 'input_size'):
        if config[name] <= 0:
            raise ValueError(f'{name} must be positive')
    for name in ('workers', 'epochs', 'start_epoch', 'gpu'):
        if config[name] is not None and config[name] < 0:
            raise ValueError(f'{name} cannot be negative')
    if not config['data']:
        raise ValueError('data requires a path')
    if config['start_epoch'] > config['epochs']:
        raise ValueError('start_epoch cannot exceed epochs')


def parse_args(argv=None):
    default_path = Path(__file__).resolve().parent / 'experiments' / 'config_default.yaml'
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config_path', '--config-path', type=Path, default=default_path,
                        help='experiment YAML to overlay on config_default.yaml')
    args = parser.parse_args(argv)
    try:
        config = read_config(default_path)
        validate_config(config)
        if args.config_path.resolve() != default_path:
            overrides = read_config(args.config_path)
            unknown = overrides.keys() - config.keys()
            if unknown:
                raise ValueError(f'Unknown YAML keys: {sorted(unknown)}')
            config.update(overrides)
            validate_config(config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return argparse.Namespace(config_path=str(args.config_path), **config)


def local_models():
    """Discover exact filenames plus short aliases: APAGNet -> apag."""
    result = {}
    for path in sorted((Path(__file__).resolve().parent / 'net').glob('*.py')):
        if not path.stem.startswith('_'):
            result[path.stem] = path.stem
            result.setdefault(path.stem.lower(), path.stem)
            if path.stem.endswith('Net'):
                result.setdefault(path.stem[:-3].lower(), path.stem)
    return result


def create_model(args):
    num_classes = get_dataset_module(args.dataset).NUM_CLASSES
    kwargs = dict(args.model_kwargs)
    if 'num_classes' in kwargs and kwargs['num_classes'] != num_classes:
        raise ValueError(f'{args.dataset} requires num_classes={num_classes}')
    kwargs['num_classes'] = num_classes
    module_name = local_models().get(args.arch)
    if module_name:
        if args.pretrained:
            raise ValueError('Local models use resume for weights; pretrained is for torchvision')
        module = importlib.import_module(f'net.{module_name}')
        return module.Network(**kwargs)

    from torchvision import models

    constructor = getattr(models, args.arch, None)
    if constructor is None or not callable(constructor):
        raise ValueError(f'Unknown architecture {args.arch!r}; check torchvision or net/*.py')
    if args.pretrained and num_classes != 1000:
        raise ValueError('pretrained: true currently requires ImageNet; use pretrained: false for CIFAR-10')
    return constructor(weights='DEFAULT' if args.pretrained else None, **kwargs)


def main_ddp():
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_accel
    gpus_per_node = torch.cuda.device_count() if use_cuda else 0
    launched = 'LOCAL_RANK' in os.environ
    if launched and args.multiprocessing_distributed:
        raise ValueError('Set multiprocessing_distributed: false when using torchrun')

    if launched:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        args.gpu = int(os.environ['LOCAL_RANK']) if use_cuda else None
        args.dist_url = 'env://'
    elif args.dist_url == 'env://':
        if args.world_size == -1:
            args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        if args.rank == -1:
            args.rank = int(os.environ.get('RANK', '0'))

    args.distributed = launched or args.world_size > 1 or args.multiprocessing_distributed
    if args.distributed and not use_cuda and args.dist_backend == 'nccl':
        raise ValueError('CPU distributed training requires dist_backend: gloo')

    if args.multiprocessing_distributed:
        if not use_cuda:
            raise ValueError('multiprocessing_distributed requires CUDA; use torchrun for CPU DDP')
        if args.gpu is not None:
            raise ValueError('Set gpu: null with multiprocessing_distributed: true')
        # This launch mode follows the example: world_size/rank refer to nodes.
        args.world_size = max(args.world_size, 1) * gpus_per_node
        args.rank = max(args.rank, 0)
        args.local_world_size = gpus_per_node
        mp.spawn(main_worker, nprocs=gpus_per_node, args=(args,))
    else:
        local_processes = gpus_per_node if args.distributed and use_cuda else 1
        args.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', str(local_processes)))
        if args.distributed and args.rank < 0:
            raise ValueError('Distributed training requires rank in YAML or RANK from torchrun')
        if args.distributed and use_cuda and args.gpu is None:
            raise ValueError('Distributed training requires gpu in YAML or LOCAL_RANK from torchrun')
        main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    if args.multiprocessing_distributed:
        args.rank = args.rank * args.local_world_size + gpu
    use_cuda = torch.cuda.is_available() and not args.no_accel
    device = torch.device('cuda', gpu if gpu is not None else 0) if use_cuda else torch.device('cpu')
    if use_cuda:
        torch.cuda.set_device(device)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    try:
        main(args, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(args, device):
    is_main = not args.distributed or args.rank == 0
    dataset_tag = 'imgn' if args.dataset == 'image_net_1K' else args.dataset
    suffix = f'-{dataset_tag}' + ('-dummy' if args.dummy else '')
    save_dir = Path(args.save) / (args.arch + suffix)
    handlers = [logging.StreamHandler(sys.stdout)]
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(save_dir / f'log-{args.arch}{suffix}.txt', mode='a'))
        config = {key: value for key, value in vars(args).items()
                  if key not in ('config_path', 'distributed', 'local_world_size')}
        if args.multiprocessing_distributed:
            config['world_size'] //= args.local_world_size
            config['rank'] //= args.local_world_size
        with (save_dir / 'config.resolved.yml').open('w', encoding='utf-8') as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format='%(message)s', handlers=handlers, force=True)
    logging.info('Arguments: %s', vars(args))

    model = create_model(args).to(device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu] if device.type == 'cuda' else None)
    elif device.type == 'cuda' and args.gpu is None and torch.cuda.device_count() > 1:
        model = DataParallel(model)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    best_acc1 = load_checkpoint(args, model, optimizer, scheduler)
    conf = vars(args)
    dataset_module = get_dataset_module(args.dataset)
    train_loader = dataset_module.get_train_loader(conf)
    val_loader = dataset_module.get_valid_loader(conf)
    trainer = Trainer(conf, train_loader, optimizer, scheduler)
    validationer = Validationer(conf, optimizer=optimizer, scheduler=scheduler, best_acc=best_acc1)
    if args.evaluate:
        validationer(model, args.start_epoch, val_loader)
        return

    started = time.perf_counter()
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        epoch_started = time.perf_counter()
        trainer(model, epoch)
        validationer(model, epoch, val_loader)
        logging.info('Epoch %d: %.2f minutes; best Acc@1 %.3f', epoch,
                     (time.perf_counter() - epoch_started) / 60, validationer.best_acc)
    logging.info('Total training time: %.2f hours', (time.perf_counter() - started) / 3600)


def load_checkpoint(args, model, optimizer, scheduler):
    if not args.resume:
        return 0.0
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
    state = checkpoint.get('state_dict', checkpoint.get('model'))
    if state is None:
        raise ValueError('Checkpoint must contain state_dict (ImageNet) or model (legacy framework)')
    # Accept older DataParallel/DDP checkpoints while saving new ones without wrapper prefixes.
    state = {key.removeprefix('module.').replace('features.module.', 'features.', 1): value
             for key, value in state.items()}
    unwrap_model(model).load_state_dict(state)
    if all(key in checkpoint for key in ('epoch', 'optimizer', 'scheduler', 'best_acc1')):
        args.start_epoch = int(checkpoint['epoch'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logging.info('Resumed %s at epoch %d', args.resume, args.start_epoch)
        return float(checkpoint['best_acc1'])
    logging.warning('Loaded weights only from %s; optimizer/scheduler/epoch start fresh', args.resume)
    return 0.0


if __name__ == '__main__':
    main_ddp()
