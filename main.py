"""Classification training configured through config_default.yaml.

Examples (from the project directory):
    python main.py
    python main.py --config_path experiments/config.yml
    python main.py --config_path experiments/APAGNet_cifar10.yml
    torchrun --standalone --nproc_per_node=2 main.py --config_path experiments/config.yml

Training settings live in YAML. Use --plot-lr to save an LR preview without training.
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
import yaml

from utils.train_utils import get_optimizer, get_scheduler, optimizer_kwargs
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
        str: ('dataset', 'data', 'arch', 'save', 'resume', 'resume_task_id', 'dist_url', 'dist_backend', 'lr_scheduler', 'OPT_NAME', 'amp_dtype'),
        int: ('workers', 'epochs', 'start_epoch', 'batch_size', 'print_freq',
              'num_nodes', 'node_rank', 'seed', 'gpu', 'input_size'),
        float: ('LR', 'MOMENTUM', 'WEIGHT_DECAY'),
        bool: ('evaluate', 'pretrained', 'no_accel', 'multiprocessing_distributed', 'dummy', 'download', 'AMS_GRAD', 'USE_LR_MULT', 'amp', 'fast_mode'),
        dict: ('model_kwargs', 'SGD_KWARGS', 'ADAM_KWARGS', 'ADAMW_KWARGS', 'RMSPROP_KWARGS', 'ADADELTA_KWARGS',
               'STEPLR_KWARGS', 'LAMBDALR_KWARGS', 'MULTISTEPLR_KWARGS', 'POLYLR_KWARGS', 'AUGMENTATION'),
        list: ('LR_MULT_TARGET', 'LR_MULT_VAL'),
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
    from dataset.augmentation import validate_augmentation
    validate_augmentation(config['AUGMENTATION'])
    if not all(isinstance(key, str) for key in config['model_kwargs']):
        raise ValueError('model_kwargs must contain string keys')
    if config['dataset'] not in SUPPORTED_DATASETS:
        raise ValueError(f'dataset must be one of {SUPPORTED_DATASETS}')
    for name in ('batch_size', 'print_freq', 'input_size'):
        if config[name] <= 0:
            raise ValueError(f'{name} must be positive')
    for name in ('workers', 'epochs', 'start_epoch', 'gpu'):
        if config[name] is not None and config[name] < 0:
            raise ValueError(f'{name} cannot be negative')
    if config['lr_scheduler'] not in ('steplr', 'lambdalr', 'multisteplr', 'polylr'):
        raise ValueError('lr_scheduler must be steplr, lambdalr, multisteplr or polylr')
    schemas = {
        'STEPLR_KWARGS': {'step_size': int, 'gamma': float},
        'LAMBDALR_KWARGS': {'points': dict},
        'MULTISTEPLR_KWARGS': {'milestones': list, 'gamma': float},
        'POLYLR_KWARGS': {'power': float},
    }
    for section, schema in schemas.items():
        settings = config[section]
        optional = {'interpolation', 'scale'} if section == 'LAMBDALR_KWARGS' else set()
        if section == 'LAMBDALR_KWARGS':
            if settings.keys() - (schema.keys() | optional):
                raise ValueError('Unknown LAMBDALR_KWARGS interpolation keys')
            from utils.lr_interpolation import epoch_lr_curve
            epoch_lr_curve(settings)
            continue
        if not schema.keys() <= settings.keys() or settings.keys() - (schema.keys() | optional):
            raise ValueError(f'{section} requires {sorted(schema)}')
        for key, expected in schema.items():
            value = settings[key]
            if expected is float:
                if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                    raise ValueError(f'{section}.{key} must be a number')
                try:
                    settings[key] = float(value)
                except ValueError as exc:
                    raise ValueError(f'{section}.{key} must be a number') from exc
            elif type(value) is not expected:
                raise ValueError(f'{section}.{key} must be {expected.__name__}')
    step = config['STEPLR_KWARGS']
    multi, poly = config['MULTISTEPLR_KWARGS'], config['POLYLR_KWARGS']
    if step['step_size'] <= 0 or not 0 < step['gamma'] <= 1:
        raise ValueError('STEPLR_KWARGS requires step_size > 0 and gamma in (0, 1]')
    milestones = multi['milestones']
    if (any(type(epoch) is not int or epoch <= 0 for epoch in milestones)
            or milestones != sorted(set(milestones))):
        raise ValueError('MULTISTEPLR_KWARGS.milestones must be strictly increasing positive integers')
    if not 0 < multi['gamma'] <= 1:
        raise ValueError('MULTISTEPLR_KWARGS.gamma must be in (0, 1]')
    if not 0 < poly['power'] < float('inf'):
        raise ValueError('POLYLR_KWARGS.power must be finite and positive')
    if config['amp_dtype'] not in ('float16', 'bfloat16'):
        raise ValueError('amp_dtype must be float16 or bfloat16')
    if config['OPT_NAME'] not in ('sgd', 'adam', 'adamw', 'rmsprop', 'adadelta'):
        raise ValueError('OPT_NAME must be sgd, adam, adamw, rmsprop or adadelta')
    if not 0 < config['LR'] < float('inf'):
        raise ValueError('LR must be finite and positive')
    if not 0 <= config['MOMENTUM'] < float('inf') or not 0 <= config['WEIGHT_DECAY'] < float('inf'):
        raise ValueError('MOMENTUM and WEIGHT_DECAY must be finite and nonnegative')
    targets, values = config['LR_MULT_TARGET'], config['LR_MULT_VAL']
    if len(targets) != len(values) or not all(isinstance(k, str) and k for k in targets):
        raise ValueError('LR_MULT_TARGET and LR_MULT_VAL require matching keyword/value lists')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v < float('inf') for v in values):
        raise ValueError('LR_MULT_VAL must contain finite positive numbers')
    allowed = {
        'sgd': {'dampening', 'nesterov'},
        'adam': {'betas', 'eps'}, 'adamw': {'betas', 'eps'},
        'rmsprop': {'alpha', 'eps', 'centered'}, 'adadelta': {'rho', 'eps'},
    }
    for name, keys in allowed.items():
        options = config[name.upper() + '_KWARGS']
        if options.keys() - keys:
            raise ValueError(f'{name.upper()}_KWARGS supports only {sorted(keys)}')
    if not config['data']:
        raise ValueError('data requires a path')
    if config['num_nodes'] <= 0 or not 0 <= config['node_rank'] < config['num_nodes']:
        raise ValueError('num_nodes must be positive; node_rank must be in [0, num_nodes)')
    if config['start_epoch'] > config['epochs']:
        raise ValueError('start_epoch cannot exceed epochs')


def parse_args(argv=None):
    default_path = Path(__file__).resolve().parent / 'experiments' / 'config_default.yaml'
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config_path', '--config-path', type=Path, default=default_path,
                        help='experiment YAML to overlay on config_default.yaml')
    parser.add_argument('--plot-lr', '--plot_lr', nargs='*', default=None, metavar='SCHEDULER',
                        help='plot all schedulers, or list steplr lambdalr multisteplr polylr; no training')
    parser.add_argument('--steps-per-epoch', type=int,
                        help='legacy LR preview option; accepted but no longer needed for epoch-based curves')
    parser.add_argument('--resume-task-id', '--resume_task_id', default=None,
                        help='download previous ClearML task last.pt and restore its config and state')
    parser.add_argument('--fast-mode', action='store_true',
                        help='train only one batch per epoch, then run full validation')
    parser.add_argument('--hpo', action='store_true',
                        help='remote HPO only: apply Training hyperparameters over YAML values')
    args = parser.parse_args(argv)
    if args.hpo and args.plot_lr is not None:
        parser.error('--hpo cannot be combined with --plot-lr')
    if args.steps_per_epoch is not None and (args.plot_lr is None or args.steps_per_epoch <= 0):
        parser.error('--steps-per-epoch requires --plot-lr and a positive integer')
    if args.plot_lr is not None:
        names = args.plot_lr or ['steplr', 'lambdalr', 'multisteplr', 'polylr']
        names = ['lambdalr' if name == 'lamdalr' else name for name in names]
        invalid = set(names) - {'steplr', 'lambdalr', 'multisteplr', 'polylr'}
        if invalid:
            parser.error(f'Unknown plot schedulers: {sorted(invalid)}')
        args.plot_lr = list(dict.fromkeys(names))
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
        from utils.clearml_utils import apply_hpo_overrides, restore_task_config
        resume_task_id = args.resume_task_id if args.resume_task_id is not None else config['resume_task_id']
        resume_model_id = None
        if resume_task_id:
            if args.hpo or args.fast_mode or args.plot_lr is not None:
                raise ValueError('Task resume cannot be combined with --hpo, --fast-mode or --plot-lr; saved config is authoritative')
            launch_node_rank = config['node_rank']
            config, resume_model_id = restore_task_config(resume_task_id, read_config(default_path))
            # A rank-zero checkpoint cannot provide another server's launch identity.
            config['node_rank'] = launch_node_rank
            hpo_overrides = {}
        else:
            config, hpo_overrides = apply_hpo_overrides(config, enabled=args.hpo)
            if args.fast_mode:
                config['fast_mode'] = True
        validate_config(config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    return argparse.Namespace(config_path=str(args.config_path), plot_lr=args.plot_lr,
                              steps_per_epoch=args.steps_per_epoch, hpo=args.hpo, hpo_overrides=hpo_overrides,
                              _resume_model_id=resume_model_id, **config)


def plot_learning_rate(args):
    """Plot epoch-based schedule formulas without data access or optimizer steps."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    if args.epochs <= 0:
        raise ValueError('LR plotting requires epochs > 0')
    # Curves are defined in epoch coordinates; no dataset scan or optimizer simulation.
    names = args.plot_lr
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names:
        settings = vars(args)[name.upper() + '_KWARGS']
        smooth = name in ('lambdalr', 'polylr')
        if smooth:
            # Bounded sampling plus all epoch/keyframe boundaries preserves short segments.
            sample_count = min(4000, max(1000, args.epochs * 20))
            positions = {args.epochs * i / sample_count for i in range(sample_count + 1)}
            positions.update(range(args.epochs + 1))
            if name == 'lambdalr':
                from utils.lr_interpolation import epoch_lr_curve
                curve = epoch_lr_curve(settings)
                positions.update(epoch for epoch in settings['points'] if 0 <= epoch <= args.epochs)
                # Resolve very short keyframe intervals as well.
                knots = sorted({0, args.epochs} | {epoch for epoch in settings['points']
                                                 if 0 <= epoch <= args.epochs})
                for left, right in zip(knots, knots[1:]):
                    positions.update(left + (right - left) * i / 32 for i in range(33))
            else:
                curve = lambda epoch: args.LR * max(0.0, 1 - epoch / args.epochs) ** settings['power']
            x = sorted(positions)
            y = [curve(epoch) for epoch in x]
        else:
            x = list(range(args.epochs + 1))
            if name == 'steplr':
                y = [args.LR * settings['gamma'] ** (epoch // settings['step_size']) for epoch in x]
            else:
                y = [args.LR * settings['gamma'] ** sum(epoch >= m for m in settings['milestones'])
                     for epoch in x]
        label = name + (f" ({settings.get('interpolation', 'hermite')}, scale={settings.get('scale', 1.0):g})"
                        if name == 'lambdalr' else '')
        ax.plot(x, y, label=label,
                drawstyle='default' if smooth else 'steps-post')
    ax.set(xlabel='Epoch', ylabel='Learning rate', xlim=(0, args.epochs),
           title=f'{Path(args.config_path).stem} | LR schedulers')
    ax.legend()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', axis='x', alpha=0.15)
    fig.tight_layout()
    output = Path(__file__).resolve().parent / 'experiments' / (Path(args.config_path).stem + '_lr_curve.png')
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f'Saved LR plot: {output} (epoch-based curves; no dataset access)')
    return output


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
    return constructor(weights='DEFAULT' if args.pretrained and not args.resume else None, **kwargs)


def main_ddp():
    from utils.clearml_utils import remote_bootstrap
    if not any(arg in ('--plot-lr', '--plot_lr') for arg in sys.argv):
        remote_bootstrap()
    args = parse_args()
    if args.plot_lr is not None:
        if int(os.environ.get('RANK', '0')) == 0:
            plot_learning_rate(args)
        return
    use_cuda = torch.cuda.is_available() and not args.no_accel
    gpus_per_node = torch.cuda.device_count() if use_cuda else 0
    configure_process_ranks(args, use_cuda, gpus_per_node)
    if args.distributed and not use_cuda and args.dist_backend == 'nccl':
        raise ValueError('CPU distributed training requires dist_backend: gloo')
    if args.multiprocessing_distributed:
        mp.spawn(main_worker, nprocs=args.processes_per_node, args=(args,))
    else:
        main_worker(args.local_rank, args)


def configure_process_ranks(args, use_cuda, visible_gpu_count, environ=None):
    """Keep node, local process and global process identities separate."""
    env = os.environ if environ is None else environ
    launched = 'LOCAL_RANK' in env
    if launched and args.multiprocessing_distributed:
        raise ValueError('Set multiprocessing_distributed: false when using torchrun')
    if launched:
        args.total_processes = int(env['WORLD_SIZE'])
        args.global_rank = int(env['RANK'])
        args.local_rank = int(env['LOCAL_RANK'])
        args.processes_per_node = int(env['LOCAL_WORLD_SIZE'])
        if (args.processes_per_node <= 0 or args.total_processes <= 0
                or args.total_processes % args.processes_per_node
                or not 0 <= args.global_rank < args.total_processes
                or not 0 <= args.local_rank < args.processes_per_node):
            raise ValueError('Invalid torchrun process ranks/counts')
        args.num_nodes = args.total_processes // args.processes_per_node
        args.node_rank = args.global_rank // args.processes_per_node
        args.gpu = args.local_rank if use_cuda else None
        args.dist_url = 'env://'
    elif args.multiprocessing_distributed:
        if not use_cuda or visible_gpu_count < 1:
            raise ValueError('multiprocessing_distributed requires CUDA; use torchrun for CPU DDP')
        if args.gpu is not None:
            raise ValueError('Set gpu: null with multiprocessing_distributed: true')
        args.processes_per_node = visible_gpu_count
        args.total_processes = args.num_nodes * args.processes_per_node
        # Local/global identities are assigned separately in each spawned worker.
        args.local_rank = None
        args.global_rank = None
    else:
        args.processes_per_node = 1
        args.total_processes = args.num_nodes
        args.local_rank = 0
        args.global_rank = args.node_rank
        if args.num_nodes > 1 and use_cuda and args.gpu is None:
            args.gpu = 0
    args.distributed = launched or args.total_processes > 1 or args.multiprocessing_distributed


def main_worker(local_rank, args):
    args.local_rank = local_rank
    if args.multiprocessing_distributed:
        args.gpu = local_rank
        args.global_rank = args.node_rank * args.processes_per_node + local_rank
    use_cuda = torch.cuda.is_available() and not args.no_accel
    device = torch.device('cuda', args.gpu if args.gpu is not None else 0) if use_cuda else torch.device('cpu')
    if use_cuda:
        torch.cuda.set_device(device)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.total_processes, rank=args.global_rank)
    try:
        main(args, device)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(args, device):
    is_main = not args.distributed or args.global_rank == 0
    dataset_tag = 'imgn' if args.dataset == 'image_net_1K' else args.dataset
    suffix = f'-{dataset_tag}' + ('-dummy' if args.dummy else '')
    from utils.run_output import reserve_run_directory, save_run_config
    output = [None]
    if is_main:
        output[0] = str(reserve_run_directory(Path(args.save) / (args.arch + suffix)))
    if args.distributed:
        dist.broadcast_object_list(output, src=0)
    args.output_dir = output[0]
    save_dir = Path(args.output_dir)
    handlers = [logging.StreamHandler(sys.stdout)] if is_main else [logging.NullHandler()]
    if is_main:
        handlers.append(logging.FileHandler(save_dir / f'log-{args.arch}{suffix}.txt', mode='a'))
    logging.basicConfig(level=logging.INFO if is_main else logging.WARNING,
                        format='%(message)s', handlers=handlers, force=True)
    logging.info('Ranks: node_rank=%d/%d local_rank=%d/%d global_rank=%d/%d',
                 args.node_rank, args.num_nodes, args.local_rank, args.processes_per_node,
                 args.global_rank, args.total_processes)
    logging.info('Output directory: %s', save_dir)
    from utils.clearml_utils import init_logger, init_best_model_registry
    model = create_model(args).to(device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu] if device.type == 'cuda' else None)
    elif device.type == 'cuda' and args.gpu is None and torch.cuda.device_count() > 1:
        model = DataParallel(model)

    optimizer = get_optimizer(unwrap_model(model), **optimizer_kwargs(vars(args)))
    conf = vars(args)
    dataset_module = get_dataset_module(args.dataset)
    train_loader = dataset_module.get_train_loader(conf)
    val_loader = dataset_module.get_valid_loader(conf)
    scheduler = get_scheduler(optimizer, {**conf, 'total_steps': len(train_loader) * args.epochs})
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda' and args.amp_dtype == 'float16')
    best_acc1 = load_checkpoint(args, model, optimizer, scheduler, scaler)
    # Checkpoint loading restores start_epoch; record the effective run settings afterward.
    if is_main:
        save_run_config(args, save_dir)
    logging.info('Arguments: %s', vars(args))
    if args.hpo:
        logging.info('HPO config overrides: %s', args.hpo_overrides)
    logger = init_logger(conf)
    trainer = Trainer(conf, train_loader, optimizer, scheduler, logger=logger, scaler=scaler)
    registry = init_best_model_registry(conf) if not args.evaluate else None
    validationer = Validationer(conf, optimizer=optimizer, scheduler=scheduler, best_acc=best_acc1,
                               logger=logger, scaler=scaler, registry=registry)
    if args.evaluate:
        validationer(model, args.start_epoch, val_loader)
        return

    started = time.perf_counter()
    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        run_training_epoch(model, epoch, trainer, validationer, val_loader,
                           device=device, distributed=args.distributed, logger=logger)
    logging.info('Total training time: %.2f hours', (time.perf_counter() - started) / 3600)



def run_training_epoch(model, epoch, trainer, validationer, val_loader, *, device,
                       distributed=False, logger=None):
    """Measure completed work; validation includes checkpoint saves and uploads."""
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        if distributed:
            dist.barrier()

    synchronize()
    started = time.perf_counter()
    trainer(model, epoch)
    synchronize()
    train_finished = time.perf_counter()
    validationer(model, epoch, val_loader)
    synchronize()
    finished = time.perf_counter()
    minutes = {'train': (train_finished - started) / 60,
               'validation': (finished - train_finished) / 60,
               'total': (finished - started) / 60}
    logging.info('Epoch %d time (minutes): train %.2f | validation %.2f | total %.2f',
                 epoch, minutes['train'], minutes['validation'], minutes['total'])
    if logger is not None:
        for series, value in minutes.items():
            logger.report_scalar('epoch time (minutes)', series, value, iteration=epoch)
    return minutes


def load_checkpoint(args, model, optimizer, scheduler, scaler=None):
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
        if scaler is not None and checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        logging.info('Resumed %s at epoch %d', args.resume, args.start_epoch)
        return float(checkpoint['best_acc1'])
    logging.warning('Loaded weights only from %s; optimizer/scheduler/epoch start fresh', args.resume)
    return 0.0


if __name__ == '__main__':
    main_ddp()
