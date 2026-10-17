"""ClearML native Bayesian LR HPO. Default: print ranges without submitting jobs."""
import argparse
import math
from pathlib import Path

LR = 'Training/LAMBDALR_KWARGS/'
DEFAULT_RANGES = {
    'scale': (0.1, 1.0),
    '15': (0.006, 0.008),
    '20': (0.006, 0.008),
    '43': (0.001, 0.003),
    '48': (0.001, 0.003),
}


def search_ranges(args):
    return {LR + ('scale' if key == 'scale' else 'points/' + key):
            tuple(getattr(args, 'scale_range' if key == 'scale' else 'lr_' + key + '_range'))
            for key in DEFAULT_RANGES}


def validate_base(task, ranges):
    parameters = task.get_parameters(cast=True)
    if parameters.get('Training/lr_scheduler') != 'lambdalr':
        raise ValueError('Base task must use lr_scheduler: lambdalr')
    if (parameters.get('Training/resume') or parameters.get('Training/start_epoch') != 0
            or parameters.get('Training/evaluate') is not False):
        raise ValueError('Use fresh training: resume empty, start_epoch 0, evaluate false')
    if parameters.get('Training/seed') is None:
        raise ValueError('Set an explicit training seed in the baseline')
    if parameters.get('Training/num_nodes', 1) != 1:
        raise ValueError('This controller supports single-node trials only')
    if parameters.get('Args/plot_lr') not in (None, '', []):
        raise ValueError('Use a training task, not an LR plot task')
    for name in ranges:
        if name not in parameters:
            raise ValueError(f'Base task is missing {name}')
    return parameters


def create_optimizer(args):
    from clearml.automation import HyperParameterOptimizer, UniformParameterRange, DiscreteParameterRange
    from clearml.automation.optuna import OptimizerOptuna
    from optuna.samplers import TPESampler
    from optuna.pruners import NopPruner
    parameters = [UniformParameterRange(name, min_value=low, max_value=high)
                  if low < high else DiscreteParameterRange(name, values=[low])
                  for name, (low, high) in search_ranges(args).items()]
    # A fixed execution switch; not an additional search dimension.
    parameters.append(DiscreteParameterRange('Args/hpo', values=[True]))
    return HyperParameterOptimizer(
        base_task_id=args.base_task_id,
        hyper_parameters=parameters,
        objective_metric_title='accuracy',
        objective_metric_series='valid_top1',
        objective_metric_sign='max_global',
        optimizer_class=OptimizerOptuna,
        max_number_of_concurrent_tasks=1,
        execution_queue=args.queue,
        total_max_jobs=args.trials,
        max_iteration_per_job=None,
        min_iteration_per_job=None,
        optuna_sampler=TPESampler(seed=args.seed, n_startup_trials=args.startup_trials),
        optuna_pruner=NopPruner(),
        pool_period_min=args.poll_seconds / 60,
        auto_connect_task=False,
        save_top_k_tasks_only=None,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--base-task-id')
    parser.add_argument('--queue', default='defalult')
    parser.add_argument('--project', default='APAG/HPO')
    parser.add_argument('--name', default='bo-lr-search')
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--startup-trials', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1024, help='BO sampler seed; training seed stays from baseline')
    parser.add_argument('--poll-seconds', type=float, default=30)
    for key, bounds in DEFAULT_RANGES.items():
        option = '--scale-range' if key == 'scale' else '--lr-' + key + '-range'
        parser.add_argument(option, type=float, nargs=2, default=bounds, metavar=('MIN', 'MAX'))
    args = parser.parse_args(argv)
    if args.trials < 1 or not 1 <= args.startup_trials <= args.trials:
        parser.error('Require trials >= startup-trials >= 1')
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error('--seed must be within uint32 range')
    if not math.isfinite(args.poll_seconds) or not 1 <= args.poll_seconds <= 60:
        parser.error('--poll-seconds must be between 1 and 60')
    for name, (low, high) in search_ranges(args).items():
        if not all(math.isfinite(value) and value > 0 for value in (low, high)) or low > high:
            parser.error(f'{name}: require finite 0 < MIN <= MAX')
    if args.execute and not args.base_task_id:
        parser.error('--execute requires --base-task-id')
    return args


def main_cli(argv=None):
    args = parse_args(argv)
    ranges = search_ranges(args)
    if not args.execute:
        print(f'ClearML HyperParameterOptimizer + OptimizerOptuna (TPE); trials={args.trials}; concurrency=1')
        for name, bounds in ranges.items():
            print(f'  {name}: {bounds[0]} ~ {bounds[1]}')
        print(f'Initial random trials={args.startup_trials}; objective=maximum validation Top-1')
        print('No Grid results or baseline scores imported; no early stopping; fresh training per trial.')
        print('Plan only. Add --execute --base-task-id TASK_ID to submit jobs.')
        return
    try:
        from clearml import Task
        from clearml.automation.optuna import OptimizerOptuna
    except ImportError as exc:
        raise SystemExit('Install controller dependencies: pip install -r requirements-hpo.txt') from exc
    base = Task.get_task(task_id=args.base_task_id)
    validate_base(base, ranges)
    controller = Task.init(project_name=args.project, task_name=args.name,
                           task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False,
                           auto_connect_arg_parser=False, auto_connect_frameworks=False)
    optimizer = None
    try:
        controller.connect({
            'base_task_id': args.base_task_id, 'ranges': ranges, 'trials': args.trials,
            'startup_trials': args.startup_trials, 'sampler_seed': args.seed,
            'queue': args.queue, 'concurrency': 1,
        }, name='Search', ignore_remote_overrides=True)
        optimizer = create_optimizer(args)
        optimizer.set_report_period(args.poll_seconds / 60)
        if not optimizer.start():
            raise RuntimeError('ClearML HPO failed to start')
        # Bounded waits keep interrupts responsive while ClearML manages all jobs.
        while not optimizer.wait(timeout=0.5):
            pass
        results = optimizer.get_top_experiments_details(
            top_k=args.trials, all_metrics=True, all_hyper_parameters=True, only_completed=True)
        controller.upload_artifact(name='hpo_results', artifact_object=results, wait_on_upload=True)
        if not results:
            raise RuntimeError('No completed HPO trials; inspect ClearML controller and trial logs')
        print(f"Best task: {results[0]['task_id']}")
    finally:
        if optimizer is not None:
            optimizer.stop()
        controller.close()


if __name__ == '__main__':
    main_cli()
