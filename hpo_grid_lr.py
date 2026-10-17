"""Three-stage ClearML LR grid search. Default mode only prints a local plan."""
import argparse
from copy import deepcopy
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import time

LR = 'Training/LAMBDALR_KWARGS/'
SCALE_VALUES = [.1, .325, .55, .775, 1.0]
STAGES = [
    ('scale', {LR + 'scale': SCALE_VALUES}),
    ('points-15-20', {LR + 'points/15': [.008, .007, .006], LR + 'points/20': [.008, .007, .006]}),
    ('points-43-48', {LR + 'points/43': [.001, .002, .003], LR + 'points/48': [.001, .002, .003]}),
]
COMPLETED = {'completed', 'published', 'closed'}
FAILED = {'failed', 'stopped'}


def stages(scale_values):
    result = deepcopy(STAGES)
    result[0][1][LR + 'scale'] = list(scale_values)
    return result


def candidates(parameters, axes):
    """All combinations, preserving every previously chosen parameter."""
    for name in axes:
        if name not in parameters:
            raise ValueError(f'Base task is missing {name}; use a current config/template task')
    for values in product(*axes.values()):
        result = deepcopy(parameters)
        result.update(zip(axes, values))
        result['Args/hpo'] = True
        yield result


def fingerprint(parameters):
    # hpo is an execution switch, not a training hyperparameter.
    training = {key: value for key, value in parameters.items() if key.startswith('Training/')}
    return hashlib.sha256(json.dumps(training, sort_keys=True).encode()).hexdigest()


def best_result(results):
    # Including cached rows retains the incumbent if no new trial improves it.
    # Ties retain the stage incumbent, including on restart.
    return max(results, key=lambda row: (row['top1'], row.get('incumbent', False)))


def status(task):
    value = task.status
    return str(getattr(value, 'value', value)).split('.')[-1].lower()


def score(task):
    if status(task) not in COMPLETED:
        raise ValueError(f'Task {task.id} is not successfully completed')
    try:
        value = float(task.get_last_scalar_metrics()['accuracy']['valid_top1']['max'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Task {task.id} has no validation Top-1 metric') from exc
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError(f'Task {task.id} has invalid Top-1: {value}')
    return value


def write_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
    temporary.replace(path)


def validate_base(parameters):
    if parameters.get('Training/lr_scheduler') != 'lambdalr':
        raise ValueError('Base task must use lr_scheduler: lambdalr')
    if (parameters.get('Training/resume') or parameters.get('Training/resume_task_id')
            or parameters.get('Training/start_epoch') != 0
            or parameters.get('Training/evaluate') is not False):
        raise ValueError('HPO requires fresh training: resume empty, start_epoch 0, evaluate false')
    if parameters.get('Training/seed') is None:
        raise ValueError('Set an explicit seed in the base config for comparable trials')
    for _, axes in STAGES:
        for name in axes:
            if name not in parameters:
                raise ValueError(f'Base task is missing {name}')
    if parameters.get('Args/plot_lr') not in (None, '', []):
        raise ValueError('Base task must be training, not an LR plot')
    # This controller launches one task at a time, not a coordinated group of nodes.
    if parameters.get('Training/num_nodes', 1) != 1:
        raise ValueError('This HPO controller supports single-node trials only')


def wait_for_trial(task, poll_seconds):
    while True:
        task.reload()
        current = status(task)
        if current in COMPLETED:
            return score(task)
        if current in FAILED:
            raise RuntimeError(f'Trial {task.id} ended with status {current}')
        time.sleep(poll_seconds)


def run_stages(api, controller, base, search_stages, state, state_path, poll_seconds=30):
    parameters = deepcopy(base.get_parameters(cast=True))
    validate_base(parameters)
    if status(base) not in COMPLETED:
        raise ValueError('Use a successfully completed base task as the experiment template')
    cache = state.setdefault('trials', {})
    signature = hashlib.sha256(json.dumps({
        'base_task_id': base.id, 'parameters': parameters, 'stages': search_stages,
    }, sort_keys=True).encode()).hexdigest()
    if state.get('signature', signature) != signature:
        raise ValueError('State file belongs to a different base task or search plan; use another --state')
    state['signature'] = signature
    state['base_task_id'] = base.id
    state['controller_task_id'] = controller.id
    state['stages'] = []
    # A completed template may itself match a candidate; never repeat the same training.
    base_key = fingerprint(parameters)
    if base_key not in cache:
        cache[base_key] = {'task_id': base.id, 'top1': score(base), 'parameters': deepcopy(parameters)}
    write_state(state_path, state)
    logger = controller.get_logger()
    for stage_index, (stage_name, axes) in enumerate(search_stages, 1):
        rows = []
        for trial_index, candidate in enumerate(candidates(parameters, axes), 1):
            key = fingerprint(candidate)
            reused = key in cache
            entry = cache.get(key)
            if entry is None:
                trial = api.clone(source_task=base.id,
                                  name=f'{base.name}/hpo-{stage_index}-{stage_name}/{trial_index:02d}',
                                  parent=controller.id)
                trial.set_parameters(candidate)
                entry = {'task_id': trial.id, 'parameters': candidate, 'top1': None}
                cache[key] = entry
                # Save the task ID before enqueueing so restart never silently clones a duplicate.
                write_state(state_path, state)
            elif entry.get('top1') is not None:
                trial = None
            else:
                trial = api.get_task(task_id=entry['task_id'])
            if entry.get('top1') is None:
                if status(trial) == 'created':
                    api.enqueue(task=trial, queue_name=state['queue'])
                entry['top1'] = wait_for_trial(trial, poll_seconds)
                write_state(state_path, state)
            row = {**entry, 'reused': reused, 'incumbent': key == fingerprint(parameters),
                   'searched': {name: candidate[name] for name in axes}}
            rows.append(row)
            logger.report_text(f'Stage {stage_index} {stage_name} {trial_index}: '
                               f'task={row["task_id"]} Top-1={row["top1"]:.6f} reused={reused} '
                               f'parameters={row["searched"]}')
        winner = best_result(rows)
        parameters = deepcopy(winner['parameters'])
        state['stages'].append({'name': stage_name, 'results': rows,
                                'winner_task_id': winner['task_id'], 'winner_top1': winner['top1']})
        state['best_task_id'] = winner['task_id']
        state['best_top1'] = winner['top1']
        state['best_parameters'] = parameters
        write_state(state_path, state)
        logger.report_scalar('HPO', 'stage_best_top1', winner['top1'], iteration=stage_index)
    return state


def local_plan(config_path, scale_values):
    # No ClearML imports, server access or task creation in plan mode.
    import main
    args = main.parse_args(['--config_path', str(config_path)])
    from utils.run_output import resolved_config
    def flatten(value, prefix='Training'):
        result = {}
        for key, item in value.items():
            name = prefix + '/' + str(key)
            if isinstance(item, dict): result.update(flatten(item, name))
            else: result[name] = item
        return result
    parameters = flatten(resolved_config(vars(args)))
    validate_base(parameters)
    total = 0
    for index, (name, axes) in enumerate(stages(scale_values), 1):
        count = math.prod(len(values) for values in axes.values())
        reusable = int(index > 1 and all(parameters[key] in values for key, values in axes.items()))
        new = count - reusable
        total += new
        print(f'Stage {index}: {name}; candidates={count}; previous-winner reuse={reusable}; new <= {new}')
        for key, values in axes.items(): print(f'  {key}: {values}')
    print(f'New training runs <= {total}; concurrency=1; objective=max validation Top-1')
    print('All 9 point combinations are included, including LR increases between the two points.')
    print('Plan only. Add --execute --base-task-id TASK_ID to submit actual training jobs.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-path', default='experiments/config.yml')
    parser.add_argument('--base-task-id')
    parser.add_argument('--queue', default='defalult')
    parser.add_argument('--project', default='APAG/HPO')
    parser.add_argument('--name', default='staged-lr-search')
    parser.add_argument('--scale-values', type=float, nargs='+', default=SCALE_VALUES)
    parser.add_argument('--state', type=Path, default=Path('outputs/hpo/lr_search.json'))
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if (not args.scale_values or len(set(args.scale_values)) != len(args.scale_values)
            or any(not math.isfinite(value) or value <= 0 for value in args.scale_values)):
        parser.error('--scale-values requires distinct finite positive values')
    if not 1 <= args.poll_seconds <= 60:
        parser.error('--poll-seconds must be between 1 and 60')
    if args.execute and not args.base_task_id:
        parser.error('--execute requires --base-task-id')
    return args


def main_cli(argv=None):
    args = parse_args(argv)
    if not args.execute:
        local_plan(args.config_path, args.scale_values)
        return
    from clearml import Task
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    if state.get('queue', args.queue) != args.queue:
        raise ValueError('Use the original queue when resuming this state file')
    state['queue'] = args.queue
    base = Task.get_task(task_id=args.base_task_id)
    # Capture the base parameters before initializing the separate controller task.
    validate_base(base.get_parameters(cast=True))
    controller = Task.init(project_name=args.project, task_name=args.name,
                           task_type=Task.TaskTypes.optimizer, reuse_last_task_id=False,
                           auto_connect_arg_parser=False, auto_connect_frameworks=False)
    controller.connect({'base_task_id': args.base_task_id, 'scale_values': args.scale_values,
                        'queue': args.queue, 'concurrency': 1}, name='Search', ignore_remote_overrides=True)
    run_stages(Task, controller, base, stages(args.scale_values), state, args.state, args.poll_seconds)
    # Keep the machine-readable experiment summary with the controller task as well.
    controller.upload_artifact(name='hpo_results', artifact_object=str(args.state), wait_on_upload=True)
    print(f'Best task: {state["best_task_id"]}; Top-1={state["best_top1"]:.6f}')
    print(f'Final config overrides: {state["best_parameters"]}')
    controller.close()


if __name__ == '__main__':
    main_cli()
