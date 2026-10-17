"""ClearML remote tasks only; ordinary Python runs never initialize the SDK."""
import os


def remote_bootstrap():
    # clearml-task passes CLI values through its task's Args section.
    if os.environ.get('CLEARML_TASK_ID') and int(os.environ.get('RANK', '0')) == 0:
        from clearml import Task
        Task.init(auto_connect_frameworks=False)


# Execution paths and distributed identities always come from YAML/launcher.
HPO_CONFIG_KEYS = {
    'arch', 'model_kwargs', 'epochs', 'batch_size', 'seed', 'amp', 'amp_dtype',
    'OPT_NAME', 'LR', 'MOMENTUM', 'WEIGHT_DECAY', 'AMS_GRAD',
    'USE_LR_MULT', 'LR_MULT_TARGET', 'LR_MULT_VAL',
    'SGD_KWARGS', 'ADAM_KWARGS', 'ADAMW_KWARGS', 'RMSPROP_KWARGS', 'ADADELTA_KWARGS',
    'lr_scheduler', 'STEPLR_KWARGS', 'LAMBDALR_KWARGS', 'MULTISTEPLR_KWARGS', 'POLYLR_KWARGS',
    'AUGMENTATION',
}


def apply_hpo_overrides(config, enabled=False):
    """Apply explicit remote HPO values to a copy, never to the YAML file."""
    from copy import deepcopy
    if not enabled:
        return config, {}
    task_id = os.environ.get('CLEARML_TASK_ID')
    if not task_id:
        raise ValueError('--hpo requires a ClearML remote task')
    from clearml import Task
    # Nonzero torchrun ranks only read the task; they must not initialize a task.
    task = Task.current_task() or Task.get_task(task_id=task_id)
    parameters = task.get_parameters(cast=True)
    result = deepcopy(config)
    changed = {}
    for path, value in parameters.items():
        if not path.startswith('Training/'):
            continue
        keys = path[len('Training/'):].split('/')
        if keys[0] not in HPO_CONFIG_KEYS:
            continue
        current = result
        for index, name in enumerate(keys):
            if not isinstance(current, dict):
                raise ValueError(f'HPO parameter is not a config field: {path}')
            matches = [key for key in current if str(key) == name]
            if len(matches) != 1:
                raise ValueError(f'Unknown or ambiguous HPO config field: {path}')
            key = matches[0]
            if index < len(keys) - 1:
                current = current[key]
                continue
            previous = current[key]
            # The SDK normally casts types; accept serialized numeric/list values too.
            if isinstance(value, str) and not isinstance(previous, str):
                import yaml
                value = yaml.safe_load(value)
                if isinstance(value, str) and type(previous) is float:
                    value = float(value)
            if value != previous:
                changed[path] = {'before': deepcopy(previous), 'after': deepcopy(value)}
            current[key] = deepcopy(value)
    return result, changed


def restore_task_config(task_id, defaults):
    """Download an explicit last model and restore its config before model/DDP setup."""
    from copy import deepcopy
    from pathlib import Path
    import torch
    from clearml import Task
    source = Task.get_task(task_id=task_id)
    status = getattr(source, 'status', None)
    status = str(getattr(status, 'value', status)).split('.')[-1].lower()
    if status in ('in_progress', 'queued'):
        raise ValueError('Stop the source task before resuming its last checkpoint')
    models = [model for model in source.models['output']
              if (getattr(model, 'name', '') or '').endswith('/last')
              or 'last' in (getattr(model, 'tags', None) or [])]
    if len(models) != 1:
        raise ValueError(f'Task {task_id} must have exactly one last model; found {len(models)}. '
                         'Best models are not used for resume.')
    model = models[0]
    path = model.get_local_copy(raise_on_error=True, force_download=True)
    if not path or not Path(path).is_file():
        raise ValueError(f'Task {task_id}: last checkpoint could not be downloaded')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    required = {'config', 'epoch', 'state_dict', 'optimizer', 'scheduler', 'scaler', 'best_acc1'}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError('Last checkpoint is missing config or resumable training state')
    if not isinstance(checkpoint['config'], dict) or type(checkpoint['epoch']) is not int or checkpoint['epoch'] < 0:
        raise ValueError('Last checkpoint has invalid config or epoch')
    restored = deepcopy(defaults)
    restored.update(deepcopy(checkpoint['config']))
    for field in ('arch', 'dataset'):
        if field in checkpoint and checkpoint[field] != restored[field]:
            raise ValueError(f'Last checkpoint {field} does not match its config')
    restored.update(resume=str(Path(path).resolve()), resume_task_id=task_id,
                    start_epoch=checkpoint['epoch'])
    return restored, model.id


def init_logger(conf):
    if conf.get('distributed', False) and conf['global_rank'] != 0:
        return None
    if not os.environ.get('CLEARML_TASK_ID'):
        return None
    try:
        from clearml import Task
    except ImportError as exc:
        raise RuntimeError('ClearML remote task: install clearml from requirements.txt') from exc
    task = Task.current_task() or Task.init(auto_connect_frameworks=False)
    # YAML remains the source of training settings; remote Args selects config_path.
    def string_keys(value):
        if isinstance(value, dict):
            return {str(key): string_keys(item) for key, item in value.items()}
        if isinstance(value, list):
            return [string_keys(item) for item in value]
        return value
    from utils.run_output import resolved_config
    task.connect(string_keys(resolved_config(conf)), name='Training', ignore_remote_overrides=True)
    if conf.get('_resume_model_id'):
        task.connect({'source_task_id': conf['resume_task_id'], 'source_model_id': conf['_resume_model_id']},
                     name='Resume', ignore_remote_overrides=True)
    return task.get_logger()


class BestModelRegistry:
    """Maintain separate best and last output models, with synchronous uploads."""
    def __init__(self, task, conf):
        self.task = task
        self.conf = conf
        self.model = None
        self.last_model = None

    def _update(self, kind, checkpoint, *, epoch, accuracy, accuracy_top5):
        attribute = 'model' if kind == 'best' else 'last_model'
        model = getattr(self, attribute)
        if model is None:
            from clearml import OutputModel
            import yaml
            if not self.task.output_uri:
                self.task.output_uri = True
            model = OutputModel(
                task=self.task, framework='PyTorch',
                name=f'{self.task.get_project_name()}/{self.task.name}/{kind}',
                tags=[kind, self.conf['arch'], self.conf['dataset']],
                config_text=yaml.safe_dump(self.conf, sort_keys=False))
            setattr(self, attribute, model)
        model.comment = (f'{"Best" if kind == "best" else "Last"} validation Top-1: {accuracy:.6f}%; '
                         f'Top-5: {accuracy_top5:.6f}%; epoch: {epoch}')
        uri = model.update_weights(
            weights_filename=str(checkpoint),
            target_filename='model_best.pt' if kind == 'best' else 'last.pt',
            iteration=epoch, auto_delete_file=False, async_enable=False, update_comment=False)
        if not uri:
            raise RuntimeError(f'ClearML {kind}-model upload returned no storage URI')
        return uri

    def update(self, checkpoint, *, epoch, accuracy, accuracy_top5):
        return self._update('best', checkpoint, epoch=epoch, accuracy=accuracy, accuracy_top5=accuracy_top5)

    def update_last(self, checkpoint, *, epoch, accuracy, accuracy_top5):
        return self._update('last', checkpoint, epoch=epoch, accuracy=accuracy, accuracy_top5=accuracy_top5)


def init_best_model_registry(conf):
    if (not os.environ.get('CLEARML_TASK_ID')
            or (conf.get('distributed', False) and conf['global_rank'] != 0)):
        return None
    from clearml import Task
    task = Task.current_task()
    if task is None:
        raise RuntimeError('Initialize the remote ClearML task before model registration')
    return BestModelRegistry(task, conf)
