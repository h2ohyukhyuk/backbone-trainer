"""ClearML remote tasks only; ordinary Python runs never initialize the SDK."""
import os


def remote_bootstrap():
    # clearml-task passes CLI values through its task's Args section.
    if os.environ.get('CLEARML_TASK_ID') and int(os.environ.get('RANK', '0')) == 0:
        from clearml import Task
        Task.init(auto_connect_frameworks=False)


def init_logger(conf):
    if conf.get('distributed', False) and conf['rank'] != 0:
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
    task.connect(string_keys(conf), name='Training', ignore_remote_overrides=True)
    return task.get_logger()


class BestModelRegistry:
    """Maintain one output model per task; upload only improved checkpoints."""
    def __init__(self, task, conf):
        self.task = task
        self.conf = conf
        self.model = None

    def update(self, checkpoint, *, epoch, accuracy, accuracy_top5):
        if self.model is None:
            from clearml import OutputModel
            import yaml
            # Without a destination the SDK may register only a container-local path.
            # True resolves the configured default storage or the ClearML file server.
            if not self.task.output_uri:
                self.task.output_uri = True
            self.model = OutputModel(
                task=self.task, framework='PyTorch',
                name=f'{self.task.get_project_name()}/{self.task.name}/best',
                tags=['best', self.conf['arch'], self.conf['dataset']],
                config_text=yaml.safe_dump(self.conf, sort_keys=False))
        self.model.comment = (f'Best validation Top-1: {accuracy:.6f}%; '
                              f'Top-5: {accuracy_top5:.6f}%; epoch: {epoch}')
        # Finish the upload before returning: the file may be replaced next epoch,
        # or the container may be removed as soon as training exits.
        uri = self.model.update_weights(
            weights_filename=str(checkpoint), target_filename='model_best.pt',
            iteration=epoch, auto_delete_file=False, async_enable=False,
            update_comment=False)
        if not uri:
            raise RuntimeError('ClearML best-model upload returned no storage URI')
        return uri


def init_best_model_registry(conf):
    if (not os.environ.get('CLEARML_TASK_ID')
            or (conf.get('distributed', False) and conf['rank'] != 0)):
        return None
    from clearml import Task
    task = Task.current_task()
    if task is None:
        raise RuntimeError('Initialize the remote ClearML task before model registration')
    return BestModelRegistry(task, conf)
