"""Reserve a fresh output directory without overwriting previous runs."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import yaml


def reserve_run_directory(base):
    base = Path(base).resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%y%m%d')
    attempt = 0
    while True:
        suffix = '' if attempt == 0 else f'-{stamp}' + (f'-{attempt}' if attempt > 1 else '')
        candidate = base.with_name(base.name + suffix)
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            attempt += 1


def resolved_config(conf):
    """Snapshot merged training settings without runtime-only attributes."""
    excluded = {'_resume_model_id', 'hpo', 'hpo_overrides', 'config_path', 'plot_lr', 'steps_per_epoch', 'distributed', 'processes_per_node', 'total_processes', 'local_rank', 'global_rank', 'output_dir'}
    config = deepcopy({key: value for key, value in conf.items() if key not in excluded})
    if conf.get('multiprocessing_distributed', False):
        config['gpu'] = None  # Worker device assignment is not a user GPU override.
    return config


def save_run_config(args, directory):
    source = Path(args.config_path)
    name = source.name if source.name != 'config.resolved.yml' else 'config.used.yml'
    shutil.copy2(source, directory / name)
    with (directory / 'config.resolved.yml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(resolved_config(vars(args)), stream, sort_keys=False, allow_unicode=True)
