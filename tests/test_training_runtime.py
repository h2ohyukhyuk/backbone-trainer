import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import main
import train_val.classification as runtime
from utils.clearml_utils import init_logger, remote_bootstrap, init_best_model_registry, BestModelRegistry
from utils.run_output import reserve_run_directory, save_run_config
from utils.train_utils import rank_zero_print
from datetime import datetime
from zoneinfo import ZoneInfo


class Logger:
    def __init__(self):
        self.values = {}

    def report_scalar(self, title, series, value, iteration):
        self.values[title, series] = value


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.conf = vars(main.parse_args([]))
        self.conf.update(no_accel=True, dataset='cifar10', dummy=True,
                         start_epoch=0, save=self.directory.name, arch='tiny',
                         distributed=False, rank=0, print_freq=100)
        self.loader = DataLoader(TensorDataset(torch.randn(5, 3), torch.tensor([0, 1, 0, 1, 0])), batch_size=2)
        self.model = torch.nn.Linear(3, 2)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=.01)

    def test_train_has_no_accuracy_and_validation_logs_both_losses(self):
        logger = Logger()
        scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 1)
        trainer = runtime.Trainer(self.conf, self.loader, self.opt, scheduler, logger=logger)
        with patch.object(runtime, 'accuracy', side_effect=AssertionError('training accuracy forbidden')):
            loss = trainer(self.model, 1)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(trainer.amp_enabled)  # AMP requested, CUDA unavailable/CPU explicitly selected.
        self.assertEqual(scheduler.last_epoch, 1)
        validator = runtime.Validationer(self.conf, optimizer=self.opt, scheduler=scheduler,
                                         logger=logger, scaler=trainer.scaler)
        validator(self.model, 1, self.loader)
        self.assertIn(('loss', 'train'), logger.values)
        self.assertIn(('loss', 'valid'), logger.values)
        self.assertIn(('accuracy', 'valid_top1'), logger.values)
        self.assertFalse(any(k[0] == 'accuracy' and 'train' in k[1] for k in logger.values))
        checkpoint = validator.local_path / '001.pt'
        saved = torch.load(checkpoint, weights_only=True)
        self.assertIn('scaler', saved)
        self.assertEqual(saved['config']['arch'], 'tiny')
        self.assertEqual(saved['config']['LAMBDALR_KWARGS'], self.conf['LAMBDALR_KWARGS'])
        self.assertNotIn('distributed', saved['config'])
        self.assertEqual(saved['metrics']['valid_loss'], logger.values['loss', 'valid'])
        self.assertEqual(saved['metrics']['top1'], logger.values['accuracy', 'valid_top1'])
        self.assertEqual(saved['metrics']['top5'], logger.values['accuracy', 'valid_top5'])
        main.validate_config(saved['config'])
        args = type('Args', (), {'resume': str(checkpoint), 'start_epoch': 0})()
        main.load_checkpoint(args, self.model, self.opt, scheduler, trainer.scaler)
        self.assertEqual(args.start_epoch, 1)

    def test_skipped_amp_update_does_not_advance_step_scheduler(self):
        class SkipScaler:
            def __init__(self): self.value = 8
            def scale(self, loss): return loss
            def get_scale(self): return self.value
            def step(self, opt): pass
            def update(self): self.value /= 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda step: 1)
        scheduler.step_per_batch = True
        before = copy.deepcopy(self.model.state_dict())
        trainer = runtime.Trainer(self.conf, self.loader, self.opt, scheduler, scaler=SkipScaler())
        trainer(self.model, 1)
        self.assertEqual(scheduler.last_epoch, 0)
        for key, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, before[key])

    def test_local_and_nonzero_rank_do_not_create_task(self):
        with patch.dict('os.environ', {}, clear=True), patch.dict('sys.modules', {'clearml': None}):
            remote_bootstrap()
            self.assertIsNone(init_logger(self.conf))
        with patch.dict('os.environ', {'CLEARML_TASK_ID': 'submitted'}, clear=True):
            self.conf.update(distributed=True, rank=1)
            self.assertIsNone(init_logger(self.conf))

    def test_registry_only_remote_rank_zero(self):
        with patch.dict('os.environ', {}, clear=True), patch.dict('sys.modules', {'clearml': None}):
            self.assertIsNone(init_best_model_registry(self.conf))
        with patch.dict('os.environ', {'CLEARML_TASK_ID': 'submitted'}, clear=True):
            self.assertIsNone(init_best_model_registry({**self.conf, 'distributed': True, 'rank': 1}))

    def test_best_model_reuses_one_registry_entry_and_uploads_synchronously(self):
        created, uploads = [], []
        class OutputModel:
            def __init__(self, **kwargs): created.append(kwargs)
            def update_weights(self, **kwargs):
                uploads.append(kwargs)
                return 'https://files.example/model_best.pt'
        task = types.SimpleNamespace(output_uri=None, name='baseline', get_project_name=lambda: 'APAG')
        registry = BestModelRegistry(task, self.conf)
        with patch.dict('sys.modules', {'clearml': types.SimpleNamespace(OutputModel=OutputModel)}):
            registry.update('/tmp/model_best.pt', epoch=1, accuracy=50, accuracy_top5=80)
            registry.update('/tmp/model_best.pt', epoch=3, accuracy=60, accuracy_top5=90)
        self.assertIn('Top-1: 60.000000%', registry.model.comment)
        self.assertIn('Top-5: 90.000000%', registry.model.comment)
        self.assertEqual(len(created), 1)
        self.assertIs(created[0]['task'], task)
        self.assertEqual(created[0]['name'], 'APAG/baseline/best')
        self.assertTrue(task.output_uri)
        self.assertEqual([u['iteration'] for u in uploads], [1, 3])
        self.assertTrue(all(not u['async_enable'] and not u['auto_delete_file'] for u in uploads))

    def test_validation_uploads_only_when_accuracy_improves(self):
        from unittest.mock import Mock
        registry = Mock()
        validator = runtime.Validationer(self.conf, registry=registry)
        with patch.object(runtime, 'accuracy', return_value=(60., 100.)):
            validator(self.model, 1, self.loader)
            validator(self.model, 2, self.loader)
        self.assertEqual(registry.update.call_count, 1)
        with patch.object(runtime, 'accuracy', return_value=(80., 100.)):
            validator(self.model, 3, self.loader)
        self.assertEqual(registry.update.call_count, 2)
        self.assertEqual(registry.update.call_args.kwargs, {'epoch': 3, 'accuracy': 80., 'accuracy_top5': 100.})
        self.assertTrue((validator.local_path / '001.pt').exists())
        self.assertTrue((validator.local_path / '002.pt').exists())
        self.assertTrue((validator.local_path / 'model_best.pt').exists())
        with patch.object(runtime, 'accuracy', return_value=(40., 90.)):
            validator(self.model, 4, self.loader)
        latest = torch.load(validator.local_path / '004.pt', weights_only=True)
        best = torch.load(validator.local_path / 'model_best.pt', weights_only=True)
        self.assertEqual(best['epoch'], 3)
        self.assertEqual(best['metrics']['top1'], 80.)
        self.assertEqual(best['metrics']['top5'], 100.)
        self.assertEqual(latest['metrics']['top1'], 40.)
        self.assertEqual(latest['metrics']['top5'], 90.)
        self.assertEqual(latest['best_acc1'], 80.)
        self.assertEqual(registry.update.call_count, 2)

    def test_only_global_rank_zero_prints(self):
        with patch('torch.distributed.is_initialized', return_value=True), patch('builtins.print') as output:
            with patch('torch.distributed.get_rank', return_value=1):
                rank_zero_print('hidden')
                output.assert_not_called()
            with patch('torch.distributed.get_rank', return_value=0):
                rank_zero_print('visible')
                output.assert_called_once_with('visible')

    def test_run_directories_and_config_copy(self):
        base = Path(self.directory.name) / 'run'
        stamp = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%y%m%d')
        paths = [reserve_run_directory(base) for _ in range(3)]
        self.assertEqual([p.name for p in paths], ['run', f'run-{stamp}', f'run-{stamp}-2'])
        args = main.parse_args(['--config_path', 'experiments/APAGNet.yml'])
        args.world_size = 2
        args.rank = 0
        args.local_world_size = 2
        args.output_dir = str(paths[0])
        save_run_config(args, paths[0])
        self.assertEqual((paths[0] / 'APAGNet.yml').read_bytes(), Path(args.config_path).read_bytes())
        resolved = main.read_config(paths[0] / 'config.resolved.yml')
        main.validate_config(resolved)
        self.assertEqual(resolved['world_size'], 1)
        self.assertNotIn('output_dir', resolved)
        conf = {**self.conf, 'output_dir': str(paths[0])}
        validator = runtime.Validationer(conf)
        self.assertEqual(validator.local_path, paths[0])

    def test_remote_task_is_reused_and_config_keys_are_serializable(self):
        logger = Logger()
        connected = []
        class Task:
            current = None
            calls = 0
            @classmethod
            def init(cls, **kwargs):
                cls.calls += 1
                cls.current = cls()
                return cls.current
            @classmethod
            def current_task(cls): return cls.current
            def connect(self, config, **kwargs): connected.append(config)
            def get_logger(self): return logger
        with patch.dict('os.environ', {'CLEARML_TASK_ID': 'submitted'}, clear=True), patch.dict(
                'sys.modules', {'clearml': types.SimpleNamespace(Task=Task)}):
            remote_bootstrap()
            self.assertIs(init_logger(self.conf), logger)
        self.assertEqual(Task.calls, 1)
        self.assertNotIn('CLEARML', connected[0])
        self.assertTrue(all(isinstance(key, str) for key in connected[0]['LAMBDALR_KWARGS']['points']))


if __name__ == '__main__':
    unittest.main()
