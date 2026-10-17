import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import main
from train_val.classification import Trainer, Validationer
from utils.clearml_utils import restore_task_config, BestModelRegistry
from utils.train_utils import get_optimizer, get_scheduler, optimizer_kwargs


class TaskResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.conf = vars(main.parse_args([]))
        self.conf.update(arch='tiny', dataset='cifar10', no_accel=True, dummy=True,
                         multiprocessing_distributed=False, num_nodes=1, node_rank=0,
                         batch_size=2, workers=0, epochs=2, start_epoch=0,
                         save=str(self.path), output_dir=str(self.path / 'source'),
                         distributed=False, global_rank=0, lr_scheduler='lambdalr')
        self.model = torch.nn.Linear(3, 10)
        self.loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 2, 3])), batch_size=2)
        self.optimizer = get_optimizer(self.model, **optimizer_kwargs(self.conf))
        self.scheduler = get_scheduler(self.optimizer, {**self.conf, 'total_steps': 4})
        self.trainer = Trainer(self.conf, self.loader, self.optimizer, self.scheduler)
        self.trainer(self.model, 1)
        self.validator = Validationer(self.conf, optimizer=self.optimizer,
                                      scheduler=self.scheduler, scaler=self.trainer.scaler)
        with patch('train_val.classification.accuracy', return_value=(60., 90.)):
            self.validator(self.model, 1, self.loader)
        self.last = self.validator.local_path / 'last.pt'
        self.last_model = Mock(id='last-model', tags=['last'])
        self.last_model.name = 'project/task/last'
        self.last_model.get_local_copy.return_value = str(self.last)
        self.best_model = Mock(id='best-model', tags=['best'])
        self.best_model.name = 'project/task/best'
        self.source = types.SimpleNamespace(models={'output': [self.last_model, self.best_model]})
        self.sdk = types.SimpleNamespace(Task=types.SimpleNamespace(get_task=lambda task_id: self.source))

    def test_restore_config_and_train_next_epoch(self):
        # A task ID supersedes unrelated current YAML model/batch/LR settings.
        with patch.dict('sys.modules', {'clearml': self.sdk}):
            args = main.parse_args(['--resume-task-id', 'source-task'])
        self.assertEqual(args.arch, 'tiny')
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.dataset, 'cifar10')
        self.assertEqual(args.start_epoch, 1)
        self.assertEqual(args.LAMBDALR_KWARGS, self.conf['LAMBDALR_KWARGS'])
        self.assertEqual(args._resume_model_id, 'last-model')
        self.last_model.get_local_copy.assert_called_once_with(raise_on_error=True, force_download=True)
        self.best_model.get_local_copy.assert_not_called()
        main.configure_process_ranks(args, False, 0, environ={})
        fresh = torch.nn.Linear(3, 10)
        recorded = []
        dataset = main.get_dataset_module(args.dataset)
        def record(conf): recorded.append(copy.deepcopy(conf))
        with patch.object(main, 'create_model', return_value=fresh), patch.object(
                dataset, 'get_train_loader', return_value=self.loader), patch.object(
                dataset, 'get_valid_loader', return_value=self.loader), patch(
                'utils.clearml_utils.init_logger', side_effect=record), patch(
                'utils.clearml_utils.init_best_model_registry', return_value=None):
            main.main(args, torch.device('cpu'))
        state = torch.load(Path(args.output_dir) / 'last.pt', weights_only=True)
        self.assertEqual(state['epoch'], 2)
        self.assertEqual(state['scheduler']['last_epoch'], 4)
        self.assertEqual(state['config']['resume_task_id'], 'source-task')
        self.assertNotIn('_resume_model_id', state['config'])
        self.assertEqual(recorded[0]['start_epoch'], 1)
        self.assertEqual(main.read_config(Path(args.output_dir) / 'config.resolved.yml')['start_epoch'], 1)

    def test_model_optimizer_scheduler_scaler_restore(self):
        with patch.dict('sys.modules', {'clearml': self.sdk}):
            args = main.parse_args(['--resume-task-id', 'source-task'])
        model = torch.nn.Linear(3, 10)
        optimizer = get_optimizer(model, **optimizer_kwargs(vars(args)))
        scheduler = get_scheduler(optimizer, {**vars(args), 'total_steps': 4})
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        best = main.load_checkpoint(args, model, optimizer, scheduler, scaler)
        self.assertEqual(best, 60.)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, self.model.state_dict()[key])
        self.assertEqual(scheduler.last_epoch, self.scheduler.last_epoch)
        self.assertEqual(optimizer.param_groups[0]['lr'], self.optimizer.param_groups[0]['lr'])
        old_states = list(self.optimizer.state.values())
        for current, old in zip(optimizer.state.values(), old_states):
            for key in old:
                torch.testing.assert_close(current[key], old[key])
        self.assertEqual(scaler.state_dict(), self.trainer.scaler.state_dict())

    def test_last_updates_even_without_new_best(self):
        self.validator.registry = Mock()
        with patch('train_val.classification.accuracy', return_value=(40., 80.)):
            self.validator(self.model, 2, self.loader)
        last = torch.load(self.last, weights_only=True)
        best = torch.load(self.validator.local_path / 'model_best.pt', weights_only=True)
        self.assertEqual(last['epoch'], 2)
        self.assertEqual(last['best_acc1'], 60.)
        self.assertEqual(best['epoch'], 1)
        self.validator.registry.update_last.assert_called_once()
        self.validator.registry.update.assert_not_called()

    def test_missing_or_ambiguous_last_does_not_use_best(self):
        for outputs in ([self.best_model], [self.last_model, self.last_model]):
            self.source.models['output'] = outputs
            with patch.dict('sys.modules', {'clearml': self.sdk}), self.assertRaises(ValueError):
                restore_task_config('source-task', main.read_config('experiments/config_default.yaml'))
        self.best_model.get_local_copy.assert_not_called()

    def test_invalid_checkpoint_rejected(self):
        bad = self.path / 'bad.pt'
        torch.save({'state_dict': self.model.state_dict()}, bad)
        self.last_model.get_local_copy.return_value = str(bad)
        with patch.dict('sys.modules', {'clearml': self.sdk}), self.assertRaises(ValueError):
            restore_task_config('source-task', main.read_config('experiments/config_default.yaml'))

    def test_best_and_last_are_separate_synchronous_models(self):
        task = types.SimpleNamespace(output_uri=None, name='training', get_project_name=lambda: 'APAG')
        created = []
        class OutputModel:
            def __init__(self, **kwargs):
                self.name = kwargs['name']
                self.uploads = []
                created.append(self)
            def update_weights(self, **kwargs):
                self.uploads.append(kwargs)
                return 'https://files.example/checkpoint.pt'
        registry = BestModelRegistry(task, self.conf)
        with patch.dict('sys.modules', {'clearml': types.SimpleNamespace(OutputModel=OutputModel)}):
            registry.update_last(self.last, epoch=1, accuracy=60., accuracy_top5=90.)
            registry.update(self.last, epoch=1, accuracy=60., accuracy_top5=90.)
            registry.update_last(self.last, epoch=2, accuracy=40., accuracy_top5=80.)
        self.assertEqual([model.name for model in created], ['APAG/training/last', 'APAG/training/best'])
        self.assertEqual([len(model.uploads) for model in created], [2, 1])
        self.assertTrue(all(not upload['async_enable'] for model in created for upload in model.uploads))
        self.assertEqual(created[0].uploads[-1]['target_filename'], 'last.pt')


if __name__ == '__main__':
    unittest.main()
