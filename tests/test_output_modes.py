from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader, TensorDataset
import main


class OutputModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.args = main.parse_args([])
        self.args.save = str(Path(self.tmp.name) / 'local-only')
        self.args.arch = 'tiny'
        self.args.dataset = 'cifar10'
        self.args.dummy = True
        self.args.no_accel = True
        self.args.multiprocessing_distributed = False
        self.args.fast_mode = True
        self.args.epochs = 2
        self.args.batch_size = 2
        self.args.workers = 0
        main.configure_process_ranks(self.args, False, 0, environ={})
        self.model = torch.nn.Linear(3, 2)
        self.loader = DataLoader(TensorDataset(torch.randn(4, 3), torch.tensor([0, 1, 0, 1])), batch_size=2)

    def test_remote_ignores_save_stages_only_last_best_and_cleans_up(self):
        snapshots, staged_files = [], []
        registry = Mock()
        def upload_last(path, **kwargs):
            snapshots.append(torch.load(path, weights_only=True))
            staged_files.append({p.name for p in path.parent.iterdir()})
            return 'https://files.example/last.pt'
        registry.update_last.side_effect = upload_last
        registry.update.return_value = 'https://files.example/model_best.pt'
        logger = Mock()
        dataset = main.get_dataset_module(self.args.dataset)
        with patch.dict('os.environ', {'CLEARML_TASK_ID': 'remote-task'}), patch.object(
                main, 'create_model', return_value=self.model), patch.object(
                dataset, 'get_train_loader', return_value=self.loader), patch.object(
                dataset, 'get_valid_loader', return_value=self.loader), patch(
                'utils.clearml_utils.init_logger', return_value=logger), patch(
                'utils.clearml_utils.init_best_model_registry', return_value=registry), patch(
                'utils.run_output.save_run_config') as save_config, patch(
                'utils.run_output.reserve_run_directory') as reserve, patch(
                'train_val.classification.accuracy', return_value=(60., 100.)):
            main.main(self.args, torch.device('cpu'))
        reserve.assert_not_called()
        save_config.assert_not_called()
        self.assertFalse(Path(self.args.save).exists())
        self.assertTrue(Path(self.args.output_dir).name.startswith('backbone-clearml-'))
        self.assertFalse(Path(self.args.output_dir).exists())
        self.assertEqual([state['epoch'] for state in snapshots], [1, 2])
        self.assertEqual(registry.update_last.call_count, 2)
        self.assertEqual(registry.update.call_count, 1)
        for names in staged_files:
            self.assertLessEqual(names, {'last.pt', 'model_best.pt'})
        for state in snapshots:
            self.assertEqual(state['config']['arch'], 'tiny')
            self.assertEqual(state['config']['save'], self.args.save)
            self.assertNotIn('output_dir', state['config'])
            self.assertTrue({'optimizer', 'scheduler', 'scaler'} <= state.keys())

    def test_remote_temp_cleanup_even_on_failure(self):
        directories = []
        def fail(args, device):
            directories.append(Path(args.output_dir))
            self.assertTrue(directories[-1].is_dir())
            raise RuntimeError('simulated training failure')
        with patch.dict('os.environ', {'CLEARML_TASK_ID': 'remote-task'}), patch.object(
                main, '_run_training', side_effect=fail), self.assertRaises(RuntimeError):
            main.main(self.args, torch.device('cpu'))
        self.assertFalse(directories[0].exists())
        self.assertFalse(Path(self.args.save).exists())

    def test_local_outputs_remain_persistent(self):
        dataset = main.get_dataset_module(self.args.dataset)
        with patch.dict('os.environ', {}, clear=True), patch.object(
                main, 'create_model', return_value=self.model), patch.object(
                dataset, 'get_train_loader', return_value=self.loader), patch.object(
                dataset, 'get_valid_loader', return_value=self.loader), patch(
                'train_val.classification.accuracy', return_value=(60., 100.)):
            main.main(self.args, torch.device('cpu'))
        output = Path(self.args.output_dir)
        self.assertEqual(output.parent, Path(self.args.save))
        self.assertTrue({'001.pt', '002.pt', 'last.pt', 'model_best.pt',
                         'config_default.yaml', 'config.resolved.yml', 'log-tiny-cifar10-dummy.txt'}
                        <= {p.name for p in output.iterdir()})


if __name__ == '__main__':
    unittest.main()
