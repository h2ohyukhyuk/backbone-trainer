import copy
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import main
from utils.clearml_utils import apply_hpo_overrides, init_logger
from utils.run_output import resolved_config


class HPOTests(unittest.TestCase):
    def setUp(self):
        self.config = main.read_config('experiments/config_default.yaml')
        # Keep test search points independent of edits to the user's LR schedule.
        self.config['LAMBDALR_KWARGS']['points'] = {0: .0001, 2: .01, 15: .008, 45: .002, 70: .0003, 90: .00001}

    def sdk(self, parameters, current=True):
        task = types.SimpleNamespace(get_parameters=lambda cast: parameters)
        task_type = types.SimpleNamespace(current_task=lambda: task if current else None,
                                         get_task=lambda task_id: task)
        return types.SimpleNamespace(Task=task_type)

    def test_normal_execution_never_reads_clearml_parameters(self):
        with patch.dict('os.environ', {'CLEARML_TASK_ID': 'remote'}, clear=True), patch.dict(
                'sys.modules', {'clearml': None}):
            result, changes = apply_hpo_overrides(self.config, enabled=False)
        self.assertIs(result, self.config)
        self.assertEqual(changes, {})

    def test_hpo_requires_remote_task(self):
        with patch.dict('os.environ', {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'remote task'):
                apply_hpo_overrides(self.config, enabled=True)

    def test_numeric_lr_points_applied_without_modifying_source(self):
        original = copy.deepcopy(self.config)
        parameters = {'Training/LAMBDALR_KWARGS/points/15': .007,
                      'Training/LAMBDALR_KWARGS/points/45': .001,
                      'Training/WEIGHT_DECAY': .01,
                      'Training/LAMBDALR_KWARGS/scale': .5,
                      'Training/global_rank': 7, 'Training/gpu': 1,
                      'Training/save': '/tmp/ignored', 'Args/config_path': 'ignored'}
        with patch.dict('os.environ', {'CLEARML_TASK_ID': 'remote'}, clear=True), patch.dict(
                'sys.modules', {'clearml': self.sdk(parameters)}):
            result, changes = apply_hpo_overrides(self.config, enabled=True)
        self.assertEqual(self.config, original)
        self.assertEqual(result['LAMBDALR_KWARGS']['points'][15], .007)
        self.assertEqual(result['LAMBDALR_KWARGS']['points'][45], .001)
        self.assertEqual(result['WEIGHT_DECAY'], .01)
        self.assertEqual(result['save'], original['save'])
        self.assertEqual(result['gpu'], original['gpu'])
        self.assertEqual(result['LAMBDALR_KWARGS']['scale'], .5)
        self.assertEqual(len(changes), 4)
        main.validate_config(result)

    def test_parse_applies_hpo_and_records_effective_config(self):
        path = Path('experiments/config.yml')
        before = path.read_bytes()
        params = {'Training/LAMBDALR_KWARGS/points/15': .006}
        connected = []
        class Task:
            def get_parameters(self, cast): return params
            def connect(self, config, **kwargs):
                connected.append((config, kwargs))
            def get_logger(self): return 'logger'
        task = Task()
        sdk = types.SimpleNamespace(Task=types.SimpleNamespace(current_task=lambda: task))
        with patch.dict('os.environ', {'CLEARML_TASK_ID': 'remote'}, clear=True), patch.dict(
                'sys.modules', {'clearml': sdk}):
            args = main.parse_args(['--config_path', str(path), '--hpo'])
            self.assertEqual(init_logger(vars(args)), 'logger')
        self.assertEqual(args.LAMBDALR_KWARGS['points'][15], .006)
        effective = resolved_config(vars(args))
        main.validate_config(effective)
        self.assertNotIn('hpo_overrides', effective)
        self.assertEqual(connected[0][0]['LAMBDALR_KWARGS']['points']['15'], .006)
        self.assertTrue(connected[0][1]['ignore_remote_overrides'])
        self.assertEqual(path.read_bytes(), before)

    def test_nonzero_torchrun_rank_reads_task_without_initialization(self):
        params = {'Training/WEIGHT_DECAY': .002}
        with patch.dict('os.environ', {'CLEARML_TASK_ID': 'remote', 'RANK': '1'}, clear=True), patch.dict(
                'sys.modules', {'clearml': self.sdk(params, current=False)}):
            result, _ = apply_hpo_overrides(self.config, enabled=True)
        self.assertEqual(result['WEIGHT_DECAY'], .002)

    def test_unknown_tuning_field_and_invalid_value_rejected(self):
        with patch.dict('os.environ', {'CLEARML_TASK_ID': 'remote'}, clear=True), patch.dict(
                'sys.modules', {'clearml': self.sdk({'Training/ADAMW_KWARGS/unknown': 3})}):
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                apply_hpo_overrides(self.config, enabled=True)
        with patch.dict('os.environ', {'CLEARML_TASK_ID': 'remote'}, clear=True), patch.dict(
                'sys.modules', {'clearml': self.sdk({'Training/LR': -1})}):
            invalid, _ = apply_hpo_overrides(self.config, enabled=True)
        with self.assertRaises(ValueError): main.validate_config(invalid)


if __name__ == '__main__': unittest.main()
