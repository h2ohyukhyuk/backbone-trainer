import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('lr_hpo', Path(__file__).resolve().parents[1] / 'hpo_bo_lr.py')
hpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpo)


class NativeHPOTests(unittest.TestCase):
    def test_default_ranges(self):
        args = hpo.parse_args([])
        self.assertEqual(len(hpo.search_ranges(args)), 5)
        self.assertEqual(hpo.search_ranges(args)[hpo.LR+'scale'], (.1, 1.0))
    def test_custom_and_fixed_ranges(self):
        args = hpo.parse_args(['--lr-15-range', '.007', '.007', '--scale-range', '.2', '.8'])
        self.assertEqual(hpo.search_ranges(args)[hpo.LR+'points/15'], (.007, .007))
        self.assertEqual(hpo.search_ranges(args)[hpo.LR+'scale'], (.2, .8))
    def test_invalid_ranges_and_execution(self):
        for argv in (['--scale-range', '1', '.1'], ['--lr-20-range', 'nan', '.1'],
                     ['--scale-range', '0', '1'], ['--execute'], ['--trials', '0']):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                hpo.parse_args(argv)
    def test_plan_has_no_sdk_dependency(self):
        import sys
        with patch.dict(sys.modules, {'clearml': None, 'optuna': None}), patch('builtins.print'):
            hpo.main_cli([])
    def test_template_validation(self):
        values = {'Training/lr_scheduler': 'lambdalr', 'Training/start_epoch': 0,
                  'Training/evaluate': False, 'Training/seed': 1024}
        values.update({key: .01 for key in hpo.search_ranges(hpo.parse_args([]))})
        task = Mock()
        task.get_parameters.return_value = values
        hpo.validate_base(task, hpo.search_ranges(hpo.parse_args([])))
        del values[hpo.LR+'points/15']
        with self.assertRaises(ValueError):
            hpo.validate_base(task, hpo.search_ranges(hpo.parse_args([])))
    @unittest.skipUnless(importlib.util.find_spec('optuna') and importlib.util.find_spec('clearml'),
                         'Install requirements-hpo.txt')
    def test_native_optimizer_configuration(self):
        from clearml.automation.optuna import OptimizerOptuna
        from optuna.pruners import NopPruner
        args = hpo.parse_args(['--base-task-id', 'baseline'])
        with patch('clearml.automation.HyperParameterOptimizer') as constructor:
            hpo.create_optimizer(args)
        kw = constructor.call_args.kwargs
        self.assertIs(kw['optimizer_class'], OptimizerOptuna)
        self.assertEqual(kw['objective_metric_sign'], 'max_global')
        self.assertEqual(kw['max_number_of_concurrent_tasks'], 1)
        self.assertEqual(kw['total_max_jobs'], 20)
        self.assertIsNone(kw['max_iteration_per_job'])
        self.assertIsInstance(kw['optuna_pruner'], NopPruner)
        self.assertNotIn('continue_previous_study', kw)
        self.assertFalse(kw['auto_connect_task'])
        ranges = kw['hyper_parameters']
        self.assertEqual(len(ranges), 6)
        self.assertEqual(ranges[-1].name, 'Args/hpo')
        self.assertEqual(ranges[-1].values, [True])
        for parameter in ranges[:-1]:
            self.assertEqual((parameter.min_value, parameter.max_value), hpo.search_ranges(args)[parameter.name])


if __name__ == '__main__':
    unittest.main()
