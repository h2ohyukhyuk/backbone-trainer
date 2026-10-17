from copy import deepcopy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock

import importlib.util

spec = importlib.util.spec_from_file_location('hpo', Path(__file__).resolve().parents[1] / 'hpo_grid_lr.py')
hpo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hpo)


def base_parameters():
    return {
        'Training/lr_scheduler': 'lambdalr', 'Training/resume': '',
        'Training/start_epoch': 0, 'Training/evaluate': False,
        'Training/seed': 1024, 'Training/num_nodes': 1, 'Training/epochs': 90,
        'Training/batch_size': 256, 'Training/WEIGHT_DECAY': .0001,
        'Args/config_path': 'experiments/config.yml', 'Args/hpo': False,
        hpo.LR + 'scale': .95,
        hpo.LR + 'points/15': .007, hpo.LR + 'points/20': .007,
        hpo.LR + 'points/43': .002, hpo.LR + 'points/48': .002,
    }


class FakeTask:
    def __init__(self, task_id, parameters, objective, completed=False):
        self.id, self.name = task_id, 'baseline'
        self.parameters = deepcopy(parameters)
        self.objective = objective
        self.status = 'completed' if completed else 'created'
    def get_parameters(self, cast=True): return deepcopy(self.parameters)
    def set_parameters(self, value): self.parameters = deepcopy(value)
    def reload(self): pass
    def get_last_scalar_metrics(self):
        return {'accuracy': {'valid_top1': {'max': self.objective(self.parameters)}}}


class FakeAPI:
    def __init__(self, objective):
        self.objective = objective
        self.base = FakeTask('base', base_parameters(), objective, completed=True)
        self.tasks = {'base': self.base}
        self.clones = []
        self.enqueued = []
    def clone(self, source_task, name, parent):
        assert source_task == 'base'
        task = FakeTask(f'trial-{len(self.clones)+1}', self.base.parameters, self.objective)
        self.tasks[task.id] = task
        self.clones.append(task)
        return task
    def enqueue(self, task, queue_name):
        self.enqueued.append(task.id)
        task.status = 'completed'
    def get_task(self, task_id): return self.tasks[task_id]


class StagedHPOTests(unittest.TestCase):
    def run_search(self, objective):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / 'state.json'
        api = FakeAPI(objective)
        logger = Mock()
        controller = types.SimpleNamespace(id='controller', get_logger=lambda: logger)
        state = {'queue': 'test'}
        hpo.run_stages(api, controller, api.base, hpo.stages(hpo.SCALE_VALUES), state, path, 1)
        return api, state, path, controller

    def test_twenty_one_runs_and_previous_optima_are_frozen(self):
        def objective(p):
            return (80 - abs(p[hpo.LR+'scale']-.325)
                    - 100*abs(p[hpo.LR+'points/15']-.008)
                    - 100*abs(p[hpo.LR+'points/20']-.006)
                    - 100*abs(p[hpo.LR+'points/43']-.003)
                    - 100*abs(p[hpo.LR+'points/48']-.001))
        api, state, path, controller = self.run_search(objective)
        self.assertEqual(len(api.clones), 21)
        self.assertEqual([len(s['results']) for s in state['stages']], [5, 9, 9])
        self.assertEqual([sum(r['reused'] for r in s['results']) for s in state['stages']], [0, 1, 1])
        selected = state['best_parameters']
        self.assertEqual(selected[hpo.LR+'scale'], .325)
        self.assertEqual(selected[hpo.LR+'points/15'], .008)
        self.assertEqual(selected[hpo.LR+'points/20'], .006)
        self.assertEqual(selected[hpo.LR+'points/43'], .003)
        self.assertEqual(selected[hpo.LR+'points/48'], .001)
        for trial in api.clones:
            self.assertIs(trial.parameters['Args/hpo'], True)
            self.assertEqual(trial.parameters['Training/seed'], 1024)
            self.assertEqual(trial.parameters['Training/resume'], '')
            self.assertEqual(trial.parameters['Training/epochs'], 90)
        self.assertEqual(api.base.parameters, base_parameters())
        self.assertTrue(path.exists())

    def test_incumbent_is_kept_when_all_new_candidates_tie(self):
        api, state, path, controller = self.run_search(lambda p: 70.)
        winners = [stage['winner_task_id'] for stage in state['stages']]
        self.assertEqual(winners[0], winners[1])
        self.assertEqual(winners[1], winners[2])
        self.assertEqual(len(api.clones), 21)
        hpo.run_stages(api, controller, api.base, hpo.stages(hpo.SCALE_VALUES), state, path, 1)
        self.assertEqual(len(api.clones), 21)  # Restart reuses every completed trial.

    def test_different_plan_cannot_reuse_state(self):
        api, state, path, controller = self.run_search(lambda p: 70.)
        with self.assertRaisesRegex(ValueError, 'different'):
            hpo.run_stages(api, controller, api.base, hpo.stages([.25,.5,.75,1.]), state, path, 1)

    def test_failed_trial_stops_stage_and_remains_tracked(self):
        api = FakeAPI(lambda p: 70.)
        api.enqueue = lambda task, queue_name: setattr(task, 'status', 'failed')
        controller = types.SimpleNamespace(id='controller', get_logger=lambda: Mock())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.json'
            state = {'queue': 'test'}
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                hpo.run_stages(api, controller, api.base, hpo.stages(hpo.SCALE_VALUES), state, path, 1)
            self.assertEqual(len(api.clones), 1)
            self.assertIn('trial-1', path.read_text())

    def test_config_fingerprint_ignores_execution_switch(self):
        parameters = base_parameters()
        self.assertEqual(hpo.fingerprint(parameters), hpo.fingerprint({**parameters, 'Args/hpo': True}))
        self.assertNotEqual(hpo.fingerprint(parameters), hpo.fingerprint({**parameters, hpo.LR+'scale': .5}))


if __name__ == '__main__': unittest.main()
