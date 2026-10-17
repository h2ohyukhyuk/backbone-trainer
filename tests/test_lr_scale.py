import copy
import math
import unittest
from unittest.mock import patch

import torch
import main
from utils.lr_interpolation import epoch_lr_curve
from utils.train_utils import get_scheduler


class LRScaleTests(unittest.TestCase):
    def setUp(self):
        self.conf = vars(main.parse_args([]))

    def test_scales_every_point_and_interpolated_value_without_mutation(self):
        settings = dict(points={0: .001, 2: .01, 5: .002}, interpolation='hermite')
        original = copy.deepcopy(settings)
        baseline = epoch_lr_curve(settings)
        for scale in (.5, 2.):
            scaled = epoch_lr_curve({**settings, 'scale': scale})
            for epoch in [-1, 0, .5, 1, 2, 3.5, 5, 10]:
                self.assertAlmostEqual(scaled(epoch), baseline(epoch) * scale)
        self.assertEqual(settings, original)

    def test_each_segment_is_monotone_and_never_overshoots(self):
        settings = dict(points={0: .0001, 2: .01, 15: .008, 45: .002, 70: .0003, 90: .00001}, scale=.8)
        curve = epoch_lr_curve(settings)
        points = sorted(settings['points'].items())
        for (left, a), (right, b) in zip(points, points[1:]):
            values = [curve(left + (right-left) * i/100) for i in range(101)]
            for value in values:
                self.assertGreaterEqual(value + 1e-15, min(a,b)*.8)
                self.assertLessEqual(value - 1e-15, max(a,b)*.8)
            for previous, current in zip(values, values[1:]):
                self.assertGreaterEqual((current-previous)*(b-a), -1e-15)
        with self.assertRaisesRegex(ValueError, 'Only Hermite'):
            epoch_lr_curve({**settings, 'interpolation': 'catmullrom'})

    def test_scheduler_preserves_group_multipliers(self):
        conf = self.conf
        conf.update(lr_scheduler='lambdalr', epochs=90, total_steps=900)
        conf['LAMBDALR_KWARGS']['scale'] = .5
        base_lr = conf['LR']
        opt = torch.optim.SGD([
            {'params': [torch.nn.Parameter(torch.zeros(()))], 'lr': base_lr},
            {'params': [torch.nn.Parameter(torch.zeros(()))], 'lr': base_lr * 10},
        ], lr=base_lr)
        scheduler = get_scheduler(opt, conf)
        for _ in range(20): opt.step(); scheduler.step()
        expected = conf['LAMBDALR_KWARGS']['points'][2] * .5
        self.assertAlmostEqual(scheduler.get_last_lr()[0], expected)
        self.assertAlmostEqual(scheduler.get_last_lr()[1], expected * 10)

    def test_invalid_scales_and_omitted_default(self):
        settings = self.conf['LAMBDALR_KWARGS']
        baseline = epoch_lr_curve({k: v for k, v in settings.items() if k != 'scale'})
        self.assertAlmostEqual(baseline(2), settings['points'][2])
        for value in (0, -1, True, float('nan'), float('inf')):
            with self.subTest(scale=value), self.assertRaises(ValueError):
                epoch_lr_curve({**settings, 'scale': value})

    def test_plot_uses_scaled_curve(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot
        import matplotlib.axes
        import matplotlib.figure
        args = main.parse_args(['--plot-lr', 'lambdalr'])
        args.LAMBDALR_KWARGS['scale'] = 2.
        plotted = []
        original_plot = matplotlib.axes.Axes.plot
        def capture(ax, x, y, *a, **kw):
            plotted.append(dict(zip(x, y)))
            return original_plot(ax, x, y, *a, **kw)
        with patch.object(matplotlib.axes.Axes, 'plot', capture), patch.object(matplotlib.figure.Figure, 'savefig'):
            main.plot_learning_rate(args)
        self.assertAlmostEqual(plotted[0][15], args.LAMBDALR_KWARGS['points'][15] * 2)


if __name__ == '__main__': unittest.main()
