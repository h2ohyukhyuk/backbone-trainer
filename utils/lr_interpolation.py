"""Epoch-keyframe cubic interpolation for LambdaLR (no SciPy dependency)."""
from bisect import bisect_right
import math


def epoch_lr_curve(settings):
    """Return absolute LR(epoch). Outside keyframes, hold the endpoint LR."""
    points = settings.get('points')
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    if not isinstance(points, dict) or len(points) < 2:
        raise ValueError('interpolation points must be an epoch: LR mapping with at least two entries')
    if not all(finite(epoch) and epoch >= 0 for epoch in points):
        raise ValueError('points keys must be finite nonnegative epoch numbers')
    if not all(finite(lr) and lr >= 0 for lr in points.values()):
        raise ValueError('points values must be finite nonnegative learning rates')
    xs = sorted(points)
    ys = [points[epoch] for epoch in xs]
    method = settings.get('interpolation', 'hermite')
    method = {'hemit': 'hermite', 'catmulrome': 'catmullrom',
              'catmull_rom': 'catmullrom', 'catmull-rom': 'catmullrom'}.get(method, method)
    if method == 'hermite':
        slopes = settings.get('tangents') or [0.0] * len(xs)
        if not isinstance(slopes, list) or len(slopes) != len(xs) or not all(finite(v) for v in slopes):
            raise ValueError('hermite tangents must match epochs and contain finite LR/epoch slopes')
    elif method == 'catmullrom':
        # Epoch-aware centered secants, with one-sided secants at endpoints.
        slopes = [(ys[1] - ys[0]) / (xs[1] - xs[0])]
        slopes += [(ys[i+1] - ys[i-1]) / (xs[i+1] - xs[i-1]) for i in range(1, len(xs)-1)]
        slopes += [(ys[-1] - ys[-2]) / (xs[-1] - xs[-2])]
    else:
        raise ValueError('interpolation must be hermite or catmullrom')

    def curve(epoch):
        if epoch <= xs[0]:
            return ys[0]
        if epoch >= xs[-1]:
            return ys[-1]
        i = bisect_right(xs, epoch) - 1
        width = xs[i+1] - xs[i]
        t = (epoch - xs[i]) / width
        value = ((2*t**3 - 3*t**2 + 1) * ys[i]
                 + (t**3 - 2*t**2 + t) * width * slopes[i]
                 + (-2*t**3 + 3*t**2) * ys[i+1]
                 + (t**3 - t**2) * width * slopes[i+1])
        return max(0.0, value)

    return curve
