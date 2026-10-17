"""Epoch-keyframe cubic interpolation for LambdaLR (no SciPy dependency)."""
from bisect import bisect_right
import math


def epoch_lr_curve(settings):
    """Shared training/preview curve: completed fractional epoch -> scaled absolute LR.

    Outside keyframes, hold the endpoint LR. Dataset size is irrelevant here.
    """
    if settings.keys() - {'points', 'scale', 'interpolation'}:
        raise ValueError('Unknown LambdaLR interpolation settings')
    points = settings.get('points')
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    if not isinstance(points, dict) or len(points) < 2:
        raise ValueError('interpolation points must be an epoch: LR mapping with at least two entries')
    if not all(finite(epoch) and epoch >= 0 for epoch in points):
        raise ValueError('points keys must be finite nonnegative epoch numbers')
    if not all(finite(lr) and lr >= 0 for lr in points.values()):
        raise ValueError('points values must be finite nonnegative learning rates')
    scale = settings.get('scale', 1.0)
    if not finite(scale) or scale <= 0:
        raise ValueError('LambdaLR scale must be finite and positive')
    if not all(math.isfinite(lr * scale) for lr in points.values()):
        raise ValueError('Scaled point learning rates must be finite')
    xs = sorted(points)
    ys = [points[epoch] for epoch in xs]
    if settings.get('interpolation', 'hermite') != 'hermite':
        raise ValueError('Only Hermite interpolation is supported')

    def curve(epoch):
        if epoch <= xs[0]:
            return ys[0] * scale
        if epoch >= xs[-1]:
            return ys[-1] * scale
        i = bisect_right(xs, epoch) - 1
        width = xs[i+1] - xs[i]
        t = (epoch - xs[i]) / width
        # Cubic Hermite with zero slope at both endpoints (smoothstep).
        blend = t * t * (3 - 2 * t)
        value = ys[i] * (1 - blend) + ys[i+1] * blend
        return value * scale

    return curve
