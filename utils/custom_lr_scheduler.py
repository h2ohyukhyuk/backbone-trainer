from torch.optim.lr_scheduler import _LRScheduler
import warnings

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        power: The power of the polynomial.
    """
    def __init__(self, optimizer, max_decay_steps, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.power = power
        super(PolynomialLRDecay, self).__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [base_lr * (1 - self.last_epoch / self.max_decay_steps) ** self.power for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_decay_steps) ** self.power for base_lr in self.base_lrs]