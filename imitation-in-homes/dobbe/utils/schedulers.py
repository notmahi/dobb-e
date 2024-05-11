import math
import warnings

from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealWithWarmupLR(LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Adapted from karpathy/nanoGPT
    https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py#L228

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_epochs (int): The number of epochs for warmup
        lr_decay_epoch (int): The index of lr_decay epoch, at which we reach min_LR. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
        min_lr_multiplier: The lowest lr multiplier the schedule will ever get to.

    Example:
        >>> warmup_epochs = 10
        >>> lr_decay_epochs = 600
        >>> scheduler = CosineAnnealWithWarmupLR(optimizer, warmup_epochs, lr_decay_epochs)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        lr_decay_epochs,
        last_epoch=-1,
        min_lr_multiplier=0.1,
        verbose=False,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.last_epoch = last_epoch
        self.lr_decay_epoch = lr_decay_epochs
        self.min_lr_multiplier = min_lr_multiplier

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )

        return [
            base_lr * self._calculate_lr_formula(self.last_epoch)
            for base_lr in self.base_lrs
        ]

    def _calculate_lr_formula(self, last_epoch):
        if last_epoch < self.warmup_epochs:
            return (last_epoch + 1) / self.warmup_epochs
        elif last_epoch > self.lr_decay_epoch:
            return self.min_lr_multiplier
        decay_ratio = (last_epoch - self.warmup_epochs) / (
            self.lr_decay_epoch - self.warmup_epochs
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr_multiplier + coeff * (1 - self.min_lr_multiplier)
