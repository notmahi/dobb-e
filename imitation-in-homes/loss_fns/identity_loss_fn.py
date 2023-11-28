from typing import Optional

import torch

from loss_fns.abstract_loss_fn import AbstractLossFn


class IdentityLossFn(AbstractLossFn):
    def __init__(self, model: Optional[torch.nn.Module] = None, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def forward(self, data, output, *args, **kwargs):
        return output, {"loss": output}
