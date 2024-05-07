import torch

from loss_fns.abstract_loss_fn import AbstractLossFn
from models.ssl.moco.builder import MoCo


class MocoLossFn(AbstractLossFn):
    def __init__(
        self,
        model: torch.nn.Module = None,
        dim=256,
        mlp_dim=4096,
        T=1.0,
        moco_m=0.99,
        *args,
        **kwargs
    ):
        super().__init__(model, *args, **kwargs)
        self.encoder = model

        self.learner = MoCo(self.encoder, dim, mlp_dim, T, moco_m)

    def _begin_epoch(self, optimizer, **kwargs):
        self.learner._begin_epoch(**kwargs)
        lr_0 = optimizer.param_groups[0]["lr"]
        lr_neg1 = optimizer.param_groups[-1]["lr"]
        return {"lr_0": lr_0, "lr_neg1": lr_neg1}

    def _begin_batch(self, **kwargs):
        return {"moco_m": self.learner.moco_m}

    def forward(self, data, output, *args, **kwargs):
        loss = self.learner(data)
        return loss, {"loss": loss}
