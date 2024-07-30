from typing import Optional

import torch
from byol_pytorch import BYOL
from einops import rearrange

from dobbe.loss_fns.abstract_loss_fn import AbstractLossFn
from dobbe.utils.ssl_transfroms import IMAGENET_MEAN, IMAGENET_STD, get_byol_transforms


class ByolLossFn(AbstractLossFn):
    def __init__(
        self,
        model: torch.nn.Module = None,
        img_size=256,
        img_mean: Optional[torch.Tensor] = IMAGENET_MEAN,
        img_std: Optional[torch.Tensor] = IMAGENET_STD,
        *args,
        **kwargs
    ):
        super().__init__(model, *args, **kwargs)
        self.encoder = model
        self.augment1 = get_byol_transforms(img_size, img_mean, img_std)
        self.augment2 = None

        self.learner = BYOL(
            self.encoder,
            image_size=img_size,
            hidden_layer=-1,
            augment_fn=self.augment1,
            augment_fn2=self.augment2,
        )

    def _begin_epoch(self, optimizer, **kwargs):
        lr_0 = optimizer.param_groups[0]["lr"]
        lr_neg1 = optimizer.param_groups[-1]["lr"]
        return {"lr_0": lr_0, "lr_neg1": lr_neg1}

    def forward(self, data, output, *args, **kwargs):
        x, actions = data
        log_dict = {}
        flattened_images = rearrange(x, "b t c h w -> (b t) c h w")
        if self.training:
            self.learner.update_moving_average()
        else:
            with torch.no_grad():
                batch_rep = self.encoder(flattened_images)
                rep_mean = batch_rep.mean()
                rep_std = batch_rep.std()
                rep_median = batch_rep.median()
                log_dict["rep_mean"] = rep_mean
                log_dict["rep_std"] = rep_std
                log_dict["rep_median"] = rep_median

        loss = self.learner(flattened_images)
        log_dict["loss"] = loss
        return loss, log_dict
