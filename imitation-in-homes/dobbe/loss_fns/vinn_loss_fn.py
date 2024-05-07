from typing import Optional

import torch

from loss_fns.abstract_loss_fn import AbstractLossFn


class VINNLoss(AbstractLossFn):
    def __init__(self, model: Optional[torch.nn.Module] = None, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

        self.loss_fn = torch.nn.MSELoss()

    def forward(self, data, output, *args, **kwargs):
        if output is None:
            return torch.zeros(1, requires_grad=True), {}
        else:
            _, y = data
            y = y.squeeze()

            loss = self.loss_fn(output, y).item()
            translation_val_loss = self.loss_fn(output[:, :3], y[:, :3]).item()
            rotation_val_loss = self.loss_fn(output[:, 3:6], y[:, 3:6]).item()
            gripper_val_loss = self.loss_fn(output[:, 6:], y[:, 6:]).item()

            return torch.ones(1) * loss, {
                "val_loss": loss,
                "translational_val_loss": translation_val_loss,
                "rotational_val_loss": rotation_val_loss,
                "gripper_val_loss": gripper_val_loss,
            }
