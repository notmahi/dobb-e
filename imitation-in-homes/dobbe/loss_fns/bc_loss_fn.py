from typing import Optional

import torch
from einops import rearrange
from torchvision.ops import MLP

from dobbe.loss_fns.abstract_loss_fn import AbstractLossFn


class BCLossFn(AbstractLossFn):
    def __init__(
        self,
        action_dim: int,
        model: Optional[torch.nn.Module] = None,
        network_type: str = "mlp",
        network_depth: int = 2,
        network_width: int = 256,
        network_activation=torch.nn.ReLU,
        reduction: str = "mean",
        *args,
        **kwargs
    ):
        super().__init__(model, *args, **kwargs)
        input_dim = model.feature_dim
        output_dim = action_dim
        if network_type == "mlp":
            self._network = MLP(
                in_channels=input_dim,
                hidden_channels=((network_width,) * network_depth) + (output_dim,),
                activation_layer=network_activation,
            )
        else:
            raise NotImplementedError("Only MLP is supported for now.")
        self._loss_fn = torch.nn.MSELoss(reduction=reduction)

    def forward(self, data, output, *args, **kwargs):
        start_state, *_ = torch.unbind(output.detach(), dim=1)
        *_, actions = data
        first_action = actions[:, 0, :]
        # The loss is the triplet loss.
        loss = self._loss_fn(self._network(start_state), first_action)
        return loss, {
            "loss": loss,
        }


class BCPolicyLossFn(AbstractLossFn):
    def __init__(self, model: Optional[torch.nn.Module] = None, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.loss_fn = torch.nn.MSELoss()

    def _normalize(self, x, mean, std):
        return (x - mean) / std

    def forward(self, data, output, *args, **kwargs):
        pred_act, act_stats = output
        *_, actions = data

        norm_actions = self._normalize(
            actions, act_stats["act_mean"], act_stats["act_std"]
        )
        loss = self.loss_fn(pred_act, norm_actions)
        translation_loss = self.loss_fn(
            pred_act[:, :, :3], norm_actions[:, :, :3]
        ).detach()
        rotation_loss = self.loss_fn(
            pred_act[:, :, 3:6], norm_actions[:, :, 3:6]
        ).detach()
        gripper_loss = self.loss_fn(pred_act[:, :, 6:], norm_actions[:, :, 6:]).detach()
        return loss, {
            "loss": loss.detach(),
            "translation_loss": translation_loss,
            "rotation_loss": rotation_loss,
            "gripper_loss": gripper_loss,
        }


class BCPolicyLossFn(AbstractLossFn):
    def __init__(self, model: Optional[torch.nn.Module] = None, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.loss_fn = torch.nn.MSELoss()

    def _normalize(self, x, mean, std):
        return (x - mean) / std

    def forward(self, data, output, *args, **kwargs):
        pred_act, act_stats = output
        *_, actions = data

        norm_actions = self._normalize(
            actions, act_stats["act_mean"], act_stats["act_std"]
        )
        loss = self.loss_fn(pred_act, norm_actions)
        translation_loss = self.loss_fn(
            pred_act[:, :, :3], norm_actions[:, :, :3]
        ).detach()
        rotation_loss = self.loss_fn(
            pred_act[:, :, 3:6], norm_actions[:, :, 3:6]
        ).detach()
        gripper_loss = self.loss_fn(pred_act[:, :, 6:], norm_actions[:, :, 6:]).detach()
        return loss, {
            "loss": loss.detach(),
            "translation_loss": translation_loss,
            "rotation_loss": rotation_loss,
            "gripper_loss": gripper_loss,
        }


class BCClsPolicyLossFn(BCPolicyLossFn):
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        MAX_BOUND_M: int = 1e10,
        MIN_BOUND_M: int = -1e10,
        *args,
        **kwargs
    ):
        super().__init__(model, *args, **kwargs)
        self.action_space = model.action_space
        self.bins = model.bins
        self.boundaries = model.boundaries
        self.bin_centers = model.bin_centers
        self.boundaries_wide = self.boundaries.clone()
        self.boundaries_wide[:, 0], self.boundaries_wide[:, -1] = (
            MIN_BOUND_M,
            MAX_BOUND_M,
        )

        self.device = next(model.parameters()).device

        self.boundaries_wide = self.boundaries_wide.to(self.device)
        self.bin_centers = self.bin_centers.to(self.device)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn_reg = torch.nn.MSELoss()

    def _normalize(self, x, mean, std):
        return (x - mean) / std

    def discrete_to_continuous(self, discrete_action):
        # discrete_action is (bs, action_space*bin)
        # convert to continuous action
        # continuous_action is (bs, action_space)
        discrete_action = rearrange(
            discrete_action, "b t (x y) -> (b t) x y", x=self.bins
        )
        pred_idx = torch.argmax(discrete_action, dim=1)
        pred_act_scalar = self.bin_centers[pred_idx]

        pred_act_scalar = rearrange(
            pred_act_scalar, "(b t) x -> b t x", b=discrete_action.shape[0]
        )
        return pred_act_scalar

    def forward(self, data, output, *args, **kwargs):
        pred_act, act_stats = output
        *_, actions = data
        norm_actions = self._normalize(
            actions, act_stats["act_mean"], act_stats["act_std"]
        )
        # norm actions have 7 dimensions with each dimension normalized (mean 0, std 1)
        # discretize all the dimensions into bins
        device = norm_actions.device
        discretized_actions = torch.zeros_like(norm_actions).to(device)
        for i in range(self.action_space):
            discretized_actions[:, :, i] = (
                torch.bucketize(norm_actions[:, :, i], self.boundaries_wide[i]) - 1
            )

        # pred_act is logits of shape (batch_size, seq_len, 7*bins)
        batch_size, seq_len, _ = pred_act.shape
        pred_act_reshaped = rearrange(pred_act, "b t (x y) -> (b t) x y", x=self.bins)
        discretized_actions = rearrange(
            discretized_actions, "b t a -> (b t) a", a=self.action_space
        )

        loss_cls = self.loss_fn(pred_act_reshaped, discretized_actions.long())

        # get index of the bin with the highest probability

        pred_idx = torch.argmax(pred_act_reshaped, dim=1)
        # get the bin center of the bin with the highest probability
        # pred_act_scalar is (batch_size*seq_len, action_space) which calculated using  pred_idx (bs*seq_len, action_space) to index into bin_centers (action_space, bins)
        pred_act_scalar = self.bin_centers[torch.arange(self.action_space), pred_idx]

        pred_act_scalar = rearrange(
            pred_act_scalar,
            "(b t) a -> b t a",
            b=batch_size,
            t=seq_len,
            a=self.action_space,
        )
        # calculate regression loss
        loss_reg = self.loss_fn_reg(pred_act_scalar, norm_actions).detach()

        translation_loss_cls = self.loss_fn(
            pred_act_reshaped[:, :, :3], discretized_actions[:, :3].long()
        ).detach()

        rotation_loss_cls = self.loss_fn(
            pred_act_reshaped[:, :, 3:6], discretized_actions[:, 3:6].long()
        ).detach()

        gripper_loss_cls = self.loss_fn(
            pred_act_reshaped[:, :, 6:], discretized_actions[:, 6:].long()
        ).detach()

        translation_loss_reg = self.loss_fn_reg(
            pred_act_scalar[:, :, :3], norm_actions[:, :, :3]
        ).detach()

        rotation_loss_reg = self.loss_fn_reg(
            pred_act_scalar[:, :, 3:6], norm_actions[:, :, 3:6]
        ).detach()

        gripper_loss_reg = self.loss_fn_reg(
            pred_act_scalar[:, :, 6:], norm_actions[:, :, 6:]
        ).detach()

        return loss_cls, {
            "loss_cls": loss_cls.detach(),
            "translation_loss_cls": translation_loss_cls,
            "rotation_loss_cls": rotation_loss_cls,
            "gripper_loss_cls": gripper_loss_cls,
            "loss_reg": loss_reg.detach(),
            "translation_loss_reg": translation_loss_reg,
            "rotation_loss_reg": rotation_loss_reg,
            "gripper_loss_reg": gripper_loss_reg,
        }
