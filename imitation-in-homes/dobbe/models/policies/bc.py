# naive bc model that takes in encoder, actionspace and predicts actions
import warnings

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from models.policies.depth_net import DepthBC
from models.policies.normalize_actions import NormalizeActions

ENCODER_LOADING_ERROR_MSG = (
    "Could not load encoder weights: trying to load as a BC model. "
    "This will throw away the fc layers weights so be careful.\n"
    "If you want to load the BC model with fc layers, pass the "
    "weight path to model_weight_pth instead of enc_weight_pth"
)


class BCModel(nn.Module, NormalizeActions):
    def __init__(
        self,
        encoder: nn.Module,
        img_size: int = 256,
        action_space: int = 7,
        h_dim: int = 512,
        enc_weight_pth: str = None,
        model_weight_pth: str = None,
        freeze_encoder: bool = False,
        freeze_encoder_for: int = 0,
        normalize_action: bool = True,
        dynamic_norm: bool = False,
        fps_subsample: int = 2,
        control_time_skip: int = 3,
        relative_gripper: bool = False,
        load_params_with_enc: bool = True,
    ):
        nn.Module.__init__(self)
        NormalizeActions.__init__(
            self,
            action_space,
            normalize_action=normalize_action,
            dynamic_norm=dynamic_norm,
            fps_subsample=fps_subsample,
            control_time_skip=control_time_skip,
            relative_gripper=relative_gripper,
        )

        self.encoder = encoder
        self.h_dim = h_dim
        self.load_params_with_enc = load_params_with_enc
        self.freeze_encoder_for = freeze_encoder_for

        with torch.no_grad():
            self.enc_dim = self.encoder(
                (torch.zeros(1, 1, 3, img_size, img_size), 0)
            ).shape[-1]

        self.fc1 = nn.Linear(self.enc_dim, self.h_dim)
        self.fc2 = nn.Linear(self.h_dim, self.action_space)
        self.img_transform = T.Resize((256, 256), antialias=True)

        if enc_weight_pth is not None:
            # first try to load the model as an encoder.
            # If the keys dont match, load the model as a bc model and then throw away the fc layers
            self.encoder.load_weights(enc_weight_pth, strict=True)
            if self.load_params_with_enc:
                self.load_non_mlp_params(enc_weight_pth)

        if model_weight_pth is not None:
            self.load_state_dict(torch.load(model_weight_pth)["model"])

        if freeze_encoder or freeze_encoder_for > 0:
            # self.encoder.freeze()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def load_non_mlp_params(self, enc_weight_pth: str):
        # Use this when the weights contain the encoder as a submodule
        # This will first remove all the keys except .encoder and then rename the keys to remove the .encoder
        weight_dict = torch.load(enc_weight_pth, map_location="cpu")["model"]
        # Load everything else except fc1, fc2, and encoder.
        self.load_state_dict(
            {
                k: v
                for k, v in weight_dict.items()
                if not k.startswith("encoder.")
                and not k.startswith("fc1")
                and not k.startswith("fc2")
            },
            strict=False,
        )

    def _begin_epoch(self, epoch, train_dataloader, is_main_process, **kwargs):
        if epoch == self.freeze_encoder_for:
            for param in self.encoder.parameters():
                param.requires_grad = True

    def step(self, input, *args, **kwargs):
        self.eval()
        normalized_image = self.img_transform(input[0].squeeze(1))
        norm_action, _ = self((normalized_image.unsqueeze(1), input[1]))
        return self.denorm_action(norm_action).squeeze(), {}

    def forward(self, x):
        images, *labels = x

        out = self.encoder(x)
        bs = out.shape[0]
        out = einops.rearrange(out, "b t c -> (b t) c")
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = einops.rearrange(out, "(b t) a -> b t a", b=bs)
        return out, {"act_mean": self.act_mean, "act_std": self.act_std}

    def reset(self):
        pass


class BCDepthModel(BCModel):
    def __init__(
        self,
        encoder: nn.Module,
        img_size: int = 256,
        action_space: int = 7,
        h_dim: int = 512,
        enc_weight_pth: str = None,
        model_weight_pth: str = None,
        freeze_encoder: bool = False,
        freeze_encoder_for: int = 0,
        normalize_action: bool = True,
        dynamic_norm: bool = False,
        fps_subsample: int = 2,
        control_time_skip: int = 3,
        relative_gripper: bool = False,
        use_depth: bool = False,
        load_params_with_enc: bool = True,
        depth_patch_dim: int = 4,
        depth_kernel_size: tuple = (12, 16),
    ):
        super().__init__(
            encoder=encoder,
            img_size=img_size,
            action_space=action_space,
            h_dim=h_dim,
            enc_weight_pth=None,
            model_weight_pth=None,  # model_weight_pth is None because we want to use the new fc1 layer and depth net
            freeze_encoder=False,
            freeze_encoder_for=freeze_encoder_for,
            normalize_action=normalize_action,
            dynamic_norm=dynamic_norm,
            fps_subsample=fps_subsample,
            control_time_skip=control_time_skip,
            relative_gripper=relative_gripper,
            load_params_with_enc=load_params_with_enc,
        )

        self.fc1 = nn.Linear(
            self.enc_dim * 2 - self.enc_dim % depth_patch_dim**2, self.h_dim
        )
        self._use_depth = use_depth
        self._depth_net = DepthBC(
            self.enc_dim, kernel_size=depth_kernel_size, patch_dim=depth_patch_dim
        )

        if enc_weight_pth is not None:
            # first try to load the model as an encoder. If the keys dont match, load the model as a bc model and then throw away the fc layers
            try:
                self.encoder.load_state_dict(torch.load(enc_weight_pth)["model"])
            except RuntimeError:
                warnings.warn(ENCODER_LOADING_ERROR_MSG)
                self.load_encoder_only(enc_weight_pth)

        if model_weight_pth is not None:
            self.load_state_dict(torch.load(model_weight_pth)["model"])

        if freeze_encoder:
            # self.encoder.freeze()
            for param in self.encoder.parameters():
                param.requires_grad = False

    def step(self, input, *args, **kwargs):
        self.eval()
        normalized_image = self.img_transform(input[0].squeeze(1))
        if self._use_depth:
            norm_action, _ = self((normalized_image.unsqueeze(1), input[1], input[2]))
        else:
            norm_action, _ = self((normalized_image.unsqueeze(1), input[1]))
        return self.denorm_action(norm_action).squeeze(), {}

    def forward(self, x):
        if self._use_depth:
            _, depths, *_ = x
            out = torch.cat([self.encoder(x), self._depth_net(depths)], dim=-1)
        else:
            enc_out = self.encoder(x)
            out = torch.cat([enc_out, enc_out], dim=-1)

        bs = out.shape[0]
        out = einops.rearrange(out, "b t c -> (b t) c")
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        out = einops.rearrange(out, "(b t) a -> b t a", b=bs)
        return out, {"act_mean": self.act_mean, "act_std": self.act_std}
