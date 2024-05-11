"""
A generic model wrapper for timm encoders.
"""
import pathlib
from typing import Union

import einops
import timm

from dobbe.models.encoders.abstract_base_encoder import AbstractEncoder
from dobbe.utils.decord_transforms import create_transform


class TimmModel(AbstractEncoder):
    def __init__(
        self,
        model_name: str = "hf-hub:notmahi/dobb-e",
        pretrained: bool = True,
        weight_path: Union[None, str, pathlib.Path] = None,
    ):
        super().__init__()
        self._model_name = model_name

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        if weight_path:
            self.load_weights(weight_path, strict=False)

    def transform(self, x):
        return x

    @property
    def feature_dim(self):
        return self.model.num_features

    def to(self, device):
        self.model.to(device)
        return self

    def forward(self, x):
        return self.model(self.transform(x))


class TimmSSL(TimmModel):
    def __init__(
        self,
        model_name: str = "hf-hub:notmahi/dobb-e",
        pretrained=True,
        override_aug_kwargs=dict(
            hflip=0.0, vflip=0.0, scale=(1.0, 1.0), crop_pct=0.875
        ),
        weight_path: Union[None, str, pathlib.Path] = None,
    ):
        super().__init__(model_name, pretrained=pretrained)
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        # Now define the transforms.
        data_cfg.update(override_aug_kwargs)
        data_cfg["is_training"] = True
        self._train_transform = create_transform(**data_cfg)
        data_cfg["is_training"] = False
        self._test_transform = create_transform(**data_cfg)
        if weight_path is not None:
            self.load_weights(weight_path, strict=True)

    def transform(self, x):
        return (
            self._train_transform(x) if self.model.training else self._test_transform(x)
        )

    def forward(self, x):
        # Split the input into frames and labels.
        images, *labels = x
        # Flatten the frames into a single batch.
        flattened_images = einops.rearrange(images, "b t c h w -> (b t) c h w")
        # Transform and pass through the model.
        result = self.model(self.transform(flattened_images))
        # Unflatten the result.
        result = einops.rearrange(result, "(b t) c -> b t c", b=images.shape[0])
        return result


class TimmSSLByolMoco(TimmSSL):
    def transform(self, x):
        # Don't transform the image here since the SSL recipe does it internally.
        return x

    def forward(self, x):
        if not isinstance(x, list) and not isinstance(x, tuple):
            flattened_images = x
        else:
            flattened_images, *labels = x
        result = self.model(self.transform(flattened_images))
        return result
