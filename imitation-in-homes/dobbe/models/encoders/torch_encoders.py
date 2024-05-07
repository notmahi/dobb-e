import pathlib
from typing import Union

import einops
import timm
import torch.nn as nn
import torchvision.models as models

from dobbe.models.encoders.abstract_base_encoder import AbstractEncoder
from dobbe.utils.decord_transforms import create_transform


class TorchModel(AbstractEncoder):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        weight_path: Union[None, str, pathlib.Path] = None,
    ):
        super().__init__()
        if not model_name.startswith("resnet"):
            raise ValueError("Unsupported encoder type: {}".format(model_name))
        self._model_name = model_name
        # create resnet model using torchvision models and model_name
        self.model = models.__dict__[model_name](pretrained=pretrained)
        self._num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        # Load weights if provided.
        if weight_path is not None:
            self.load_weights(weight_path, strict=False)

    def transform(self, x):
        return x

    @property
    def feature_dim(self):
        return self._num_features

    def to(self, device):
        self.model.to(device)
        return self

    def forward(self, x):
        return self.model(self.transform(x))


class TorchSSL(TorchModel):
    def __init__(
        self,
        model_name,
        pretrained=True,
        override_aug_kwargs=dict(
            hflip=0.0, vflip=0.0, scale=(1.0, 1.0), crop_pct=0.875
        ),
        weight_path: Union[None, str, pathlib.Path] = None,
    ):
        super().__init__(model_name, pretrained=pretrained)
        timm_model = timm.create_model(model_name, pretrained=False)
        data_cfg = timm.data.resolve_data_config(timm_model.pretrained_cfg)
        del timm_model

        # Now define the transforms.
        data_cfg.update(override_aug_kwargs)
        data_cfg["is_training"] = True
        self._train_transform = create_transform(**data_cfg)
        data_cfg["is_training"] = False
        self._test_transform = create_transform(**data_cfg)

        # Load weights if provided.
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


class TorchSSLByolMoco(TorchSSL):
    def transform(self, x):
        # Don't transform the image here, since the SSL recipe does it internally.
        return x

    def forward(self, x):
        if not isinstance(x, list) and not isinstance(x, tuple):
            flattened_images = x
        else:
            flattened_images, *_ = x
        result = self.model(self.transform(flattened_images))
        return result
