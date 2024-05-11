import pathlib
import warnings
from abc import ABC, abstractmethod
from typing import Union

import torch
import torch.nn as nn

ENCODER_LOADING_ERROR_MSG = (
    "Could not load encoder weights: defaulting to pretrained weights"
)


class AbstractEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def feature_dim(self):
        pass

    def transform(self, x):
        return x

    def to(self, device):
        self.device = device
        return super().to(device)

    @abstractmethod
    def forward(self, x):
        pass

    def load_weights(
        self, weight_path: Union[pathlib.Path, str], strict: bool = True
    ) -> bool:
        try:
            weight_dict = torch.load(weight_path, map_location="cpu")["model"]
            self.model.load_state_dict(weight_dict, strict=strict)
        except RuntimeError:
            warnings.warn(
                "Couldn't load weights from file, trying to load encoder only"
            )
            try:
                enc_weights_filtered = {
                    k.replace("model.", ""): v
                    for k, v in weight_dict.items()
                    if k.startswith("model.")
                }
                self.model.load_state_dict(enc_weights_filtered, strict=strict)
            except RuntimeError:
                warnings.warn(
                    "Couldn't load weights from file, trying to filter for encoder weights"
                )
                enc_weights_filtered = {
                    k.replace("encoder.", ""): v
                    for k, v in weight_dict.items()
                    if k.startswith("encoder.")
                }
                self.model.load_state_dict(enc_weights_filtered, strict=strict)
        except:
            warnings.warn(ENCODER_LOADING_ERROR_MSG)
