import abc

import torch


class AbstractLossFn(torch.nn.Module, abc.ABC):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    @abc.abstractmethod
    def forward(self, data, output, *args, **kwargs):
        raise NotImplementedError
