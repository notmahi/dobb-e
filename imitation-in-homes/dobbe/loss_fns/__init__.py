from abc import ABC, abstractmethod
from typing import Optional

import torch


class BinarizeGripper(ABC):
    def __init__(
        self,
        model: Optional[torch.nn.Module],
        binarize_gripper: bool = False,
        threshold: float = 0.5,
        upper_value: float = 1.0,
        lower_value: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__()
        assert (
            not binarize_gripper or not model.relative_gripper
        ), "Binarize gripper and relative gripper cannot be used together"

        self.binarize_gripper = binarize_gripper
        self.threshold = threshold
        self.upper_value = upper_value
        self.lower_value = lower_value

    @abstractmethod
    def binarize_gripper(self, actions):
        # here the actions will be of shape (batch_size, seq_len, action_space)
        # last element in action space is gripper values between 0 and 1

        # binarize the gripper values
        if self.binarize_gripper:
            actions[:, :, -1] = torch.where(
                actions[:, :, -1] > self.threshold, self.upper_value, self.lower_value
            )
        return actions
