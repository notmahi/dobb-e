import warnings
from abc import ABC

import einops
import torch
import tqdm
from accelerate import Accelerator

accelerator = Accelerator()


# Mean and STD of full dataset based on {key}: (fps_subsample, control_timeskip)
ACT_MEAN = {
    (2, 2): (
        6.1696e-05,
        3.8626e-03,
        7.2460e-04,
        -1.4946e-04,
        -1.1363e-03,
        2.4650e-04,
        -3.0305e-03,
    ),
    (4, 1): (
        5.5267e-05,
        2.6280e-03,
        4.9859e-04,
        -9.3217e-05,
        -7.5208e-04,
        1.7098e-04,
        -2.0481e-03,
    ),
    (2, 3): (
        7.2950e-05,
        5.2102e-03,
        9.5942e-04,
        -1.9970e-04,
        -1.5271e-03,
        3.3574e-04,
        -4.1240e-03,
    ),
    (3, 2): (
        7.1451e-05,
        3.9607e-03,
        7.3883e-04,
        -1.4677e-04,
        -1.1402e-03,
        2.5654e-04,
        -3.1165e-03,
    ),
    (2, 4): (
        8.1396e-05,
        6.5950e-03,
        1.1942e-03,
        -2.5032e-04,
        -1.9254e-03,
        4.2876e-04,
        -5.2046e-03,
    ),
    (4, 2): (
        8.3840e-05,
        4.0697e-03,
        7.5490e-04,
        -1.4368e-04,
        -1.1511e-03,
        2.6208e-04,
        -3.1635e-03,
    ),
    (5, 2): (
        9.6409e-05,
        4.1856e-03,
        7.7121e-04,
        -1.3984e-04,
        -1.1743e-03,
        2.7600e-04,
        -3.3799e-03,
    ),
    (2, 5): (
        8.6731e-05,
        8.0177e-03,
        1.4270e-03,
        -3.0056e-04,
        -2.3345e-03,
        5.2420e-04,
        -6.3710e-03,
    ),
}

ACT_STD = {
    (2, 2): (0.0108, 0.0166, 0.0083, 0.0121, 0.0357, 0.0192, 0.0570),
    (4, 1): (0.0073, 0.0111, 0.0056, 0.0082, 0.0246, 0.0130, 0.0419),
    (2, 3): (0.0144, 0.0221, 0.0110, 0.0158, 0.0465, 0.0253, 0.0712),
    (3, 2): (0.0109, 0.0166, 0.0083, 0.0121, 0.0359, 0.0193, 0.0573),
    (2, 4): (0.0179, 0.0276, 0.0136, 0.0195, 0.0569, 0.0313, 0.0845),
    (4, 2): (0.0109, 0.0166, 0.0083, 0.0121, 0.0361, 0.0194, 0.0572),
    (5, 2): (0.0109, 0.0166, 0.0083, 0.0121, 0.0361, 0.0194, 0.0576),
    (2, 5): (0.0214, 0.0331, 0.0163, 0.0231, 0.0671, 0.0373, 0.0969),
}

ACT_MAX = {
    (4, 2): (0.1913, 0.2065, 0.1263, 0.1755, 0.6803, 0.3847, 0.9861),
    (5, 2): (0.1920, 0.1738, 0.1443, 0.2329, 0.6325, 0.4567, 0.9738),
    (2, 5): (0.3815, 0.3275, 0.2538, 0.3265, 1.2338, 0.6425, 0.9991),
}
ACT_MIN = {
    (4, 2): (-0.1609, -0.1652, -0.2317, -0.5132, -0.9144, -0.3120, -0.9414),
    (5, 2): (-0.1609, -0.1483, -0.2221, -0.2553, -0.5761, -0.2869, -0.9316),
    (2, 5): (-0.3287, -0.3300, -0.4670, -0.5106, -1.0652, -0.5910, -0.9966),
}


# Subclass this for action normalization
class NormalizeActions(ABC):
    def __init__(
        self,
        action_space: int = 7,
        normalize_action: bool = True,
        dynamic_norm: bool = False,
        fps_subsample: int = 2,
        control_time_skip: int = 3,
        relative_gripper: int = False,
    ):
        self.action_space = action_space
        self.normalize_action = normalize_action
        self.dynamic_norm = dynamic_norm
        self.fps_subsample = fps_subsample
        self.control_time_skip = control_time_skip
        self.relative_gripper = relative_gripper
        self.init_metrics()

    @torch.no_grad()
    def norm_action(self, action):
        # check if self.act_std is 0, and if so, raise a warning
        if torch.all(self.act_std == 0):
            warnings.warn(
                "Action std is 0, make sure you caluculated action statistics"
            )

        return (action - self.act_mean) / self.act_std

    @torch.no_grad()
    def denorm_action(self, action):
        # check if self.act_std is 0, and if so, raise a warning
        if torch.all(self.act_std == 0):
            warnings.warn(
                "Action std is 0, make sure you caluculated action statistics"
            )
        return action * self.act_std + self.act_mean

    def init_metrics(self):
        self.register_buffer("act_mean", torch.zeros(self.action_space))
        self.register_buffer("act_std", torch.ones(self.action_space))
        if self.normalize_action:
            if self.dynamic_norm:
                self.accumulate_mean = torch.zeros(self.action_space)
                self.accumulate_std = torch.zeros(self.action_space)
                self.n_accumulate = torch.zeros(1)
            else:
                self.act_mean = torch.Tensor(
                    ACT_MEAN[(self.fps_subsample, self.control_time_skip)]
                )
                self.act_std = torch.Tensor(
                    ACT_STD[(self.fps_subsample, self.control_time_skip)]
                )

                if not self.relative_gripper:
                    self.act_mean[-1] = 0
                    self.act_std[-1] = 1

                accelerator.print("Using static action normalization")
                accelerator.print(f"act_mean: {self.act_mean}")
                accelerator.print(f"act_std: {self.act_std}")

    def update_action_stats(self, train_dataloader, is_main_process):
        device = train_dataloader.device
        self.accumulate_mean, self.accumulate_std, self.n_accumulate = (
            self.accumulate_mean.to(device),
            self.accumulate_std.to(device),
            self.n_accumulate.to(device),
        )
        iterator = tqdm.tqdm(train_dataloader, disable=not is_main_process)
        iterator.set_description("Estimating action statistics")
        for x in iterator:
            images, act = x
            flatten_act = einops.rearrange(act, "b t c -> (b t) c")
            self.accumulate_mean += flatten_act.mean(dim=0)
            self.accumulate_std += flatten_act.std(dim=0)
            self.n_accumulate += torch.ones(1).to(device)
        # synchonize the action statistics across all processes in distributed training
        accelerator.wait_for_everyone()
        # torch.distributed.all_reduce(self.accumulate_mean)
        accelerator.reduce(self.accumulate_mean)
        accelerator.reduce(self.accumulate_std)
        accelerator.reduce(self.n_accumulate)
        accelerator.wait_for_everyone()

        self.act_mean = self.accumulate_mean / self.n_accumulate
        self.act_std = torch.sqrt(
            (self.accumulate_std / self.n_accumulate) - self.act_mean**2
        )

    def _begin_epoch(self, epoch, train_dataloader, is_main_process, **kwargs):
        if epoch == 0 and self.normalize_action and self.training and self.dynamic_norm:
            accelerator.print("Estimating action statistics")
            self.update_action_stats(train_dataloader, is_main_process)

    def _begin_batch(self, optimizer, **kwargs):
        # unroll act_mean and act_std list to scalar and assign it to act_mean_i and act_std_i
        log = {
            f"act_mean_{i}": self.act_mean[i].item() for i in range(self.action_space)
        }
        log.update(
            {f"act_std_{i}": self.act_std[i].item() for i in range(self.action_space)}
        )
        log.update({"lr": optimizer.param_groups[0]["lr"]})
        return log
