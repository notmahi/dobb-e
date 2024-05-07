import random
from collections import OrderedDict
from datetime import timedelta
from typing import Any

import accelerate
import numpy as np
import torch

from . import *

# import accelerate


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_accelerate(experiment, num_epochs, task_title, save_every):
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = accelerate.Accelerator(
        log_with=["wandb"],
        kwargs_handlers=[
            accelerate.InitProcessGroupKwargs(timeout=timedelta(hours=1.5)),
            ddp_kwargs,
        ],
    )
    accelerator.init_trackers(
        "experiment",
        config={
            "num_epochs": 1000,
            "task_title": task_title,
            "save_every": 20,
        },
    )

    return accelerator, ddp_kwargs


class AverageMeter:
    def __init__(self, value=None):
        super().__init__()
        if value is None:
            self._sum = 0
            self._count = 0
        else:
            self._sum = value
            self._count = 1

    def update(self, value, n=1):
        self._sum += value * n
        self._count += n
        self._avg = self._sum / self._count

    @property
    def avg(self):
        return self._avg


class AveragingDict(OrderedDict):
    def __init__(self, name="train", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name

    def update(self, other):
        for k, v in other.items():
            if k not in self:
                self[k] = AverageMeter(v)
            self[k].update(v)

    def __str__(self):
        return ", ".join(f"{self.name}/{k}: {v.avg:.2E}" for k, v in self.items())

    @property
    def full_summary(self):
        return {f"{self.name}/{k}": v.avg for k, v in self.items()}

    @property
    def summary(self):
        return {f"{self.name}/{k}": f"{v.avg:.2E}" for k, v in self.items()}


class Callbacks:
    """
    This class is used to implement callback functions inside training framework that
    passes, training params to callback functions inside model and loss objects.

    This is useful for custom logging inside models and losses like adding image
    visualisation, plotting hyperparameters like learning rate, etc.
    """

    def __init__(self):
        pass

    def set_workspace(self, workspace, accelerator):
        self.workspace = workspace
        self.accelerator = accelerator
        self._model_has_begin_epoch = hasattr(
            self.accelerator.unwrap_model(self.workspace.model), "_begin_epoch"
        )
        self._model_has_begin_batch = hasattr(
            self.accelerator.unwrap_model(self.workspace.model), "_begin_batch"
        )

        self._loss_fn_has_begin_epoch = hasattr(
            self.accelerator.unwrap_model(self.workspace.loss_fn), "_begin_epoch"
        )
        self._loss_fn_has_begin_batch = hasattr(
            self.accelerator.unwrap_model(self.workspace.loss_fn), "_begin_batch"
        )

    def begin_epoch(self):
        self.batch_step = 0

        if self._model_has_begin_epoch:
            model = self.accelerator.unwrap_model(self.workspace.model)
            is_main_process = self.accelerator.is_local_main_process
            begin_epoch_log = model._begin_epoch(
                epoch=self.workspace._epoch,
                epochs=self.workspace._cfg.num_epochs,
                train_dataloader=self.workspace._train_dataloaders,
                test_dataloader=self.workspace._test_dataloaders,
                optimizer=self.workspace.optimizer,
                scheduler=self.workspace.scheduler,
                is_main_process=is_main_process,
            )
            if begin_epoch_log is not None:
                self.accelerator.log(begin_epoch_log)

        if self._loss_fn_has_begin_epoch:
            loss_fn = self.accelerator.unwrap_model(self.workspace.loss_fn)
            is_main_process = self.accelerator.is_local_main_process
            begin_epoch_log = loss_fn._begin_epoch(
                epoch=self.workspace._epoch,
                epochs=self.workspace._cfg.num_epochs,
                train_dataloader=self.workspace._train_dataloaders,
                test_dataloader=self.workspace._test_dataloaders,
                optimizer=self.workspace.optimizer,
                scheduler=self.workspace.scheduler,
                is_main_process=is_main_process,
            )
            if begin_epoch_log is not None:
                self.accelerator.log(begin_epoch_log)

    def begin_batch(self):
        if self._model_has_begin_batch:
            model = self.accelerator.unwrap_model(self.workspace.model)
            is_main_process = self.accelerator.is_local_main_process
            begin_batch_log = model._begin_batch(
                epoch=self.workspace._epoch,
                batch_step=self.batch_step,
                epochs=self.workspace._cfg.num_epochs,
                train_dataloader=self.workspace._train_dataloaders,
                test_dataloader=self.workspace._test_dataloaders,
                optimizer=self.workspace.optimizer,
                scheduler=self.workspace.scheduler,
                is_main_process=is_main_process,
            )
            if begin_batch_log is not None:
                self.accelerator.log(begin_batch_log)

        if self._loss_fn_has_begin_batch:
            loss_fn = self.accelerator.unwrap_model(self.workspace.loss_fn)
            is_main_process = self.accelerator.is_local_main_process
            begin_batch_log = loss_fn._begin_batch(
                epoch=self.workspace._epoch,
                batch_step=self.batch_step,
                epochs=self.workspace._cfg.num_epochs,
                train_dataloader=self.workspace._train_dataloaders,
                test_dataloader=self.workspace._test_dataloaders,
                optimizer=self.workspace.optimizer,
                scheduler=self.workspace.scheduler,
                is_main_process=is_main_process,
            )
            if begin_batch_log is not None:
                self.accelerator.log(begin_batch_log)

        self.batch_step += 1


def log_wandb_image(image, label):
    """
    image: torch.Tensor or np.ndarray
    label: str, label for the image to categorize it in wandb

    returns: dict, {label: wandb.Image(image)}
    """
    import wandb

    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
    while image.shape[0] == 1:
        image = image.squeeze(0)
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    image = wandb.Image(image)
    return {label: image}
