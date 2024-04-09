import logging
import shutil
from datetime import timedelta
from itertools import chain
from os import environ
from pathlib import Path
from typing import Iterable, Tuple

import hydra
import torch
import tqdm
from accelerate import (
    Accelerator,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from utils import AveragingDict, Callbacks, set_seed_everywhere

logger = logging.getLogger(__name__)
if environ.get("ACCEL_DEBUG", False):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    kwargs_handlers = [
        InitProcessGroupKwargs(timeout=timedelta(hours=1.5)),
        ddp_kwargs,
    ]
else:
    kwargs_handlers = None
accelerator = None

callback = Callbacks()


class Workspace:
    def __init__(self, cfg: OmegaConf):
        self._cfg = cfg
        self.work_dir = Path.cwd()
        self.model_path = str(cfg.model_path)
        self.pretrained_model_weights = cfg.get("pretrained_model_weights")
        logger.info(f"Working in {self.work_dir}")
        self._epoch = 0
        self.device = accelerator.device
        if (
            self._cfg.resume_training
            and (self.work_dir / self.model_path / "checkpoint.pt").exists()
        ):
            self.resume_training()
        else:
            self.init_everything()
            if (
                self.pretrained_model_weights
                and (self.work_dir / self.pretrained_model_weights).exists()
            ):
                logger.info("Loading pretrained weights for finetuning")
                self.load_model_and_loss_fn(
                    torch.load(
                        self.work_dir / self.pretrained_model_weights,
                        map_location=self.device,
                    )
                )
                # Check if we need to freeze part of the network.
                if freeze := cfg.get("freeze_submodule"):
                    if freeze.get("encoder"):
                        logger.info(f"Freezing the encoder")
                        for param in self.model.parameters():
                            param.requires_grad = False
                    if freeze.get("loss_fn"):
                        logger.info(f"Freezing the loss fn")
                        for param in self.loss_fn.parameters():
                            param.requires_grad = False

        self.convert_to_accelerator()
        accelerator.wait_for_everyone()
        callback.set_workspace(self, accelerator)

    def init_everything(self):
        set_seed_everywhere(self._cfg.seed)
        new_config = OmegaConf.to_container(self._cfg, resolve=True)
        new_config.pop("wandb")  # Was potentially edited by loader.
        accelerator.init_trackers(
            project_name=self._cfg.wandb.project,
            config=new_config,
            init_kwargs={
                "wandb": {
                    "id": self._cfg.wandb.id,
                    "resume": ("allow" if self._cfg.wandb.id else None),
                    "entity": self._cfg.wandb.entity,
                    "name": self._cfg.wandb.get("name"),
                }
            },
        )
        if accelerator.is_local_main_process:
            self._wandb_run = accelerator.get_tracker("wandb", unwrap=True)
            self._cfg.wandb.id = self._wandb_run.id

        # Set up dataloaders.
        self._train_dataloaders, self._test_dataloaders = self._setup_dataloaders()
        # Set up model.
        self.model = hydra.utils.instantiate(self._cfg.model)
        self.model = self.model.to(self.device)
        # Set up loss. Sometimes, model is optionally passed to loss.
        self.loss_fn = hydra.utils.instantiate(self._cfg.loss_fn, model=self.model)
        self.loss_fn = self.loss_fn.to(self.device)

        # First, figure out the number of processes, and scale batch size accordingly.
        # We scale by the batch size and downscale by 256 to make sure the learning rate
        # is constant across different machines.
        self._scaled_lr = (
            self._cfg.optimizer.lr
            * accelerator.num_processes
            * self._cfg.batch_size
            * self._cfg.get("gradient_accumulation_steps", 1)
        ) / 256
        # Set up optimizer and scheduler.
        self._setup_opt_and_sched()

    def _setup_opt_and_sched(self):
        # Set up optimizer.
        self._params = chain(self.model.parameters(), self.loss_fn.parameters())
        self.optimizer = hydra.utils.instantiate(
            self._cfg.optimizer,
            params=self._params,
            lr=self._scaled_lr,
        )
        # Set up scheduler.
        self.scheduler = hydra.utils.instantiate(
            self._cfg.scheduler, optimizer=self.optimizer
        )
        self._norm_clip = self._cfg.get("clip_gradient_norm", None)

    def convert_to_accelerator(self):
        (
            self.model,
            self.loss_fn,
            self._train_dataloaders,
            self._test_dataloaders,
            self.optimizer,
            self.scheduler,
        ) = accelerator.prepare(
            self.model,
            self.loss_fn,
            self._train_dataloaders,
            self._test_dataloaders,
            self.optimizer,
            self.scheduler,
        )

    def _setup_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader]:
        logger.info("Setting up dataloaders.")
        train_dataset = hydra.utils.instantiate(self._cfg.dataset.train)
        train_sampler = hydra.utils.instantiate(
            self._cfg.sampler, dataset=train_dataset
        )
        train_batch_sampler = hydra.utils.instantiate(
            self._cfg.batch_sampler, dataset=train_dataset
        )
        train_dataloader = hydra.utils.instantiate(
            self._cfg.dataloader,
            dataset=train_dataset,
            sampler=train_sampler,
            batch_sampler=train_batch_sampler,
        )
        test_dataset = hydra.utils.instantiate(self._cfg.dataset.test)
        test_sampler = hydra.utils.instantiate(self._cfg.sampler, dataset=test_dataset)
        test_batch_sampler = hydra.utils.instantiate(
            self._cfg.batch_sampler, dataset=test_dataset
        )
        test_dataloader = hydra.utils.instantiate(
            self._cfg.dataloader,
            dataset=test_dataset,
            sampler=test_sampler,
            batch_sampler=test_batch_sampler,
        )
        return train_dataloader, test_dataloader

    def run(self):
        for epoch in (
            iterator := tqdm.trange(
                self._epoch,
                self._cfg.num_epochs,
                disable=not accelerator.is_local_main_process,
            )
        ):
            iterator.set_description("Epoch {}".format(epoch + 1))
            self._epoch = epoch
            self._train_epoch()
            if epoch % self._cfg.eval_every == 0:
                self._eval_epoch()
            if epoch % self._cfg.save_every == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:
                    self.save_model()
        accelerator.end_training()

    def to_device(self, manyTensors: Iterable[torch.Tensor]):
        manyTensors = [
            tensor.to(self.device, non_blocking=True) for tensor in manyTensors
        ]
        manyTensors[0] = manyTensors[0].float() / 255.0
        return manyTensors

    def _train_epoch(self):
        self.model.train()
        self.loss_fn.train()

        callback.begin_epoch()

        iterator = tqdm.tqdm(
            self._train_dataloaders, disable=not accelerator.is_local_main_process
        )
        self._total_loss = 0
        overall_loss_dict = AveragingDict()
        for batch in iterator:
            with accelerator.accumulate(self.model, self.loss_fn):
                self._train_step(batch, overall_loss_dict, iterator)

        if not accelerator.optimizer_step_was_skipped:
            self.scheduler.step(self._epoch)
        self._total_loss /= len(self._train_dataloaders)
        overall_loss_dict.update({"total_loss": self._total_loss})
        if accelerator.is_local_main_process:
            logger.info(f"Train loss: {str(overall_loss_dict)}")
        epoch_summary = overall_loss_dict.full_summary
        epoch_summary.update(
            {"_epoch": self._epoch, "lr": self.scheduler.get_last_lr()[0]}
        )
        accelerator.log(epoch_summary)

    def _train_step(self, batch, overall_loss_dict, iterator):
        callback.begin_batch()
        self.optimizer.zero_grad(set_to_none=True)
        batch_in_device = self.to_device(batch)
        model_outputs = self.model(batch_in_device)
        loss, loss_dict = self.loss_fn(batch_in_device, model_outputs)
        accelerator.backward(loss)
        if self._norm_clip and accelerator.sync_gradients:
            accelerator.clip_grad_norm_(self._params, self._norm_clip)
        self.optimizer.step()
        self._total_loss += loss.cpu().item()
        overall_loss_dict.update(loss_dict)
        iterator.set_postfix(overall_loss_dict.summary)
        accelerator.log(overall_loss_dict.full_summary)

    @torch.no_grad()
    def _eval_epoch(self):
        self.model.eval()
        self.loss_fn.eval()

        callback.begin_epoch()

        iterator = tqdm.tqdm(
            self._test_dataloaders, disable=not accelerator.is_local_main_process
        )
        self._total_loss = 0
        overall_loss_dict = AveragingDict("test")
        for batch in iterator:
            self._eval_step(batch, overall_loss_dict, iterator)
        self._total_loss /= len(self._test_dataloaders)
        overall_loss_dict.update({"total_loss": self._total_loss})
        if accelerator.is_local_main_process:
            logger.info(f"Test loss: {str(overall_loss_dict)}")
        epoch_summary = overall_loss_dict.full_summary
        epoch_summary.update({"_epoch": self._epoch})
        accelerator.log(epoch_summary)

    def _eval_step(self, batch, overall_loss_dict, iterator):
        callback.begin_batch()
        batch_in_device = self.to_device(batch)
        model_outputs = self.model(batch_in_device)
        loss, loss_dict = self.loss_fn(batch_in_device, model_outputs)
        self._total_loss += loss.cpu().item()
        overall_loss_dict.update(loss_dict)
        iterator.set_postfix(overall_loss_dict.summary)
        accelerator.log(overall_loss_dict.full_summary)

    def save_model(self):
        """
        Save the model, optimizer, and scheduler.
        At the same time, save information about the training progress.
        At this point, assume everything has been wrapped in Accelerator.
        """
        # Create the path if it doesn't exist already.
        (self.work_dir / self.model_path).mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(self.model)
        unwrapped_optimizer = accelerator.unwrap_model(self.optimizer)
        unwrapped_scheduler = accelerator.unwrap_model(self.scheduler)
        unwrapped_loss_fn = accelerator.unwrap_model(self.loss_fn)
        accelerator.save(
            {
                "cfg": OmegaConf.to_container(self._cfg, resolve=True),
                "epoch": self._epoch,
                "model": unwrapped_model.state_dict(),
                "optimizer": unwrapped_optimizer.state_dict(),
                "scheduler": unwrapped_scheduler.state_dict(),
                "loss_fn": unwrapped_loss_fn.state_dict(),
            },
            self.work_dir / self.model_path / "checkpoint.pt",
        )
        shutil.copy(
            self.work_dir / self.model_path / "checkpoint.pt",
            self.work_dir / self.model_path / f"checkpoint_{self._epoch}.pt",
        )

    def load_pretrained_encoder_model(self, pretrained_model_path):
        self.model.load_state_dict(
            torch.load(pretrained_model_path, map_location=self.device)["model"]
        )

    def load_model_and_loss_fn(self, checkpoint):
        self.model.load_state_dict(checkpoint["model"])
        if "loss_fn" in checkpoint:
            self.loss_fn.load_state_dict(checkpoint["loss_fn"])
        else:
            logger.warning("No loss_fn found in checkpoint. Using default loss_fn.")

    def resume_training(self):
        """
        Load the model, optimizer, and scheduler.
        """
        checkpoint = torch.load(
            self.work_dir / self.model_path / "checkpoint.pt", map_location=self.device
        )
        self._cfg = OmegaConf.create(checkpoint["cfg"])
        self._epoch = checkpoint["epoch"] + 1  # We want to start from the next epoch.
        # Now initialize every object of relevance.
        self.init_everything()
        # And finally load in the weight.
        self.load_model_and_loss_fn(checkpoint)
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            logger.warning("No optimizer found in checkpoint. Using default optimizer.")
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        else:
            logger.warning("No scheduler found in checkpoint. Using default scheduler.")
        logger.info(f"Loaded model from {self.work_dir / self.model_path}")
        logger.info(f"Resuming training from epoch {self._epoch}")
        # Delete the checkpoint we just loaded to free up memory.
        del checkpoint


class SSLWorkspace(Workspace):
    def _setup_opt_and_sched(self):
        # Set up optimizer.
        self.optimizer = hydra.utils.instantiate(
            self._cfg.optimizer,
            params=self.loss_fn.parameters(),
            lr=self._scaled_lr,
        )
        # Set up scheduler.
        self.scheduler = hydra.utils.instantiate(
            self._cfg.scheduler, optimizer=self.optimizer
        )

    def _train_step(self, batch, overall_loss_dict, iterator):
        callback.begin_batch()
        self.optimizer.zero_grad(set_to_none=True)
        batch_in_device = self.to_device(batch)
        loss, loss_dict = self.loss_fn(batch_in_device, None)
        accelerator.backward(loss)
        self.optimizer.step()
        self._total_loss += loss.cpu().item()
        overall_loss_dict.update(loss_dict)
        iterator.set_postfix(overall_loss_dict.summary)
        accelerator.log(overall_loss_dict.full_summary)

    def _eval_step(self, batch, overall_loss_dict, iterator):
        callback.begin_batch()
        batch_in_device = self.to_device(batch)
        loss, loss_dict = self.loss_fn(batch_in_device, None)
        self._total_loss += loss.cpu().item()
        overall_loss_dict.update(loss_dict)
        iterator.set_postfix(overall_loss_dict.summary)
        accelerator.log(overall_loss_dict.full_summary)


class VINNWorkspace(Workspace):
    def _train_step(self, batch, overall_loss_dict, iterator):
        callback.begin_batch()
        batch_in_device = self.to_device(batch)
        model_outputs = self.model(batch_in_device)
        loss, loss_dict = self.loss_fn(batch_in_device, model_outputs)
        self._total_loss += loss.cpu().item()
        overall_loss_dict.update(loss_dict)
        iterator.set_postfix(overall_loss_dict.summary)
        accelerator.log(overall_loss_dict.full_summary)


@hydra.main(config_path="configs", config_name="finetune.yaml", version_base="1.2")
def main(cfg: OmegaConf):
    global accelerator
    accelerator = Accelerator(
        log_with="all",
        project_dir="logs/accelerate",
        kwargs_handlers=kwargs_handlers,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
    )
    loss_fn_str = str.lower(cfg.loss_fn["_target_"])
    if any((x in loss_fn_str) for x in ["byol", "moco", "mae"]):
        workspace = SSLWorkspace(cfg)
    elif "vinn" in loss_fn_str:
        workspace = VINNWorkspace(cfg)
    else:
        workspace = Workspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
