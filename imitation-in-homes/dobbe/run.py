from typing import Tuple

import cv2
import hydra
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

import wandb
from utils.trajectory_vis import visualize_trajectory


class WrapperPolicy(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def step(self, data):
        model_out = self.model(data)
        return self.loss_fn.step(data, model_out)

    def reset(self):
        pass


def _setup_dataloaders(cfg) -> Tuple[DataLoader]:
    train_dataset = hydra.utils.instantiate(cfg.dataset.train)
    train_sampler = hydra.utils.instantiate(cfg.sampler, dataset=train_dataset)
    train_batch_sampler = hydra.utils.instantiate(
        cfg.batch_sampler, dataset=train_dataset
    )
    train_dataloader = hydra.utils.instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        sampler=train_sampler,
        batch_sampler=train_batch_sampler,
    )
    return train_dataloader


def _init_run(cfg: OmegaConf):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.project,
        mode="disabled",
        entity=cfg.wandb.entity,
        config=dict_cfg,
    )
    return dict_cfg


def _init_vinn(cfg):
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model.encoder.load_state_dict(checkpoint["model"])
    model = model.to(cfg.device)
    train_dataloader = _setup_dataloaders(cfg)
    model.set_dataset(train_dataloader)
    return model


def _init_model(cfg):
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(cfg.device)
    checkpoint = torch.load(cfg.model_weight_pth, map_location=cfg.device)
    model.load_state_dict(checkpoint["model"])
    return model


def _init_model_loss(cfg):
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(cfg.device)
    checkpoint = torch.load(cfg.model_weight_pth, map_location=cfg.device)
    model.load_state_dict(checkpoint["model"])
    loss_fn = hydra.utils.instantiate(cfg.loss_fn, model=model)
    loss_fn.load_state_dict(checkpoint["loss_fn"])
    loss_fn = loss_fn.to(cfg.device)
    policy = WrapperPolicy(model, loss_fn)
    return policy


def run(cfg: OmegaConf, init_model=_init_model):
    model = init_model(cfg)
    if cfg.get("time_model"):
        # Dry run the model. Useful for running on the hardware.
        import time

        start_time = time.time()
        num_iters = 100
        with torch.no_grad():
            for _ in range(num_iters):
                image = torch.rand((1, 1, 3, 256, 256), device=cfg.device)
                depth = torch.rand((1, 1, 1, 192, 256), device=cfg.device)
                act = torch.rand((1, 1, 7), device=cfg.device)
                model.step((image, depth, act))
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iters
        print("Average time taken: ", avg_time, f" @ {1/avg_time} Hz")
        exit()

    if cfg["run_offline"] is True:
        test_dataset = hydra.utils.instantiate(cfg.dataset.test)
        visualize_trajectory(
            model, test_dataset, cfg["device"], cfg["image_buffer_size"]
        )

    else:
        # Lazy loading so we can run offline eval without the robot set up.
        from robot.controller import Controller

        dict_cfg = _init_run(cfg)
        controller = Controller(cfg=dict_cfg)
        controller.setup_model(model)
        controller.run()


@hydra.main(config_path="configs", config_name="run", version_base="1.2")
def main(cfg: OmegaConf):
    if "vinn" in str.lower(cfg.model["_target_"]):
        run(cfg, init_model=_init_vinn)
    elif "loss_fn" in cfg:
        run(cfg, init_model=_init_model_loss)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
