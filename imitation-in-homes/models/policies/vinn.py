# VINN model that uses encoder and nearest neighbor to predict action
import pickle as pkl

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

import wandb
from utils.image_plots import display_image_in_grid, overlay_action


class VINN(nn.Module):
    def __init__(
        self,
        encoder,
        k=5,
        enc_weight_pth=None,
        plot_neighbors=False,
        stochastic_nbhr_select=False,
        softmax_temperature=1,
        action_scale=1,
        cfg=None,
    ):
        super(VINN, self).__init__()
        self.encoder = encoder
        self.cfg = cfg

        self.k = k
        self.plot_neighbors = plot_neighbors
        self.stochastic_nbhr_select = stochastic_nbhr_select
        self.softmax_temperature = softmax_temperature
        self.action_scale = action_scale

        self.set_act_metrics(cfg)

        if enc_weight_pth is not None:
            self.encoder.load_state_dict(torch.load(enc_weight_pth)["model"])

        self.representations = None
        self.actions = None
        self.img_pths = None
        self.imgs = None
        softmax = nn.Softmax(dim=1)
        self.dist_scale_func = lambda x: (softmax(-x))
        self.encoder.eval()
        self.device = "cpu"
        self.encoder.to(self.device)
        self.img_transform = T.Resize((256, 256), antialias=True)

    def set_act_metrics(self, cfg):
        if cfg is not None and "act_metrics" in cfg["dataset"]:
            act_metrics = cfg["dataset"]["act_metrics"]
            # register as buffers so they are saved with the model
            self.register_buffer("act_mean", act_metrics["mean"].float())
            self.register_buffer("act_std", act_metrics["std"].float())
        else:
            # initialize to 0 and 1
            # register as buffers so they are saved with the model
            self.register_buffer("act_mean", torch.zeros(1).float())
            self.register_buffer("act_std", torch.ones(1).float())

        self.act_mean.requires_grad = False
        self.act_std.requires_grad = False

    def denorm_action(self, action):
        return action * self.act_std + self.act_mean

    def to(self, device):
        self.device = device
        self.encoder.to(device)

        return super().to(device)

    def set_dataset(self, dataloader):
        self.train_dataset = dataloader.dataset
        for i, (image, label) in tqdm(enumerate(dataloader)):
            image = image.float() / 255.0
            image = image.to(self.device)
            label = torch.Tensor(label).to("cpu").detach().squeeze()
            x = (image, label)
            representation = self.encoder(x).to("cpu").detach().squeeze(dim=1)
            if self.representations is None:
                self.representations = representation
                self.actions = label
                image = image.to("cpu").detach().numpy()
                self.imgs = list(image)
            else:
                self.representations = torch.cat(
                    (self.representations, representation), 0
                )
                self.actions = torch.cat((self.actions, label), 0)
                image = image.to("cpu").detach().numpy()
                self.imgs.extend(list(image))

    def step(self, img, **kwargs):
        normalized_image = self.img_transform(img[0].squeeze(0))
        logs = {}
        if self.plot_neighbors:
            action_tensor, indices = self.get_action(
                normalized_image.unsqueeze(0).unsqueeze(0),
                return_indices=True,
                **kwargs
            )
            action_tensor[:-1] *= self.action_scale

            indices = indices.squeeze().cpu().numpy()
            logs["indices"] = indices
            self.indices = indices
            for nbhr, idx in enumerate(indices):
                img = self.train_dataset[idx]
                img = (img[0][0] * 255).permute(1, 2, 0).cpu().numpy()
                img = wandb.Image(img)
                logs["nbhr_{}".format(nbhr)] = img

            return action_tensor, logs
        else:
            action_tensor = self.get_action(
                normalized_image.unsqueeze(0).unsqueeze(0),
                return_indices=False,
                **kwargs
            )
            action_tensor[:-1] *= self.action_scale

            return action_tensor, logs

    def get_action(self, img, return_indices=False, **kwargs):
        self.encoder.eval()
        with torch.no_grad():
            if return_indices:
                act, indices = self(img, return_indices=True)
            else:
                act = self(img).squeeze().detach()
            act = self.denorm_action(act)

        return act if not return_indices else (act, indices)

    def __call__(self, batch_images, k=None, return_indices=False):
        if self.training and self.epoch == 0:
            self.eval()
            (images, labels) = batch_images
            images = images.to(self.device)
            labels = torch.Tensor(labels).to("cpu").detach().squeeze()
            representation = (
                self.encoder(batch_images).to("cpu").detach().squeeze(dim=1)
            )
            if self.representations is None:
                self.representations = representation
                self.actions = labels
                images = images.to("cpu").detach().numpy()
                self.imgs = list(images)
            else:
                self.representations = torch.cat(
                    (self.representations, representation), 0
                )
                self.actions = torch.cat((self.actions, labels), 0)
                # self.img_pths.append(pth)
                images = images.to("cpu").detach().numpy()
                self.imgs.extend(list(images))

            self.training = True

            return None
        elif not self.training:
            if k is None:
                k = self.k

            all_distances = torch.zeros(
                (batch_images[0].shape[0], self.representations.shape[0])
            )

            batch_rep = (
                self.encoder(batch_images).squeeze(dim=1).detach().to(self.device)
            )
            dat_rep = self.representations.to(self.device)
            all_distances = torch.cdist(batch_rep, dat_rep).to("cpu")

            top_k_distances, indices = torch.topk(
                all_distances, k, dim=1, largest=False
            )
            top_k_actions = self.actions[indices].to(self.device)

            top_k_distances /= self.softmax_temperature
            weights = self.dist_scale_func(top_k_distances).to(self.device)

            if self.stochastic_nbhr_select:
                stochastic_select = torch.multinomial(weights, 1)
                selection_matrix = einops.repeat(
                    stochastic_select, "b 1 -> b 1 a", a=top_k_actions.shape[-1]
                )
                pred = top_k_actions.gather(1, selection_matrix)
                pred = einops.rearrange(pred, "b 1 a -> b a")
            else:
                pred = torch.sum(
                    top_k_actions * weights.unsqueeze(-1), dim=1
                )  # weighted average

            if return_indices:
                return pred, indices

            return pred

        else:
            return None

    def save_state_variables(self, save_path):
        save_vars = [
            self.representations,
            self.actions,
            self.act_mean,
            self.act_std,
            self.img_pths,
            self.imgs,
        ]
        save_var_names = [
            "representations",
            "actions",
            "act_mean",
            "act_std",
            "img_pths",
            "imgs",
        ]
        save_dict = {}
        for i, var in enumerate(save_vars):
            save_dict[save_var_names[i]] = var
        pkl.dump(save_dict, open(save_path, "wb"))

        # also save encder weights with same name but .pth extension
        torch.save(self.encoder.state_dict(), save_path[:-4] + ".pth")

    def load_state_variables(self, load_path):
        print("Loading state variables from {}".format(load_path))
        load_dict = pkl.load(open(load_path, "rb"))
        self.representations = load_dict["representations"]
        self.actions = load_dict["actions"]

        self.img_pths = load_dict["img_pths"]
        if "imgs" in load_dict:
            self.imgs = load_dict["imgs"]

        self.act_mean = nn.Parameter(torch.Tensor([0.0] * 7))
        self.act_std = nn.Parameter(torch.Tensor([1.0] * 7))

        self.encoder.load_state_dict(torch.load(load_path[:-4] + ".pth"))
        self.encoder.eval()

    def _begin_epoch(
        self, epoch, train_dataloader, test_dataloader, is_main_process, **kwargs
    ):
        self.epoch = epoch
        self.k = epoch + 1

        return None

    def eval_dataset(
        self,
        train_dataset,
        val_dataset,
        k=4,
        start=None,
        end=None,
        plot_freq=1,
        max_images=20,
        vector_scale=14,
    ):
        self.eval()

        img_grid = []
        label_grid = []
        loss = 0

        for i in range(len(val_dataset)):
            if i == plot_freq * max_images:
                break
            if start is not None and i < start:
                continue
            if end is not None and i > end:
                break

            frame_val, action_val = val_dataset[i]
            frame_val = frame_val.to(self.device).unsqueeze(0)

            query_img = frame_val.squeeze().permute(1, 2, 0).cpu().numpy()

            normalized_action = torch.Tensor(action_val).squeeze().to(self.device)
            action_val = self.denorm_action(normalized_action)

            query_img = overlay_action(
                action_val, query_img, color=(0, 255, 0), vector_scale=vector_scale
            )

            y_hat, indices = self((frame_val, action_val), k, return_indices=True)

            normalized_pred_action = y_hat.squeeze().detach()
            pred_action = self.denorm_action(normalized_pred_action)

            y = action_val

            y_hat = y_hat.squeeze()

            query_img = overlay_action(
                pred_action.cpu().numpy(),
                query_img,
                color=(255, 0, 0),
                vector_scale=vector_scale,
                shift_start_point=True,
            )

            if plot_freq != 0 and i % plot_freq == 0:
                img_grid.append([])
                label_grid.append([])
                img_grid[-1].append(query_img)
                label_grid[-1].append("query, loss {:.3f}".format(0.0))

                indices = indices[0]
                for j in range(k):
                    frame_train, action_train = train_dataset[indices[j]]
                    frame_train = frame_train.squeeze().cpu()

                    normalized_action = (
                        torch.Tensor(action_train).squeeze().to(self.device)
                    )
                    nbhr_action = self.denorm_action(normalized_action).cpu().numpy()

                    img = frame_train.permute(1, 2, 0).numpy()
                    img = overlay_action(
                        nbhr_action, img, color=(0, 255, 0), vector_scale=vector_scale
                    )
                    if plot_freq != 0 and i % plot_freq == 0:
                        img_grid[-1].append(img)
                        label_grid[-1].append("nbhr")

        if plot_freq != 0:
            display_image_in_grid(img_grid, label_grid)

    def eval_image(self, img, train_dataset, k=4):
        self.eval()
        self.k = k

        img_grid = []
        label_grid = []

        img = self.img_transform(img).unsqueeze(0).unsqueeze(0).to(self.device)
        action_val = torch.zeros(1, 7).to(self.device)
        y_hat, _ = self.step((img, action_val))

        query_img = img.squeeze().permute(1, 2, 0).cpu().numpy()

        normalized_action = torch.Tensor(y_hat).squeeze().to(self.device)
        action_val = self.denorm_action(normalized_action)

        query_img = overlay_action(
            action_val, query_img, color=(0, 255, 0), vector_scale=14
        )

        img_grid.append([])
        label_grid.append([])
        img_grid[-1].append(query_img)
        label_grid[-1].append("query, loss {:.3f}".format(0.0))

        for j in range(k):
            frame_train, action_train = train_dataset[self.indices[j]]
            frame_train = frame_train.squeeze().cpu()

            normalized_action = torch.Tensor(action_train).squeeze().to(self.device)
            nbhr_action = self.denorm_action(normalized_action).cpu().numpy()

            img = frame_train.permute(1, 2, 0).numpy()
            img = overlay_action(nbhr_action, img, color=(0, 255, 0), vector_scale=14)

            img_grid[-1].append(img)
            label_grid[-1].append("nbhr")

        fig, axes = plt.subplots(1, 6, figsize=(15, 3))

        for i, image_np in enumerate(img_grid[0]):
            axes[i].imshow(image_np)
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()

    def reset(self):
        pass
