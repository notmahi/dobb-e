from typing import Optional

import torch

from loss_fns.abstract_loss_fn import AbstractLossFn
from models.ssl.mae import MAE
from utils import log_wandb_image
from utils.ssl_transfroms import IMAGENET_MEAN, IMAGENET_STD


class MAELossFn(AbstractLossFn):
    def __init__(
        self,
        model: torch.nn.Module = None,
        decoder_dim=512,
        masking_ratio=0.75,
        decoder_depth=6,
        decoder_heads=8,
        decoder_dim_head=64,
        freeze_encoder=True,
        unfreeze_encoder_after=10,
        visualize_n=10,
        visualize_every_train_n=1,
        visualize_every_val_n=1,
        img_mean: Optional[torch.Tensor] = IMAGENET_MEAN,
        img_std: Optional[torch.Tensor] = IMAGENET_STD,
        load_decoder_from: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(model, *args, **kwargs)
        self.encoder = model

        self.learner = MAE(
            self.encoder,
            decoder_dim,
            masking_ratio,
            decoder_depth,
            decoder_heads,
            decoder_dim_head,
            img_mean,
            img_std,
        )

        if load_decoder_from is not None:
            self.learner.load_all_but_encoder(load_decoder_from)

        self.unfreeze_encoder_after = unfreeze_encoder_after

        self.img_mean = img_mean
        self.img_std = img_std

        if freeze_encoder:
            self.learner._freeze_encoder()

        # the following parameters are used for visualization logging

        self.visualize_n = visualize_n
        self.train_imgs = []
        self.val_imgs = []
        self.visualize_every_train_n = visualize_every_train_n
        self.visualize_every_val_n = visualize_every_val_n

    def forward(self, data, output, *args, **kwargs):
        self._sample_imgs_for_vis(data)
        loss = self.learner(data)
        return loss, {"loss": loss}

    def _begin_epoch(
        self, epoch, train_dataloader, test_dataloader, is_main_process, **kwargs
    ):
        # self._sample_imgs_for_vis(train_dataloader, test_dataloader) #TODO dataloader gets stuck because of this
        if epoch > self.unfreeze_encoder_after:
            self.learner._unfreeze_encoder()

        if (
            is_main_process
            and self.learner.training
            and epoch % self.visualize_every_train_n == 0
        ):
            return self._log_visualizations()

        if (
            is_main_process
            and not self.learner.training
            and epoch % self.visualize_every_val_n == 0
        ):
            return self._log_visualizations()
        return None

    def _sample_imgs_for_vis(self, data):
        data, _ = data
        if self.learner.training and self.train_imgs == []:
            # randomly sample n indices from the dataset
            n_indices = torch.randint(0, len(data), (self.visualize_n,))
            self.train_imgs = [data[idx] for idx in n_indices]

        if not self.learner.training and self.val_imgs == []:
            # randomly sample n indices from the dataset
            n_indices = torch.randint(0, len(data), (self.visualize_n,))
            self.val_imgs = [data[idx] for idx in n_indices]

    # reconstructs image using the trained learner and logs them in wandb for visualization
    def _log_visualizations(self):
        def format_img(img):
            while img.shape[0] == 1:
                img = img.squeeze(0)

            img = img.permute(1, 2, 0).cpu()
            if img.mean() > 1:
                img * self.img_std + self.img_mean
            return img

        if self.learner.training:
            # log the images
            logs = {}
            for idx, image in enumerate(self.train_imgs):
                image = image.unsqueeze(0).detach()

                (
                    image,
                    masked_image,
                    reconstructed_image,
                ) = self.learner.get_recosntructed_visuals(image)

                logs.update(
                    log_wandb_image(
                        format_img(image), f"train_vis/original_frame_{idx}"
                    )
                )
                logs.update(
                    log_wandb_image(
                        format_img(masked_image), f"train_vis/masked_frame_{idx}"
                    )
                )
                logs.update(
                    log_wandb_image(
                        format_img(reconstructed_image),
                        f"train_vis/filled_pred_frame_{idx}",
                    )
                )

            return logs

        else:
            # log the images
            logs = {}
            for idx, image in enumerate(self.val_imgs):
                image = image.unsqueeze(0).detach()

                (
                    image,
                    masked_image,
                    reconstructed_image,
                ) = self.learner.get_recosntructed_visuals(image)

                logs.update(
                    log_wandb_image(format_img(image), f"test_vis/original_frame_{idx}")
                )
                logs.update(
                    log_wandb_image(
                        format_img(masked_image), f"test_vis/masked_frame_{idx}"
                    )
                )
                logs.update(
                    log_wandb_image(
                        format_img(reconstructed_image),
                        f"test_vis/filled_pred_frame_{idx}",
                    )
                )

            return logs
