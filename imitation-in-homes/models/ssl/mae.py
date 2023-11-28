import math
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
)
from vit_pytorch.vit import Transformer

from utils.ssl_transfroms import IMAGENET_MEAN, IMAGENET_STD


def mask_image_using_indices(image, indices, num_patches=64):
    # function that takes 1, 3, 256, 256 image and return image of same size but black out the patches indicated by the indices
    image = image.clone()
    grid_size = math.sqrt(num_patches)
    assert grid_size == int(grid_size)
    grid_size = int(grid_size)
    patch_size = int(image.shape[-1] // grid_size)

    black_out_value = -2
    # check if image is normalized or not
    if image.max() > 1:
        black_out_value = 0

    for i in range(grid_size):
        for j in range(grid_size):
            if i * grid_size + j in indices:
                image[
                    :,
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ] = black_out_value
    return image


def fill_image_using_predicted_patches(
    masked_image, pred_patches, indices, num_patches=64
):
    # pred_patches is of the shape len(indices),
    grid_size = math.sqrt(num_patches)
    patch_size = int(masked_image.shape[-1] // grid_size)
    masked_image = masked_image.clone()
    patch_to_img = Rearrange("b 1 (p1 p2 c) -> b c p1 p2", p1=patch_size, p2=patch_size)

    grid_size = math.sqrt(num_patches)
    assert grid_size == int(grid_size)
    grid_size = int(grid_size)
    patch_size = masked_image.shape[2] // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            if i * grid_size + j in indices:
                idx = torch.where(indices == i * grid_size + j)[1][0].item()

                masked_image[
                    :,
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ] = patch_to_img(pred_patches[:, idx : idx + 1])
    return masked_image


class MAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder_dim=512,
        masking_ratio=0.75,
        decoder_depth=6,
        decoder_heads=8,
        decoder_dim_head=64,
        img_mean: Optional[torch.Tensor] = IMAGENET_MEAN,
        img_std: Optional[torch.Tensor] = IMAGENET_STD,
    ):
        super().__init__()
        self.encoder = encoder
        self.masking_ratio = masking_ratio
        self.patch_size = self.encoder.model.patch_embed.patch_size
        self.to_patch = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
        )
        self.num_patches, encoder_dim = self.encoder.model.pos_embed.shape[-2:]
        pixel_values_per_patch = self.encoder.model.patch_embed.proj.weight.shape[0]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4,
        )
        self.decoder_pos_emb = nn.Embedding(self.num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

        self.transfroms = Compose(
            [
                RandomResizedCrop(224, scale=(0.2, 1.0), antialias=True),
                RandomHorizontalFlip(),
                Normalize(mean=img_mean, std=img_std),
            ]
        )
        self.val_transforms = Compose(
            [
                Resize(224, antialias=True),
                Normalize(mean=img_mean, std=img_std),
            ]
        )

    def load_all_but_encoder(self, path):
        dict = torch.load(path, map_location="cpu")
        learner_decod_state_dict = {
            k.replace("learner.", ""): v
            for k, v in dict["loss_fn"].items()
            if ("learner" in k and "learner.encoder" not in k)
        }
        self.load_state_dict(learner_decod_state_dict, strict=False)

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def _reconstruction_params(self, image):
        training = self.training
        self.eval()
        for p in self.parameters():
            device = p.device
            break
        image = image.to(device)
        temp = torch.zeros(1, 1, 7)
        with torch.no_grad():
            pred_pixel_values, masked_patches, masked_indices = self.forward(
                (image, temp), reconstruct=True
            )
        if training:
            self.train()
        image = self.val_transforms(image.squeeze(0)).unsqueeze(0)
        return (image, pred_pixel_values, masked_patches, masked_indices)

    def get_recosntructed_visuals(self, image):
        (
            image,
            pred_pixel_values,
            masked_patches,
            masked_indices,
        ) = self._reconstruction_params(image)
        masked_image = mask_image_using_indices(
            image.squeeze(0), masked_indices, self.num_patches - 1
        )

        reconstructed_image = fill_image_using_predicted_patches(
            masked_image, pred_pixel_values, masked_indices, self.num_patches - 1
        )
        return image, masked_image, reconstructed_image

    def forward(self, x, reconstruct=False):
        # process images
        images, *labels = x
        images = rearrange(images, "b t c h w -> (b t) c h w")
        if self.training:
            images = self.transfroms(images)
        else:
            images = self.val_transforms(images)

        device = images.device
        # get patches from images
        patches = self.to_patch(images)
        batch, num_patches, *_ = patches.shape

        tokens = self.encoder.model.patch_embed(images)

        # add positional and class embeddings
        if self.encoder.model.cls_token is not None:
            tokens = torch.cat(
                (self.encoder.model.cls_token.expand(tokens.shape[0], -1, -1), tokens),
                dim=1,
            )
        tokens = tokens + self.encoder.model.pos_embed

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer
        encoded_tokens = self.encoder.model.blocks(tokens)
        encoded_tokens = self.encoder.model.norm(encoded_tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(
            unmasked_indices
        )

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoded_tokens = torch.zeros(
            batch, num_patches, self.decoder_dim, device=device
        )
        decoded_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoded_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoded_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        if reconstruct:
            return pred_pixel_values, masked_patches, masked_indices

        # calculate reconstruction loss

        if reconstruct:
            return pred_pixel_values, masked_patches, masked_indices
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        return recon_loss
