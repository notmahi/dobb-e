from typing import Optional

import torch
from torchvision import transforms as T

IMAGENET_MEAN = torch.Tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.Tensor([0.229, 0.224, 0.225])


def get_byol_transforms(
    img_size=256,
    img_mean: Optional[torch.Tensor] = IMAGENET_MEAN,
    img_std: Optional[torch.Tensor] = IMAGENET_STD,
):
    return T.Compose(
        [
            # T.RandomResizedCrop(256, scale=(0.6, 1.0))
            T.RandomResizedCrop((img_size, img_size), scale=(0.6, 1.0), antialias=True),
            T.RandomApply(
                torch.nn.ModuleList([T.ColorJitter(0.8, 0.8, 0.8, 0.2)]), p=0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply(
                torch.nn.ModuleList([T.GaussianBlur((3, 3), (1.0, 2.0))]), p=0.2
            ),
            T.Normalize(mean=img_mean, std=img_std),
        ]
    )


def get_moco_transforms(
    img_size=256,
    img_mean: Optional[torch.Tensor] = IMAGENET_MEAN,
    img_std: Optional[torch.Tensor] = IMAGENET_STD,
):
    normalize = T.Normalize(mean=img_mean, std=img_std)
    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0), antialias=True),
            T.RandomApply(
                [T.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                p=0.8,  # not strengthened
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur((3, 3), (1.0, 2.0))], p=1.0),
            normalize,
        ]
    ), T.Compose(
        [
            T.RandomResizedCrop(256, scale=(0.6, 1.0), antialias=True),
            T.RandomApply(
                [T.ColorJitter(0.4, 0.4, 0.2, 0.1)],
                p=0.8,  # not strengthened
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur((3, 3), (1.0, 2.0))], p=0.1),
            T.RandomSolarize(0.5, p=0.2),
            normalize,
        ]
    )
