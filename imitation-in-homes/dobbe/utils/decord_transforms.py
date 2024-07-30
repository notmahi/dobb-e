import logging
from typing import Iterable, Optional, Tuple, Union

import torch
from timm.data.random_erasing import RandomErasing
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DEFAULT_CROP_PCT = 0.875
DEFAULT_CROP_MODE = "center"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
IMAGENET_DPN_MEAN = (124 / 255, 117 / 255, 104 / 255)
IMAGENET_DPN_STD = tuple([1 / (0.0167 * 255)] * 3)
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def create_transform(
    input_size,
    is_training: bool = False,
    scale: Optional[Tuple[float, float]] = None,
    ratio: Optional[Tuple[float, float]] = None,
    hflip: float = 0.5,
    vflip: float = 0.0,
    grayscale: float = 0.0,
    gaussblr: float = 0.0,
    gaussblr_kernel: int = 3,
    gaussblr_sigma: tuple = (1.0, 2.0),
    color_jitter: Optional[Union[Iterable[float], float]] = 0.4,
    interpolation: Union[str, InterpolationMode] = "bilinear",
    mean: Union[Iterable, torch.Tensor] = IMAGENET_DEFAULT_MEAN,
    std: Union[Iterable, torch.Tensor] = IMAGENET_DEFAULT_STD,
    re_prob=0.0,
    re_mode="const",
    re_count=1,
    re_num_splits=0,
    crop_pct=None,
    crop_mode=None,
    *args,
    **kwargs,
):
    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size
    transform_list = []

    interpolation_mode = InterpolationMode.BILINEAR
    try:
        interpolation_mode = InterpolationMode[interpolation.upper()]
    except AttributeError:
        logging.warning(
            f"Interpolation mode {interpolation} is not recognized, "
            f"using bilinear instead."
        )
    if is_training:
        scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
        ratio = tuple(ratio or (3.0 / 4.0, 4.0 / 3.0))  # default imagenet ratio range

        transform_list.append(
            transforms.RandomResizedCrop(
                img_size,
                scale,
                ratio,
                antialias=False,
                interpolation=interpolation_mode,
            )
        )
        if hflip > 0.0:
            transform_list.append(transforms.RandomHorizontalFlip(hflip))
        if vflip > 0.0:
            transform_list.append(transforms.RandomVerticalFlip(vflip))

        if color_jitter is not None:
            if isinstance(color_jitter, (list, tuple)):
                assert len(color_jitter) in (
                    3,
                    4,
                ), "expected either 3 or 4 values for color jitter"
            else:
                color_jitter = (float(color_jitter),) * 3
            transform_list.append(transforms.ColorJitter(*color_jitter))

        if grayscale > 0.0:
            transform_list.append(transforms.RandomGrayscale(p=grayscale))
        if gaussblr > 0.0:
            transform_list.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            (gaussblr_kernel, gaussblr_kernel), gaussblr_sigma
                        )
                    ],
                    p=gaussblr,
                )
            )
        transform_list.append(transforms.Normalize(mean, std))

        if re_prob > 0.0:
            transform_list.append(
                RandomErasing(
                    re_prob,
                    mode=re_mode,
                    max_count=re_count,
                    num_splits=re_num_splits,
                    device="cuda",
                )
            )

    else:
        if crop_pct is None:
            crop_pct = DEFAULT_CROP_PCT
        if crop_mode is None:
            crop_mode = DEFAULT_CROP_MODE
        rescaled_size = (
            int(img_size / crop_pct)
            if isinstance(img_size, (int, float))
            else ([int(size / crop_pct) for size in img_size])
        )
        transform_list.append(
            transforms.Resize(rescaled_size, interpolation_mode, antialias=False)
        )
        if crop_mode == "center":
            transform_list.append(transforms.CenterCrop(img_size))
        elif crop_mode == "random":
            transform_list.append(transforms.RandomCrop(img_size))
        else:
            raise ValueError(f"crop_mode '{crop_mode}' not recognized")
        transform_list.append(transforms.Normalize(mean, std))

    return transforms.Compose(transform_list)
