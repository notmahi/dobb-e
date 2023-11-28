import logging
import os
import sys
import urllib
from abc import abstractmethod
from pathlib import Path

import accelerate
import einops
import six
import timm
import torch

from models.encoders import abstract_base_encoder
from utils.decord_transforms import create_transform

_DOWNLOAD_CACHE = "/tmp/download-cache"

_MODELS = {
    "mvp-vits-mae-hoi": "https://berkeley.box.com/shared/static/m93ynem558jo8vltlads5rcmnahgsyzr.pth",
    "mvp-vits-mae-in": "https://berkeley.box.com/shared/static/qlsjkv03nngu37eyvtjikfe7rz14k66d.pth",
    "mvp-vits-sup-in": "https://berkeley.box.com/shared/static/95a4ncqrh1o7llne2b1gpsfip4dt65m4.pth",
    "mvp-vitb-mae-egosoup": "https://berkeley.box.com/shared/static/0ckepd2ja3pi570z89ogd899cn387yut.pth",
    # "mvp-vitl-256-mae-egosoup": "https://berkeley.box.com/shared/static/6p0pc47mlpp4hhwlin2hf035lxlgddxr.pth",
    "vc1-vitb": "https://dl.fbaipublicfiles.com/eai-vc/vc1_vitb.pth",
    "vc1-vitl": "https://dl.fbaipublicfiles.com/eai-vc/vc1_vitl.pth",
    "r3m-resnet50": "https://docs.google.com/uc?export=download&confirm=t&id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA",
    "r3m-resnet34": "https://docs.google.com/uc?export=download&confirm=t&id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE",
    "r3n-resnet18": "https://docs.google.com/uc?export=download&confirm=t&id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-",
}


BASE_TIMM_VIT_MODEL = {
    "vits": "vit_small_patch16_224",
    "vitb": "vit_base_patch16_224",
    "vitl": "vit_large_patch16_224",
}

accelerator = accelerate.Accelerator()


def _progress_bar(count, total):
    """Report download progress.
    Credit:
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)

    sys.stdout.write(
        "  [{}] {}% of {:.1f}MB file  \r".format(bar, percents, total / 1024 / 1024)
    )
    sys.stdout.flush()
    if count >= total:
        sys.stdout.write("\n")


def download_model_if_needed(ckpt_file, model_name) -> int:
    model_base_dir = Path(_DOWNLOAD_CACHE)
    ckpt_file = os.path.join(model_base_dir, ckpt_file)
    bytes_downloaded = 0
    if not os.path.isfile(ckpt_file):
        os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)
        model_url = _MODELS[model_name]
        bytes_downloaded = _download_url(model_url, ckpt_file)
    return ckpt_file, bytes_downloaded


def _download_url(url, dst_file_path, chunk_size=8192, progress_hook=_progress_bar):
    """Download url and write it to dst_file_path.
    Credit:
    https://stackoverflow.com/questions/2028517/python-urllib2-progress-hook
    """
    try:
        response = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        logging.error(f"Error downloading model from {url}:\n{e}")
        raise e
    if six.PY2:
        total_size = response.info().getheader("Content-Length").strip()
    else:
        total_size = response.info().get("Content-Length").strip()
    total_size = int(total_size)
    bytes_so_far = 0

    with open(dst_file_path, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            bytes_so_far += len(chunk)
            if not chunk:
                break
            if progress_hook:
                progress_hook(bytes_so_far, total_size)
            f.write(chunk)
    return bytes_so_far


class BaselineEncoder(abstract_base_encoder.AbstractEncoder):
    def __init__(
        self,
        model_name: str,
        override_aug_kwargs=dict(
            hflip=0.0, vflip=0.0, scale=(1.0, 1.0), crop_pct=0.875
        ),
        **kwargs,
    ):
        super().__init__()
        ckpt_file = Path(f"{_DOWNLOAD_CACHE}/checkpoints") / f"{model_name}.pth"
        if accelerator.is_local_main_process:
            download_model_if_needed(ckpt_file, model_name)
        accelerator.wait_for_everyone()
        self.model = timm.create_model(
            self._process_model_name(model_name),
            pretrained=False,
            num_classes=0,
            **kwargs,
        )
        state_dict = torch.load(ckpt_file, map_location="cpu")
        processed_state_dict = self._process_state_dict(state_dict)
        self.model.load_state_dict(processed_state_dict, strict=True)
        del processed_state_dict, state_dict
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        # Now define the transforms.
        data_cfg.update(override_aug_kwargs)
        data_cfg["is_training"] = True
        self._train_transform = create_transform(**data_cfg)
        data_cfg["is_training"] = False
        self._test_transform = create_transform(**data_cfg)

    @abstractmethod
    def _process_state_dict(self, state_dict):
        raise NotImplementedError

    @abstractmethod
    def _process_model_name(self, model_name):
        raise NotImplementedError

    def feature_dim(self):
        return self.model.num_features

    def transform(self, x):
        return (
            self._train_transform(x) if self.model.training else self._test_transform(x)
        )

    def forward(self, x):
        # Split the input into frames and labels.
        images, *labels = x
        # Flatten the frames into a single batch.
        flattened_images = einops.rearrange(images, "b t c h w -> (b t) c h w")
        # Transform and pass through the model.
        result = self.model(self.transform(flattened_images))
        # Unflatten the result.
        result = einops.rearrange(result, "(b t) c -> b t c", b=images.shape[0])
        return result
