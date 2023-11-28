import zipfile
from pathlib import Path
from typing import Optional, Union

import liblzfse
import numpy as np
import torch

from dataloaders.utils import TrajectorySlice

DEPTH_FOLDER = "compressed_depths"
DEPTH_FILENAME = "compressed_np_depth_float32.bin"


class DepthDataLoader:
    def __init__(
        self,
        bin_root_path: Optional[Union[str, Path]] = None,
        zip_path: Optional[Union[str, Path]] = None,
        binarize: bool = False,
        log2_scale: float = 1.0,
        log2_x_shift: float = 0.0,
        log2_y_shift: float = 0.0,
        n_bins: int = 10,
    ):
        # Ensure bin_root_path XOR zip_path is provided
        assert (bin_root_path is None) != (
            zip_path is None
        ), "Must provide exactly one of bin_path or zip_path"
        if bin_root_path is not None:
            self.zip_path = None
            self.bin_path = Path(bin_root_path) / DEPTH_FILENAME
            assert self.bin_path.exists(), f"Path {self.bin_path} does not exist"
            self._depth_data = None
        else:
            self.bin_path = None
            self.zip_path = Path(zip_path)
            assert self.zip_path.exists(), f"Path {self.zip_path} does not exist"
            self.zip_ref = zipfile.ZipFile(zip_path, "r")
            self.depth_data_path = Path()
        self._binarize = binarize
        if binarize:
            self.bin_pixels = lambda x: (
                (
                    log2_scale * torch.log2(x.clamp(min=log2_x_shift) - log2_x_shift)
                    + log2_y_shift
                ).floor()
            ).clamp(0, n_bins - 1)

    def load_depth_zip(self, filepath: Union[str, Path]):
        with self.zip_ref.open(str(filepath), "r") as depth_fh:
            raw_bytes = depth_fh.read()
            decompressed_bytes = liblzfse.decompress(raw_bytes)
            depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

        # depth_img = depth_img.reshape((960, 720))  # For a FaceID camera 3D Video
        depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video
        depth_img = np.ascontiguousarray(np.rot90(depth_img, -1))

        depth_tensor = torch.from_numpy(depth_img)
        if self._binarize:
            depth_tensor = self.bin_pixels(depth_tensor)
        return depth_tensor  # 1,192,256

    def set_depth_data_path(self, path: Path):
        self.depth_data_path = path

    def get_batch_zip(self, indices: Union[np.ndarray, TrajectorySlice]):
        depth_batch = []
        for idx in indices:
            depth_tensor = self.load_depth_zip(
                self.depth_data_path / DEPTH_FOLDER / f"{str(idx).zfill(4)}.depth"
            )
            depth_batch.append(depth_tensor)

        return torch.stack(depth_batch)  # B, 1, 224, 224

    def get_batch_bin(self, indices: Union[np.ndarray, TrajectorySlice]):
        if self._depth_data is None:
            self._depth_data = np.frombuffer(
                liblzfse.decompress(self.bin_path.read_bytes()), dtype=np.float32
            )
            # No need to rotate because it was preprocessed already.
            self._depth_data = self._depth_data.reshape((-1, 192, 256))
        depth_tensor = torch.from_numpy(self._depth_data[indices])
        if self._binarize:
            depth_tensor = self.bin_pixels(depth_tensor)
        return depth_tensor.unsqueeze_(dim=1)  # B, 1, 192, 256

    def get_batch(self, indices: Union[np.ndarray, TrajectorySlice]):
        if self.zip_path is not None:
            return self.get_batch_zip(indices)
        elif self.bin_path is not None:
            return self.get_batch_bin(indices)
        else:
            raise ValueError("No depth data source provided")
