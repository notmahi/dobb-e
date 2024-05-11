"""
A base dataloader to benchmark against. Uses the precalculated frames to load the data.
"""


from copy import deepcopy
from os import listdir
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import PIL
import torch
from torchvision.transforms import PILToTensor

from dataloaders.abstract_dataloader import AbstractVideoDataset
from dataloaders.utils import DataLoaderConfig, TrajectorySlice, flatten_nested_lists


class DummyVideoReader:
    def __init__(self, path: Union[str, Path]):
        self._path = Path(path)
        self._num_frames = len(listdir(self._path / "compressed_images"))
        self._format = "%06d.jpg" if self._num_frames > 10000 else "%04d.jpg"
        self._convert = PILToTensor()

    def __len__(self):
        return self._num_frames

    def get_batch(self, indices: np.ndarray):
        indices = np.array(indices)
        assert np.all(indices >= 0) and np.all(
            indices < self._num_frames
        ), "Invalid indices for dummy video reader."
        frames = [
            PIL.Image.open(self._path / "compressed_images" / (self._format % index))
            for index in indices
        ]
        return torch.stack([self._convert(frame) for frame in frames])


class FileDataset(AbstractVideoDataset):
    def __init__(self, config: DataLoaderConfig):
        super().__init__(config)
        self._subslices = self._build_shuffle_index(deepcopy(self._slice_mapping))
        self._video_reader_cache: Dict[int, DummyVideoReader] = {}

    def _build_shuffle_index(
        self, initial_slice_mapping: List[Any]
    ) -> List[TrajectorySlice]:
        subslices = flatten_nested_lists(initial_slice_mapping)
        np.random.shuffle(subslices)
        return subslices

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        subslice = self._subslices[index]
        # First, get the video reader.
        video_reader = self._get_video_reader(subslice.trajectory_index)
        # Now, get the batch.
        indices = np.arange(subslice.start_index, subslice.end_index, subslice.skip + 1)
        batch = video_reader.get_batch(indices)
        actions = self._get_action_reader(subslice.trajectory_index).get_batch(indices)
        return batch, actions

    def __len__(self) -> int:
        return len(self._subslices)

    def _get_video_reader(self, index) -> DummyVideoReader:
        if index not in self._video_reader_cache:
            self._video_reader_cache[index] = DummyVideoReader(
                self._data_config.trajectories[index]
            )
        return self._video_reader_cache[index]
