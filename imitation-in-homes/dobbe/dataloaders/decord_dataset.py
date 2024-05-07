"""
Build dataloaders based on our dataset. The core philosophy should be that there are simple
video files with labels for every frame. In the dataloaders, we will load the frames from the
video files, as well as computing relative labels from the label files, to use for training.
"""

import logging
import os
import shutil
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple, Union

import accelerate
import decord
import einops
import numpy as np
import torch
import tqdm

from dobbe.dataloaders.abstract_dataloader import AbstractVideoDataset
from dobbe.dataloaders.utils import DataLoaderConfig, TrajectorySlice, flatten_nested_lists


# Create a dataclass for shuffle modes.
# Use this information
# shuffle = -1  # smart shuffle mode, based on video properties
# shuffle = 0  # all sequential, no seeking, following initial filename order
# shuffle = 1  # random filename order, no random access for each video, very efficient
# shuffle = 2  # random order
# shuffle = 3  # random frame access in each video only
class ShuffleMode(Enum):
    SMART = -1
    SEQUENTIAL = 0
    RANDOM_FILENAME = 1
    RANDOM_ORDER = 2
    RANDOM_FRAME = 3


IMAGE_RANGE = 255.0


class DecordDataset(AbstractVideoDataset):
    EOF_SAFETY_MARGIN = 10

    def __init__(
        self,
        config: DataLoaderConfig,
        max_videoreaders_in_memory: int = 128,
        shuffle_mode: Union[int, str, ShuffleMode] = ShuffleMode.RANDOM_ORDER,
        device: str = "cpu",
        verbose: bool = True,
        copy_to_memory: bool = False,
    ):
        super().__init__(config)
        decord.bridge.set_bridge("torch")
        self._offset = 0  # offset is always 0 for starters.

        # Create VideoReader cache.
        self._max_videoreaders_in_memory = max_videoreaders_in_memory
        # Now calculate the total number of frames, and the mapping from index to frames.
        if isinstance(shuffle_mode, str):
            shuffle_mode = ShuffleMode[shuffle_mode.upper()]
        elif isinstance(shuffle_mode, int):
            shuffle_mode = ShuffleMode(shuffle_mode)
        self._verbose = verbose
        self._shuffle_mode = shuffle_mode
        self._epoch = 0
        self.shuffle()
        self._device = device
        self._index = 0
        self._total_calls = 0
        self._copy_to_memory = copy_to_memory
        self._filesystem_path_to_ram_path = {}
        self._include_trajectory_end = config.include_trajectory_end_flag

        # Create the LRU cache
        class VideoReaderCache(OrderedDict):
            def __setitem__(self, __key: Any, __value: Any) -> None:
                if len(self) >= max_videoreaders_in_memory:
                    _key, _val = self.popitem(last=False)
                    logging.debug(f"Removing {_key} from cache.")
                    del _val
                super().__setitem__(__key, __value)
                super().move_to_end(__key, last=True)

        self._video_reader_cache = VideoReaderCache()
        self._cached_worker_id = None
        self._cached_gpu_id = None

    def shuffle(self):
        # Set the epoch to the current epoch.
        self._epoch += 1
        # Now set numpy RNG to the right seed.
        self._rng = np.random.default_rng(self._epoch + self._data_config.seed)
        self._subslices = self._build_shuffle_index(
            deepcopy(self._slice_mapping), self._shuffle_mode
        )

    @property
    def subslices(self):
        return self._subslices

    @property
    def _worker_id(self):
        if self._cached_worker_id is None:
            worker_info = torch.utils.data.get_worker_info()
            self._cached_worker_id = 0 if worker_info is None else worker_info.id
        return self._cached_worker_id

    @property
    def _gpu_id(self):
        if self._cached_gpu_id is None:
            gpu_count = torch.cuda.device_count()
            self._cached_gpu_id = 0 if gpu_count == 0 else self._worker_id % gpu_count
        return self._cached_gpu_id

    def _build_shuffle_index(
        self, initial_slice_mapping: List[Any], shuffle_mode: ShuffleMode
    ) -> List[TrajectorySlice]:
        if shuffle_mode == ShuffleMode.SEQUENTIAL:
            # No need to shuffle, just use the index as is.
            subslices = flatten_nested_lists(initial_slice_mapping)
        elif shuffle_mode in (ShuffleMode.RANDOM_FILENAME, ShuffleMode.SMART):
            # Shuffle the filenames, but keep the order of the frames in each video.
            # Should be very efficient.
            if shuffle_mode == ShuffleMode.SMART:
                # Modify the slice mapping.
                slice_mapping = [
                    [flatten_nested_lists(x)] for x in initial_slice_mapping
                ]
                initial_slice_mapping = slice_mapping
            else:
                # Filter out empty sublists.
                initial_slice_mapping = [
                    [x for x in slice_mapping if len(x) > 0]
                    for slice_mapping in initial_slice_mapping
                ]
            subslices = []
            currently_considered_videos = []
            already_considered_videos = {
                index
                for index, slice_mapping in enumerate(initial_slice_mapping)
                if len(slice_mapping[0]) == 0
            }
            iterator = tqdm.trange(
                self._total_slices,
                postfix={
                    "considering": len(currently_considered_videos),
                    "considered": len(already_considered_videos),
                },
                disable=not self._verbose,
                desc="Shuffling dataset",
            )
            for _ in iterator:
                # Add a new video to the list.
                while len(
                    currently_considered_videos
                ) < self._max_videoreaders_in_memory and (
                    len(currently_considered_videos) + len(already_considered_videos)
                    < self._num_videos
                ):
                    # Add a new video to the list.
                    video_index = self._rng.integers(0, self._num_videos)
                    if (
                        video_index not in currently_considered_videos
                        and video_index not in already_considered_videos
                    ):
                        currently_considered_videos.append(video_index)
                # Now add all a subslice from the video to the list.
                video_index = self._rng.choice(currently_considered_videos)
                subslices.append(initial_slice_mapping[video_index][0].pop(0))

                # If the video is empty, remove it from the list.
                if len(initial_slice_mapping[video_index][0]) == 0:
                    initial_slice_mapping[video_index].pop(0)
                    currently_considered_videos.remove(video_index)
                    if len(initial_slice_mapping[video_index]) == 0:
                        already_considered_videos.add(video_index)
                    # already_considered_videos.add(video_index)
                iterator.set_postfix(
                    {
                        "considering": len(currently_considered_videos),
                        "considered": len(already_considered_videos),
                    }
                )

        elif shuffle_mode == ShuffleMode.RANDOM_ORDER:
            # Completely random order of all frames.
            subslices = flatten_nested_lists(initial_slice_mapping)
            self._rng.shuffle(subslices)

        elif shuffle_mode == ShuffleMode.RANDOM_FRAME:
            # Random order of frames, but keep the internal order in which we access the frames.
            subslices = flatten_nested_lists(initial_slice_mapping, max_level=1)
            self._rng.shuffle(subslices)
            subslices = flatten_nested_lists(subslices)

        else:
            raise ValueError(f"Unknown shuffle mode: {shuffle_mode}")

        return subslices

    def __len__(self) -> int:
        return self._total_slices

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        while self._index < len(self):
            index = self._index
            frames, actions = self[index]
            self._index += 1
            if self._index >= len(self):
                self._index = 0
            yield frames, actions

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        self._total_calls += 1
        # NOTE: This structure assumes that you are not using multiple workers.
        # If you are using multiple workers, you need to make sure that the
        # VideoReaders are not shared between workers.
        # If index is float64, cast to int.
        if isinstance(index, np.float64):
            index = index.astype(np.int64)
        trajectory_slice = self._subslices[index]
        frames = self._get_video_frames(trajectory_slice)
        actions = self._get_action_slice(trajectory_slice)
        if self._data_config.use_depth:
            depths = self._get_depth_frames(trajectory_slice)
            return frames, depths, actions
        return frames, actions

    def _get_depth_frames(self, trajectory_slice: TrajectorySlice) -> torch.Tensor:
        video_index = trajectory_slice.trajectory_index
        depth_indices = self._convert_subslice_to_depth_indices(trajectory_slice)
        depthreader = self._get_depth_reader(video_index)
        depths = depthreader.get_batch(depth_indices)
        return depths

    def _get_video_frames(self, trajectory_slice: TrajectorySlice) -> torch.Tensor:
        frame_indices = self._convert_subslice_to_video_indices(trajectory_slice)
        video_index = trajectory_slice.trajectory_index
        videoreader = self._get_video_reader(video_index)
        frames = videoreader.get_batch(frame_indices)
        # Convert image to ToTensor() result format.
        videoreader.seek(0)
        frames = einops.rearrange(frames, "... h w c -> ... c h w")
        if self._include_trajectory_end:
            frames = frames, trajectory_slice.to_seek
        return frames

    def set_include_trajectory_end(self, include_trajectory_end: bool = True):
        self._include_trajectory_end = include_trajectory_end

    def _get_action_slice(
        self, trajectory_slice: TrajectorySlice
    ) -> Union[torch.Tensor, np.ndarray]:
        video_index = trajectory_slice.trajectory_index
        action_indices = self._convert_subslice_to_action_indices(trajectory_slice)
        actionreader = self._get_action_reader(video_index)
        actions = actionreader.get_batch(action_indices)
        return actions

    def _convert_subslice_to_indices(
        self, trajectory_slice: TrajectorySlice
    ) -> Union[List[int], np.ndarray]:
        frame_indices = np.arange(
            trajectory_slice.start_index,
            trajectory_slice.end_index,
            trajectory_slice.skip,
        )
        return frame_indices

    def _convert_subslice_to_video_indices(
        self, trajectory_slice: TrajectorySlice
    ) -> Union[List[int], np.ndarray]:
        return self._convert_subslice_to_indices(trajectory_slice)

    def _convert_subslice_to_action_indices(
        self, trajectory_slice: TrajectorySlice
    ) -> Union[List[int], np.ndarray]:
        return self._convert_subslice_to_indices(trajectory_slice)

    def _convert_subslice_to_depth_indices(
        self, trajectory_slice: TrajectorySlice
    ) -> Union[List[int], np.ndarray]:
        return self._convert_subslice_to_indices(trajectory_slice)

    def _get_video_reader(self, index) -> decord.VideoReader:
        if index not in self._video_reader_cache:
            self._video_reader_cache[index] = decord.VideoReader(
                str(self._optionally_to_ram_path(index)),
                ctx=decord.cpu(0)
                if self._device == "cpu"
                else decord.gpu(self._gpu_id),
                width=self._data_config.width,
                height=self._data_config.height,
                num_threads=1,
            )
        return self._video_reader_cache[index]

    def _optionally_to_ram_path(self, index: int) -> Path:
        video_path: Union[Path, str] = (
            self._data_config.trajectories[index] / "compressed_video_h264.mp4"
        )
        if not self._copy_to_memory:
            return video_path
        # Copy the currently pointed file to RAM in /dev/shm and return the path.
        if str(video_path) not in self._filesystem_path_to_ram_path:
            # Copy the file to RAM.
            # Hash the video path directory to create a prefix:
            dir_hash = str(hash(video_path.parent))
            ram_path = os.path.join("/dev/shm", f"{dir_hash}_{video_path.name}")
            if not Path(ram_path).exists():
                shutil.copy2(video_path, ram_path)
            self._filesystem_path_to_ram_path[str(video_path)] = Path(ram_path)
        return self._filesystem_path_to_ram_path[str(video_path)]


class DecordSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: DecordDataset):
        # We need to know the possible subslices
        self._subslices = dataset.subslices
        self._num_workers = 1
        self._worker_id = 0
        self._filtered_subslices = None

    @property
    def filtered_subslices(self):
        if self._filtered_subslices is None:
            worker_info = torch.utils.data.get_worker_info()

            if worker_info is not None:
                self._num_workers = worker_info.num_workers
                self._worker_id = worker_info.id

            self._filtered_subslice_idxes = [
                i
                for i, slices in enumerate(self._subslices)
                if slices.trajectory_index % self._num_workers == self._worker_id
            ]
        return self._filtered_subslice_idxes

    def __len__(self):
        return len(self.filtered_subslices)

    def __iter__(self):
        return iter(self.filtered_subslices)


class DecordBatchSamplerUnderlying(torch.utils.data.BatchSampler):
    def __init__(
        self,
        dataset: DecordDataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        drop_last: bool = False,
    ):
        # We need to know the possible subslices
        self._subslices = dataset.subslices
        self._num_workers = 1
        self._worker_id = 0
        self._filtered_subslices = None

        self._batch_size = batch_size
        self._num_replicas = (
            num_replicas or accelerate.state.AcceleratorState().num_processes
        )
        self._drop_last = drop_last
        self._indexes = None

    def filter_subslices(self, subslices: List[TrajectorySlice]):
        # First, create the "holder" arrays.
        subslice_holder = [[] for _ in range(self._num_replicas)]
        # Now, iterate over the subslices, and add them to the right holder.
        for i, subslice in enumerate(subslices):
            subslice_holder[subslice.trajectory_index % self._num_replicas].append(i)

        subslice_holder = [np.array(x) for x in subslice_holder]
        # We try to make things equal length as much as possible. But when that is not possible,
        # we default to merging things.
        indexes = []
        reached_end = False
        last_insert = -1
        while not reached_end:
            for i in range(self._num_replicas):
                last_insert = i
                indexes.append(subslice_holder[i][: self._batch_size])
                subslice_holder[i] = subslice_holder[i][self._batch_size :]
                if reached_end := len(subslice_holder[i]) < self._batch_size:
                    break

        # Now combine whatever is left over and split that over the whole set.
        subslice_holder = np.concatenate(subslice_holder)
        insert_index = last_insert + 1
        while len(subslice_holder) > 0 and insert_index < self._num_replicas:
            indexes.append(subslice_holder[: self._batch_size])
            subslice_holder = subslice_holder[self._batch_size :]
            insert_index += 1
            insert_index %= self._num_replicas
        return np.concatenate(indexes)

    @property
    def indexes(self):
        if self._indexes is None:
            self._indexes = self.filter_subslices(self._subslices)
        return self._indexes

    def empty_index(self):
        self._indexes = None

    def __len__(self):
        return len(self.indexes)

    def __iter__(self):
        return iter(self.indexes)


class DecordBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self,
        dataset: DecordDataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        drop_last: bool = False,
    ):
        self._underlying_sampler = DecordBatchSamplerUnderlying(
            dataset, batch_size, num_replicas, drop_last
        )
        self._underlying_dataset = dataset
        super().__init__(
            self._underlying_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
        )

    def __iter__(self) -> Iterator[List[int]]:
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
        # Shuffle at the end of every epoch.
        self._underlying_dataset.shuffle()
        self._underlying_sampler.empty_index()
