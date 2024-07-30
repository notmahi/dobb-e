from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import Dataset

from dobbe.dataloaders.depth_data import DepthDataLoader
from dobbe.dataloaders.pose_data import PoseDataLoader
from dobbe.dataloaders.utils import DataLoaderConfig, calculate_start_end_indices


class AbstractVideoDataset(ABC, Dataset):
    EOF_SAFETY_MARGIN = 0

    def __init__(self, config: DataLoaderConfig, *args, **kwargs):
        self._data_config = config
        self._num_videos = len(self._data_config.trajectory_lengths)
        self._total_slices, self._slice_mapping = calculate_start_end_indices(
            self._data_config.trajectory_lengths,
            self._data_config.control_timeskip,
            self._data_config.sequence_length,
            safety_margin=self.EOF_SAFETY_MARGIN,
            fps_subsample=self._data_config.fps_subsample,
            n_passes=self._data_config.n_passes,
        )
        self._num_subslices = len(self._slice_mapping)
        self._action_reader_cache = {}
        self._depth_reader_cache = {}
        super().__init__(*args, **kwargs)

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def _get_action_reader(self, index: int) -> PoseDataLoader:
        if index not in self._action_reader_cache:
            self._action_reader_cache[index] = PoseDataLoader(
                pose_data_path=self._data_config.trajectories[index],
                control_timeskip=self._data_config.control_timeskip,
                fps_subsample=self._data_config.fps_subsample,
                relative_gripper=self._data_config.relative_gripper,
                binarize_gripper=self._data_config.binarize_gripper,
                binray_gripper_threshold=self._data_config.binarize_gripper_threshold,
                binarize_gripper_upper_value=self._data_config.binarize_gripper_upper_value,
                binarize_gripper_lower_value=self._data_config.binarize_gripper_lower_value,
            )
        return self._action_reader_cache[index]

    def _get_depth_reader(self, index: int) -> DepthDataLoader:
        if index not in self._depth_reader_cache:
            path = self._data_config.trajectories[index]
            if self._data_config.depth_cfg is not None:
                depth_loader = DepthDataLoader(**self._data_config.depth_cfg)
                path_parts = path.parts[-4:]
                path = "/".join(path_parts)  # takes the path from task name onwards
                depth_loader.set_depth_data_path(Path(path))
            else:
                depth_loader = DepthDataLoader(bin_root_path=path)
            self._depth_reader_cache[index] = depth_loader
        return self._depth_reader_cache[index]
