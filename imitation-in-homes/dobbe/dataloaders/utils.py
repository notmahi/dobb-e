import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import tqdm

IMAGE_COMPLETION_FILENAME = "rgb_rel_videos_exported.txt"
POSES_FILENAME = "labels.json"
VIDEO_COMPLETION_FILENAME = "rgb_rel_videos_exported.txt"
SEED_OFFSET = 321654


def load_trajectory_roots(
    trajectory_root_path: Union[str, Path],
    original_root: Optional[str] = "/path/to/directory",
    new_root: Optional[str] = "/path/to/new/dataset_root",
) -> Iterable[Path]:
    try:
        with open(trajectory_root_path, "r") as f:
            trajectory_roots = json.load(f)
    except FileNotFoundError as e:
        print(f"Could not find file {trajectory_root_path}")
        raise e
    # Trim the .zip from the names before returning.
    if original_root is not None and new_root is not None:
        trajectory_roots = [
            str(root).replace(original_root, new_root) for root in trajectory_roots
        ]
    trajectory_roots = tuple(Path(str(root)[:-4]) for root in trajectory_roots)
    return trajectory_roots


def flatten_nested_lists(nested_list, level=0, max_level=-1):
    if max_level > 0 and level >= max_level:
        return nested_list
    flattened_list = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flattened_list.extend(
                flatten_nested_lists(sublist, level=level + 1, max_level=max_level)
            )
        else:
            flattened_list.append(sublist)
    return flattened_list


@dataclass
class TrajectorySlice:
    trajectory_index: int
    start_index: int
    end_index: int
    skip: int
    to_seek: bool = False


def calculate_start_end_indices(
    all_index_lengths: Iterable[int],
    skip: int,
    seq_len: int,
    safety_margin: int = 10,
    fps_subsample: int = 1,
    n_passes: int = -1,
):
    """
    A helper function to create the slices of the trajectory on which we load our data.
    Parameters:
        all_index_lengths:  A list of the lengths of all the trajectories.
        skip:               The number of frames to skip between frames in a single sample. 0
                            means no skipping, 1 means skip every other frame, etc.
        seq_len:            The number of frames to include in a single sample.
        safety_margin:      The number of frames to leave out at the end of the trajectory.
        fps_subsample:      The subsampling ratio from the original 30fps to the desired fps.
                            1 means no subsampling, 2 means subsample to 15fps, etc.
        n_passes:           The number of passes to make over the trajectory. -1 means
                            make as many passes as necessary to capture the whole dataset. 1
                            means make one pass over the dataset and then stop, etc.

    Returns:
        sum_total:          The total number of samples that will be loaded.
        result:             A list of lists of TrajectorySlice objects. The outer list
                            corresponds to the trajectories, and the inner list corresponds
                            to the different parts of the trajectory that will be loaded on a
                            single consecutive pass over the video.
    """
    result = []
    s = skip + 1
    k = seq_len
    f = fps_subsample
    sum_total = 0

    fast_forward_by = (s * k) // n_passes if n_passes > 0 else 1
    assert (
        fast_forward_by > 0
    ), f"Asked for too many passes over the dataset, can make at most {s * k} passes"
    for i, t in enumerate(all_index_lengths):
        traj_len = t - safety_margin
        start, end = (0, s * k * f)
        repeats = 0
        part_results = []
        smaller_parts = []
        total = 0
        passes_made = 0
        part_results.append(smaller_parts)
        while total < ((traj_len - (s * k * f)) // f + 1):
            if end <= traj_len:
                part_results[-1].append(TrajectorySlice(i, start, end, s * f, False))
                total += 1
                start = end
                end = start + s * k * f
            else:
                # Tell the last index to seek.
                # This is to avoid memory explosion as reported by some users.
                if len(smaller_parts):
                    smaller_parts[-1].to_seek = True
                smaller_parts = []
                part_results.append(smaller_parts)
                start, end = (
                    repeats + f * fast_forward_by,
                    repeats + (s * k * f) + f * fast_forward_by,
                )
                repeats += f
                passes_made += 1
                if n_passes > 0 and passes_made >= n_passes:
                    break
        if len(part_results[-1]):
            part_results[-1][-1].to_seek = True
        sum_total += total
        result.append(part_results)
    return sum_total, result


class DataLoaderConfig:
    """
    Generalized dataloader class that specified a common specification for all our dataloaders.

    This class will include a specification for which trajectories to include or not include,
    by specifying the task, home, and environments to include/exclude.

    On the control side, we will also specify the number of timesteps to skip between frames,
    as well as the image shape to use for the images.

    Finally, we will also specify the transformations to use on the images as we load them.

    dataset_root:       The root directory of where the dataset is stored.
    trajectory_roots:   A list of paths to the root directories of the trajectories to load.
    include_tasks:      A list of tasks to include in the dataset.
    exclude_tasks:      A list of tasks to exclude from the dataset.
    include_homes:      A list of homes to include in the dataset.
    exclude_homes:      A list of homes to exclude from the dataset.
    include_envs:       A list of environments to include in the dataset.
    exclude_envs:       A list of environments to exclude from the dataset.
    fps_subsample:      The subsampling ratio from the original 30fps to the desired fps.
                        1 means no subsampling, 2 means subsample to 15fps, etc.
    trajectory_subsample_count:
                        The number of trajectories to subsample from the dataset. None means
                        use all trajectories.
    trajectory_subsample_fraction:
                        The fraction of trajectories to subsample from the dataset. None or
                        1.0 means use all trajectories.
    n_passes:           The number of passes to make over the trajectory. -1 means
                        make as many passes as necessary to capture the whole dataset. 1
                        means make at most one pass over the dataset and then stop, etc.
    image_shape:        The shape of the images to load; should be a tuple of two integers.
    image_transforms:   A list of transformations to apply to the images as we load them.
    relative_gripper:   Whether to use the relative gripper position. Otherwise, use absolute.
    safe_action_threshold: The threshold for the safe action check. If this parameter is
                        defined, we ignore trajectories where the \delta XYZ between two
                        consecutive frames is greater than this threshold.
    trajectory_end_flag:A flag to indicate whether the dataset should include a boolean flag
                        end_trajectory to indicate where the trajectory ends.
    use_depth:              Whether to load depth data in addition to RGB data.
    depth_cfg:          A dictionary of depth-specific configuration parameters.
    seed:               The seed to use for the dataloader.
    binarize_gripper:   Whether to binarize the gripper, i.e. set all values above a threshold
                        to binarize_gripper_upper_value and all values below to binarize_gripper_lower_value.
    binarize_gripper_threshold: The threshold to use for binarizing the gripper.
    binarize_gripper_upper_value: The value to use for the upper value when binarizing the gripper.
    binarize_gripper_lower_value: The value to use for the lower value when binarizing the gripper.
    """

    def __init__(
        self,
        dataset_root: Union[str, Path] = "/vast/nms572/iphone_data_extracted",
        trajectory_roots: Iterable[Union[str, Path]] = (),
        include_tasks: Optional[Iterable[str]] = None,
        exclude_tasks: Optional[Iterable[str]] = None,
        include_homes: Optional[Iterable[str]] = None,
        exclude_homes: Optional[Iterable[str]] = None,
        include_envs: Optional[Iterable[str]] = None,
        exclude_envs: Optional[Iterable[str]] = None,
        trajectory_subsample_count: Optional[int] = None,
        trajectory_subsample_fraction: Optional[float] = None,
        fps_subsample: int = 1,
        n_passes: int = -1,
        image_shape: Tuple[int, int] = (256, 256),
        image_transforms: Iterable = (),
        control_timeskip: int = 1,
        sequence_length: int = 10,
        relative_gripper: bool = True,
        safe_action_threshold: Optional[float] = None,
        trajectory_end_flag: bool = False,
        use_depth: bool = False,
        depth_cfg: Optional[dict] = None,
        seed: int = 0,
        binarize_gripper: bool = False,
        binarize_gripper_threshold: float = 0.0,
        binarize_gripper_upper_value: float = 1.0,
        binarize_gripper_lower_value: float = 0.0,
    ):
        # We only need to specify one of include_tasks or exclude_tasks, not both.
        assert (include_tasks is None) != (exclude_tasks is None)
        # We only need to specify one of include_homes or exclude_homes, not both.
        assert (include_homes is None) != (exclude_homes is None)
        # Unspecific envs mean we should use all envs for the specified tasks and homes.
        # We only need to specify at most one of include_envs or exclude_envs, not both.
        assert (include_envs is None) or (exclude_envs is None)
        # If we specify a trajectory subsample count, it must be a positive integer.
        assert trajectory_subsample_count is None or (trajectory_subsample_count > 0)
        # If we specify a trajectory subsample fraction, it must be a positive float.
        assert trajectory_subsample_fraction is None or (
            trajectory_subsample_fraction > 0.0 and trajectory_subsample_fraction <= 1.0
        )
        # If we specify a trajectory subsample count, we cannot specify a trajectory subsample
        # fraction.
        assert (
            trajectory_subsample_count is None or trajectory_subsample_fraction is None
        )
        # Control timeskip must be a nonnegative integer.
        assert control_timeskip >= 0
        # Image shape must be a tuple of two positive integers.
        assert len(image_shape) == 2
        assert image_shape[0] > 0 and image_shape[1] > 0

        assert (
            not binarize_gripper or not relative_gripper
        ), "Cannot have both binarize_gripper and relative_gripper"

        self.binarize_gripper = binarize_gripper
        self.binarize_gripper_threshold = binarize_gripper_threshold
        self.binarize_gripper_upper_value = binarize_gripper_upper_value
        self.binarize_gripper_lower_value = binarize_gripper_lower_value

        self.dataset_root = Path(dataset_root)
        self.trajectory_roots = trajectory_roots
        if isinstance(include_tasks, str):
            include_tasks = (include_tasks,)
        self.include_tasks = set(include_tasks) if include_tasks else None
        self.exclude_tasks = set(exclude_tasks or [])
        if isinstance(include_homes, str):
            include_homes = (include_homes,)
        self.include_homes = set(include_homes) if include_homes else None
        self.exclude_homes = set(exclude_homes or [])
        if isinstance(include_envs, str):
            include_envs = (include_envs,)
        self.include_envs = set(include_envs) if include_envs else None
        self.exclude_envs = set(exclude_envs or [])
        self._trajectory_subsample_count = trajectory_subsample_count
        self._trajectory_subsample_fraction = trajectory_subsample_fraction
        self._do_subsample = (trajectory_subsample_count is not None) or (
            trajectory_subsample_fraction is not None
            and trajectory_subsample_fraction < 1.0
        )
        self.fps_subsample = fps_subsample
        self.n_passes = n_passes
        self.control_timeskip = control_timeskip
        self.sequence_length = sequence_length
        self.relative_gripper = relative_gripper
        self.image_shape = image_shape
        # TODO figure out height/width or other way around.
        self.height, self.width = image_shape
        self.image_transforms = image_transforms
        self._safe_action_threshold = safe_action_threshold or np.inf
        self.seed = seed + SEED_OFFSET
        self._include_trajectory_end_flag = trajectory_end_flag
        self.use_depth = use_depth
        self.depth_cfg = depth_cfg

        # Now, build the applicable list of trajectories
        self.trajectories = []
        self.trajectory_lengths = []
        self._build_trajectories()
        if len(self.trajectories) == 0:
            warnings.warn(
                "No trajectories were found matching the specified criteria. "
                "Make sure you have the correct dataset root and trajectory roots, "
                "and the correct include/exclude criteria."
            )

    def _validate_poses(self, poses) -> bool:
        if self._safe_action_threshold == np.inf:
            return True
        translations = []
        for _, data in poses.items():
            translations.append(data["xyz"])
        translations = np.array(translations, dtype=np.float32)
        return (
            np.linalg.norm(np.diff(translations, axis=0), axis=1).max()
            < self._safe_action_threshold
        )

    @property
    def include_trajectory_end_flag(self):
        return self._include_trajectory_end_flag

    def _build_trajectories(self):
        """
        Build the list of trajectories that will be used for this dataloader.
        """
        for root_path in tqdm.tqdm(
            self.trajectory_roots,
            desc="Filtering trajectories",
            total=len(self.trajectory_roots),
        ):
            root = Path(root_path)
            if not (root.exists() and root.is_dir()):
                continue
            # Figure out the task, home, and env for this trajecotry.
            relative_to_root_path = os.path.relpath(root, self.dataset_root)
            task, home, env, traj = relative_to_root_path.split(os.sep)[:4]
            # Check if this trajectory should be included or excluded.
            if (
                task in self.exclude_tasks
                or home in self.exclude_homes
                or f"{task}/{home}" in self.exclude_homes
                or env in self.exclude_envs
                or f"{home}/{env}" in self.exclude_envs
                or f"{task}/{home}/{env}" in self.exclude_envs
            ):
                continue

            # Now, check if the include_* sets are specified. If they are, only continue if
            # the trajectory is in the include_* set.
            if self.include_tasks and task not in self.include_tasks:
                continue
            if self.include_homes and home not in self.include_homes:
                continue
            if self.include_envs and env not in self.include_envs:
                continue

            # If we get here, then we should include this trajectory. Check for the relevant
            # files, and add it to the list of trajectories.
            if (
                not (root / VIDEO_COMPLETION_FILENAME).exists()
                or not (root / POSES_FILENAME).exists()
            ):
                continue
            with open(root / POSES_FILENAME, "r") as f:
                poses = json.load(f)
            if not self._validate_poses(poses):
                continue
            self.trajectories.append(root)
            self.trajectory_lengths.append(len(poses))
        # Subsample trajectories.
        if self._do_subsample:
            self._subsample_trajectories()

    def _subsample_trajectories(self):
        n_trajectories = len(self.trajectories)
        # If we have a trajectory subsample count, use that, otherwise calculate trajectories
        # to subsample based on the fraction.
        n_subsampled_trajectories = self._trajectory_subsample_count or int(
            self._trajectory_subsample_fraction * n_trajectories
        )
        if n_subsampled_trajectories <= 0:
            raise ValueError(
                f"Specified trajectory subsample fraction {self._trajectory_subsample_fraction} is too small."
            )
        if n_subsampled_trajectories > n_trajectories:
            raise ValueError(
                f"Specified trajectory subsample fraction {self._trajectory_subsample_fraction} is too large."
            )
        # Subsample the trajectories.
        rng = np.random.RandomState(self.seed)
        chosen_indices = rng.choice(
            n_trajectories, size=n_subsampled_trajectories, replace=False
        )
        self.trajectories = [self.trajectories[i] for i in chosen_indices]
        self.trajectory_lengths = [self.trajectory_lengths[i] for i in chosen_indices]

    def __len__(self):
        return len(self.trajectories)
