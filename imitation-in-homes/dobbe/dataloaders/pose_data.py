import json
from pathlib import Path
from typing import Union

import numpy as np
from scipy.spatial.transform import Rotation as R

from dataloaders.utils import TrajectorySlice

LABEL_FILENAME = "labels.json"
PICKLE_FILENAME = "relative_poses.pkl"


class PoseDataLoader:
    def __init__(
        self,
        pose_data_path: Union[str, Path],
        control_timeskip: int = 1,
        fps_subsample: int = 1,
        relative_gripper: bool = True,
        binarize_gripper: bool = False,
        binray_gripper_threshold: float = 0.0,
        binarize_gripper_upper_value: float = 1.0,
        binarize_gripper_lower_value: float = 0.0,
    ):
        self._binarize_gripper = binarize_gripper
        self._binarize_gripper_threshold = binray_gripper_threshold
        self._binarize_gripper_upper_value = binarize_gripper_upper_value
        self._binarize_gripper_lower_value = binarize_gripper_lower_value
        self.pose_data_path = Path(pose_data_path)
        self._k = (control_timeskip + 1) * fps_subsample
        self._load_pose_data()
        self._relative_gripper = relative_gripper

    def _load_pose_data(self):
        json_path = self.pose_data_path / LABEL_FILENAME
        # First, load the pose data from the json file.
        with json_path.open("r") as f:
            pose_data = json.load(f)
        # Now, convert the gripper data to a numpy array.
        translations, rotations, gripper = [], [], []
        for index, data in pose_data.items():
            translations.append(data["xyz"])
            rotations.append(data["quats"])
            gripper_value = data["gripper"]
            if self._binarize_gripper:
                gripper_value = (
                    self._binarize_gripper_upper_value
                    if gripper_value > self._binarize_gripper_threshold
                    else self._binarize_gripper_lower_value
                )
            gripper.append(gripper_value)

        self._translations = np.array(translations, dtype=np.float32)
        self._rotations = np.array(rotations, dtype=np.float32)
        self._gripper = np.array(gripper, dtype=np.float32)
        self._len = len(self._translations)

    def __len__(self):
        return self._len

    def get_batch(self, indices: Union[np.ndarray, TrajectorySlice]):
        if isinstance(indices, TrajectorySlice):
            indices = np.arange(
                indices.start_index, indices.end_index, indices.skip + 1
            )
        # First, make sure the indices are valid.
        indices = np.array(indices)
        n = len(indices)  # Assume the indices is a linear array.
        assert np.all(indices >= 0) and np.all(
            indices + self._k < self._len
        ), "Invalid indices for pose data loader."
        # Now, load the data.
        prior_translations, prior_rotations = (
            self._translations[indices],
            self._rotations[indices],
        )
        next_translations, next_rotations = (
            self._translations[indices + self._k],
            self._rotations[indices + self._k],
        )
        # Now, create the matrices.
        prior_rot_matrices, next_rot_matrices = (
            R.from_quat(prior_rotations).as_matrix(),
            R.from_quat(next_rotations).as_matrix(),
        )
        # Now, compute the relative matrices.
        prior_matrices = np.tile(np.eye(4), (n, 1, 1))
        prior_matrices[:, :3, :3] = prior_rot_matrices
        prior_matrices[:, :3, 3] = prior_translations

        next_matrices = np.tile(np.eye(4), (n, 1, 1))
        next_matrices[:, :3, :3] = next_rot_matrices
        next_matrices[:, :3, 3] = next_translations

        relative_transforms = np.matmul(np.linalg.inv(prior_matrices), next_matrices)
        relative_translations = relative_transforms[:, :3, 3]
        relative_rotations = R.from_matrix(relative_transforms[:, :3, :3]).as_rotvec()

        # Finally, take the gripper values
        if self._relative_gripper:
            gripper = (
                self._gripper[indices + self._k] - self._gripper[indices]
            ).reshape(-1, 1)
        else:
            gripper = self._gripper[indices + self._k].reshape(-1, 1)

        return np.concatenate(
            [relative_translations, relative_rotations, gripper],
            axis=1,
            dtype=np.float32,
        )
