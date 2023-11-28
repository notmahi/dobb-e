"""
This task runner assumes that we have the r3d files as zip files in the folder, with the 
general structure of task_name/home_id/env_id/timestamp.zip

We will unzip the files, and then process them one by one.
"""

import argparse
import json
import logging
import os
import pickle as pkl
import shutil
import traceback
import zipfile
from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple, cast

import cv2
import liblzfse
import numpy as np
import PIL
import torch
from quaternion import as_rotation_matrix, quaternion
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import has_file_allowed_extension
from tqdm import tqdm

from utils.action_transforms import apply_permutation_transform
from utils.error_handlers import CustomFormatter, send_slack_message
from utils.models import GripperNet

torch.set_num_threads(1)
COMPLETION_FILENAME = "completed.txt"
ABANDONED_FILENAME = "abandoned.txt"
logger = logging.getLogger("R3D Processing")
logger.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


class LabelFreeImageFolder(ImageFolder):
    def find_classes(self, directory: str):
        return [], {}

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int] = dict(),
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the same time"
            )

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []

        for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, 0  # The images have no classes.
                    instances.append(item)

        return instances


def label_image_folder(
    image_folder,
    labelling_model_path="./gripper_model.pth",
    device="cpu",
    batch_size=64,
):
    # We will use the labelling model to label the images in the image folder.
    # We will then save the labels in the same folder as the images.
    model = GripperNet()
    model.to(device)
    model.load_state_dict(torch.load(labelling_model_path, map_location=device))
    model.eval()

    dataset = LabelFreeImageFolder(
        image_folder,
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    labels = []
    with torch.no_grad():
        for X, _ in dataloader:
            batch = X.to(device)
            output = model(batch)
            labels.append(output.cpu().numpy())

    labels = np.concatenate(labels, axis=0)
    return labels


class R3DZipFileProcessor:
    # TODO: rotate the images and label them with the model.
    def __init__(self, path, model_path, device="cpu"):
        self.path = path
        assert self.path.endswith(".zip")
        self._extracted_path = self.path[:-4]
        self.model_path = model_path
        self.device = device

        # Keep a cache of the last rotated images because sometimes the RGB file coming out
        # of the iphone/R3D app is corrputed.
        self._last_rotated_image = None

    def process(self):
        try:
            return self._process()
        except Exception as e:
            error_message = traceback.format_exc()
            logger.error(f"Error processing {self.path}: {e}")
            logger.error(error_message)
            # send_slack_message(f"Error processing {self.path}: {e}")
            # send_slack_message(error_message)
            return self._process(redo_everything=True)

    def _process(self, redo_everything=False):
        try:
            self.process_metadata()
        except zipfile.BadZipFile as e:
            logger.error(f"Error extracting metadata from {self.path}: {e}")
            with open(os.path.join(self._extracted_path, ABANDONED_FILENAME), "w") as f:
                f.write("Abandoned\n")
                f.write(str(e))
            return False
        self.extract_images(redo_everything=redo_everything)
        transforms = self.process_poses()
        gripper_labels = self.process_gripper_positions(
            os.path.join(self._extracted_path, "images")
        )
        assert self.validate()
        self.save_transforms(transforms, gripper_labels)
        with open(os.path.join(self._extracted_path, COMPLETION_FILENAME), "w") as f:
            f.write("Completed")
        return True

    def process_metadata(self):
        # Extract and process the "metadata" file from the r3d export.
        # This file contains the camera intrinsics, the camera extrinsics, and the
        # camera transforms.
        logger.info("Processing metadata")
        os.makedirs(self._extracted_path, exist_ok=True)
        with zipfile.ZipFile(self.path, "r") as zip_ref:
            zip_ref.extract("metadata", self._extracted_path)

        with open(os.path.join(self._extracted_path, "metadata"), "r") as f:
            self._metadata_dict = json.load(f)
        metadata_dict = self._metadata_dict

        # Now figure out the details from the metadata dict.
        self.rgb_width = metadata_dict["w"]
        self.rgb_height = metadata_dict["h"]
        self.image_size = (self.rgb_width, self.rgb_height)

        self.poses = np.array(metadata_dict["poses"])
        self.camera_matrix = np.array(metadata_dict["K"]).reshape(3, 3).T

        self.fps = metadata_dict["fps"]

        self.total_images = len(self.poses)
        self.init_pose = np.array(metadata_dict["initPose"])

    @staticmethod
    def _process_filename(filename):
        if filename == "rgbd":
            return filename
        name, extension = filename.split(".")
        assert extension in ["jpg", "depth", "conf"]
        return f"rgbd/{int(name)}.{extension}"

    def extract_images(self, redo_everything=False):
        # First, we create the proper directory structure.
        # We assume that the zip files are in the format of task/home/env/timestamp.zip
        # We will unzip them to task/home/env/timestamp/
        logger.info("Extracting images")

        with zipfile.ZipFile(self.path, "r") as zip_ref:
            all_files = zip_ref.namelist()

        rgb_files = {f for f in all_files if f.endswith(".jpg")}
        depth_files = {f for f in all_files if f.endswith(".depth")}
        conf_files = {f for f in all_files if f.endswith(".conf")}

        rgbfolder = os.path.join(self._extracted_path, "unrotated_images")
        # Process the RGB images.
        os.makedirs(rgbfolder, exist_ok=True)
        # Now, remove the files that are already extracted and therefore should not be extracted again.
        to_extract = (
            rgb_files
            - {R3DZipFileProcessor._process_filename(x) for x in os.listdir(rgbfolder)}
            if not redo_everything
            else rgb_files
        )
        with zipfile.ZipFile(self.path, "r") as zip_ref:
            zip_ref.extractall(rgbfolder, members=list(to_extract))
        # TODO: Figure out how to rename in an idempotent way.
        R3DZipFileProcessor._rename_to_sequential(rgbfolder, extension=".jpg")
        # At the same time, rotate the images.
        self.rotate_images(rgbfolder, redo_everything=redo_everything)

        # Process the depth images.
        depthfolder = os.path.join(self._extracted_path, "compressed_depths")
        os.makedirs(depthfolder, exist_ok=True)
        to_extract = (
            depth_files
            - {
                R3DZipFileProcessor._process_filename(x)
                for x in os.listdir(depthfolder)
            }
            if not redo_everything
            else depth_files
        )
        with zipfile.ZipFile(self.path, "r") as zip_ref:
            zip_ref.extractall(depthfolder, members=list(to_extract))
        R3DZipFileProcessor._rename_to_sequential(depthfolder, extension=".depth")

        # Process the conf images.
        conffolder = os.path.join(self._extracted_path, "compressed_confs")
        os.makedirs(conffolder, exist_ok=True)
        to_extract = (
            conf_files
            - {R3DZipFileProcessor._process_filename(x) for x in os.listdir(conffolder)}
            if not redo_everything
            else conf_files
        )
        with zipfile.ZipFile(self.path, "r") as zip_ref:
            zip_ref.extractall(conffolder, members=conf_files)
        R3DZipFileProcessor._rename_to_sequential(conffolder, extension=".conf")
        return rgbfolder, depthfolder, conffolder

    def rotate_images(self, rgb_path, redo_everything=False):
        # Rotate the images by 90 degrees, since the iphone captures images in portrait mode.
        # We do it this way to make sure the operation is idempotent, since rotating an image
        # 90 degrees and replacing the original one is not.
        logger.info("Rotating images")
        rotated_path = os.path.join(self._extracted_path, "images")
        compressed_path = os.path.join(self._extracted_path, "compressed_images")
        os.makedirs(rotated_path, exist_ok=True)
        os.makedirs(compressed_path, exist_ok=True)
        for f in sorted(os.listdir(rgb_path)):
            if os.path.exists(os.path.join(rotated_path, f)) and not redo_everything:
                continue
            try:
                image_path = os.path.join(rgb_path, f)
                _ = PIL.Image.open(image_path)
                img = cv2.imread(image_path)
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                self._last_rotated_image = img
            except PIL.UnidentifiedImageError as e:
                logger.error(f"Error rotating {f}: {e}")
                if self._last_rotated_image is not None:
                    img = self._last_rotated_image
                    self._last_rotated_image = None
                else:
                    raise e

            cv2.imwrite(os.path.join(rotated_path, f), img)
            # Now compress the image.
            compressed_img = cv2.resize(img, (256, 256))
            cv2.imwrite(os.path.join(compressed_path, f), compressed_img)

    def process_poses(self):
        # Process the poses from the metadata file.
        # We will convert the poses to a list of rotation matrices and translation vectors.
        logger.info("Processing poses")
        self.quaternions = []
        self.translation_vectors = []
        init_pose = None
        for pose in self.poses:
            qx, qy, qz, qw, px, py, pz = pose
            extrinsic_matrix = np.eye(4)
            extrinsic_matrix[:3, :3] = as_rotation_matrix(quaternion(qw, qx, qy, qz))
            extrinsic_matrix[:3, -1] = [px, py, pz]

            if init_pose is None:
                init_pose = np.copy(extrinsic_matrix)

            # We will convert the extrinsic matrix to the camera pose.
            # The camera pose is the inverse of the extrinsic matrix.
            relative_pose = np.linalg.inv(init_pose) @ extrinsic_matrix
            transformed_pose = apply_permutation_transform(relative_pose)
            self.translation_vectors.append(transformed_pose[:3, -1])
            self.quaternions.append(R.from_matrix(transformed_pose[:3, :3]).as_quat())
        quats = np.array(self.quaternions)
        translations = np.array(self.translation_vectors)
        transforms = np.concatenate([translations, quats], axis=1)
        return transforms

    def process_gripper_positions(self, rgb_folder):
        logger.info("Processing gripper positions")
        gripper_labels = label_image_folder(
            rgb_folder, self.model_path, device=self.device
        )
        return gripper_labels

    @staticmethod
    def _rename_to_sequential(path, extension=".jpg"):
        filenames, file_indices = [], []
        if os.path.exists(os.path.join(path, "rgbd")):
            base_path = os.path.join(path, "rgbd")
        else:
            base_path = path
        for i, f in enumerate(sorted(os.listdir(base_path))):
            filenames.append(f)
            index, ext = f.split(".")
            assert ext == extension[1:]
            file_index = int(index)
            assert file_index >= 0
            file_indices.append(file_index)

        for f, i in zip(filenames, file_indices):
            new_fname_short = f"{i:04d}{extension}"
            new_fname_long = f"{i:06d}{extension}"
            if len(filenames) <= 10_000:
                new_fname = new_fname_short
            else:
                new_fname = new_fname_long
            os.rename(os.path.join(base_path, f), os.path.join(path, new_fname))
            if len(filenames) > 10_000:
                if len(new_fname_short) != len(new_fname_long):
                    # Make sure only one of those exists.
                    assert not (
                        os.path.exists(os.path.join(base_path, new_fname_short))
                        and os.path.exists(os.path.join(base_path, new_fname_long))
                    )

        if os.path.exists(os.path.join(path, "rgbd")):
            assert len(os.listdir(base_path)) == 0
            shutil.rmtree(base_path)

    def validate(self):
        logger.info("Validating the extracted data")
        # TODO: Add validation functions on the images and the poses.
        return True

    def save_transforms(self, transforms, gripper_labels):
        logger.info("Saving the extracted data")
        translations, rotations = transforms[:, :3], transforms[:, 3:]
        new_data = {
            i: {
                "xyz": xyz.tolist(),
                "quats": quats.tolist(),
                "gripper": grp.item(),
            }
            for i, (xyz, quats, grp) in enumerate(
                zip(translations, rotations, gripper_labels)
            )
        }
        with open(os.path.join(self._extracted_path, "labels.json"), "w") as f:
            json.dump(new_data, f)

        with open(os.path.join(self._extracted_path, "relative_poses.pkl"), "wb") as f:
            pkl.dump(transforms, f)


def filter_r3d_files_to_process(r3d_paths_file):
    with open(r3d_paths_file, "r") as f:
        r3d_paths = json.load(f)

    to_process = []
    # We will filter out the ones that have already been processed.
    for path in r3d_paths:
        assert path.endswith(".zip")
        completed = os.path.exists(os.path.join(path[:-4], COMPLETION_FILENAME))
        abandoned = os.path.exists(os.path.join(path[:-4], ABANDONED_FILENAME))
        if os.path.exists(path[:-4]) and (completed or abandoned):
            continue
        to_process.append(path)

    return to_process


def process_r3d_file(file_path, model_path):
    logger.info(f"Processing {file_path}")
    processor = R3DZipFileProcessor(file_path, model_path)
    try:
        processor.process()
        logger.info(f"Finished processing {file_path}")
        return True
    except Exception as e:
        error_message = traceback.format_exc()
        logger.error(f"Error processing {file_path}: {e}")
        logger.error(error_message)
        return False


# Define argparse arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--r3d_paths_file",
    type=str,
    required=True,
    help="Path to the file containing the paths to the r3d files.",
)
parser.add_argument(
    "--count_only",
    action="store_true",
    help="If set, will only count the number of r3d files to process.",
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the model used to detect the gripper.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="Number of workers to use to process the r3d files.",
)
parser.add_argument(
    "--start_index",
    type=int,
    default=0,
    help="Index to start processing the r3d files from.",
)
parser.add_argument(
    "--end_index",
    type=int,
    default=-1,
    help="Index to end processing the r3d files at.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    r3d_paths_file = args.r3d_paths_file
    model_path = args.model_path
    num_workers = args.num_workers
    start_index = args.start_index
    end_index = args.end_index

    # Filter out the r3d files that have already been processed.
    r3d_paths = filter_r3d_files_to_process(r3d_paths_file)
    if end_index == -1:
        end_index = len(r3d_paths)
    logger.info(f"Number of r3d files to process: {len(r3d_paths)}")
    if args.count_only:
        exit()
    r3d_paths = r3d_paths[start_index:end_index]

    # Process the r3d files.
    with Pool(num_workers) as p:
        p.map(
            partial(process_r3d_file, model_path=model_path),
            tqdm(r3d_paths, desc="Processing r3d files"),
        )

    logger.info("Finished processing all r3d files.")
