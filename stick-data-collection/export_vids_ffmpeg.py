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
import subprocess
import time
import traceback
from enum import Enum
from functools import partial
from itertools import tee
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Tuple, Union

import liblzfse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

from utils.error_handlers import CustomFormatter, send_slack_message

logger = logging.getLogger("R3D Processing")
logger.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)


COMPLETION_FILENAME = "rgb_rel_videos_exported.txt"
IMG_COMPLETION_FILENAME = "completed.txt"
ABANDONED_FILENAME = "abandoned.txt"
RGB_VIDEO_NAME = "compressed_video.mp4"
RGB_VIDEO_H264_NAME = "compressed_video_h264.mp4"
REL_ACTIONS_VIDEO_NAME = "video_rel_actions.avif"
DEPTH_FOLDER_NAME = "compressed_depths"
COMPLETED_DEPTH_FILENAME = "compressed_np_depth_float32.bin"
METADATA = "metadata"


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def load_depth(raw_bytes, aspect_ratio):
    decompressed_bytes = liblzfse.decompress(raw_bytes)
    depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)
    if aspect_ratio > 1:
        depth_img = depth_img.reshape((192, 256))
    else:
        depth_img = depth_img.reshape((256, 192))
        depth_img = np.ascontiguousarray(np.rot90(depth_img, -1))
    return depth_img  # 1,192,256

def get_aspect_ratio(root_path: Path) -> float:
    metadata_dict = json.load(open(root_path / METADATA))
    width = metadata_dict["w"]
    height = metadata_dict["h"]
    return width / height

class VideoModes(Enum):
    RGB = 0
    REL_ACTIONS = 1
    RECONSTRUCTED_PCD = 2
    DEPTH = 3


class VideoProcessor:
    def __init__(
        self,
        root_path: Union[Path, str],
        export_path: Union[Path, str],
        video_modes: Iterable[VideoModes],
        fps: int = 6,
        skip_frames: int = 5,
    ):
        self.root_path = Path(root_path)
        # Extract the timestamp from the root path.
        self.timestamp = self.root_path.name
        self.video_modes = video_modes
        self.export_path = Path(export_path)
        self._fps = fps
        self._skip_frames = skip_frames
        self.aspect_ratio = get_aspect_ratio(self.root_path)

    def process(self):
        # if VideoModes.REL_ACTIONS in self.video_modes:
        #     self.process_rel_actions()
        if VideoModes.RECONSTRUCTED_PCD in self.video_modes:
            self.process_reconstructed_pcd()
        if VideoModes.DEPTH in self.video_modes:
            self.process_depth()
        if VideoModes.RGB in self.video_modes:
            self.process_rgb()
        with open(self.export_path / COMPLETION_FILENAME, "w") as f:
            f.write("Done.")

    def process_rgb(self):
        start_time = time.perf_counter()
        # First, find out a sample filename
        sample_filename = next((self.root_path / "compressed_images").glob("*.jpg"))
        if sample_filename is None:
            logging.error(f"No images found in {self.root_path}")
            return
        # Find out if the filename is 4 or 6 digits long.
        if len(sample_filename.stem) == 4:
            filename_format = "%04d.jpg"
        elif len(sample_filename.stem) == 6:
            filename_format = "%06d.jpg"
        else:
            logging.error(f"Unknown filename format: {sample_filename.stem}")
            return
        # Now, we create the videos using ffmpeg.
        # First, we will create the h264 video.
        hevc_video_path = self.export_path / RGB_VIDEO_NAME
        h264_video_path = self.export_path / RGB_VIDEO_H264_NAME
        crfs = [30, 30]
        video_codecs = ["hevc", "h264"]
        for enc_lib, crf, final_video_path in zip(
            video_codecs, crfs, [hevc_video_path, h264_video_path]
        ):
            command = [
                "ffmpeg",
                "-y",
                "-framerate",
                "30",
                "-i",
                "compressed_images/{}".format(filename_format),
                "-c:v",
                enc_lib,
                "-crf",
                str(crf),
                str(final_video_path),
            ]
            process = subprocess.run(
                command,
                capture_output=True,
                check=True,
                cwd=self.root_path,
            )
            process.check_returncode()
            logging.info(process.stdout.decode("utf-8"))
            logging.debug(process.stderr.decode("utf-8"))

        end_time = time.perf_counter()
        logger.info(
            f"Saved RGB video to {self.export_path} in {end_time - start_time}s"
        )

    def process_depth(self):
        # First, find out a sample filename
        target_depth_filename = self.root_path / COMPLETED_DEPTH_FILENAME
        if not target_depth_filename.exists():
            sample_filename = next((self.root_path / DEPTH_FOLDER_NAME).glob("*.depth"))
            if sample_filename is None:
                logging.error(f"No depth data found in {self.root_path}")
                return
            # Find out if the filename is 4 or 6 digits long.
            if len(sample_filename.stem) == 4:
                filename_format = "%04d.depth"
            elif len(sample_filename.stem) == 6:
                filename_format = "%06d.depth"
            else:
                logging.error(f"Unknown filename format: {sample_filename.stem}")
                return

            idx = 0
            all_depth_data = []
            while True:
                new_path = self.root_path / DEPTH_FOLDER_NAME / (filename_format % idx)
                if not new_path.exists():
                    break
                all_depth_data.append(load_depth(new_path.read_bytes(), self.aspect_ratio))
                idx += 1
            all_depth_data = np.stack(all_depth_data, axis=0)
            # Now zip and save this depth data.
            depth_array = all_depth_data
            depth_bytes = liblzfse.compress(depth_array.tobytes())
            target_depth_filename.write_bytes(depth_bytes)

    def process_rel_actions(self):
        start_time = time.perf_counter()
        # First, load the actions file.
        transforms = pkl.load(open(self.root_path / "relative_poses.pkl", "rb"))
        # Now, figure out the relative actions.
        first_matrix = np.eye(4)
        all_transform_matrices = [first_matrix]
        for prior_transform, next_transform in pairwise(
            transforms[:: self._skip_frames]
        ):
            translation, rotation = prior_transform[:3], prior_transform[3:]
            prior_matrix = np.eye(4)
            prior_matrix[:3, :3] = Rot.from_quat(rotation).as_matrix()
            prior_matrix[:3, 3] = translation
            translation, rotation = next_transform[:3], next_transform[3:]
            next_matrix = np.eye(4)
            next_matrix[:3, :3] = Rot.from_quat(rotation).as_matrix()
            next_matrix[:3, 3] = translation
            all_transform_matrices.append(np.linalg.inv(prior_matrix) @ next_matrix)

        T = all_transform_matrices[0]
        R, t = T[:3, :3], T[:3, 3]
        # Now, we will create a video of the relative actions.
        segment_length = 0.01  # 1 cm
        # Define the original points
        points = np.array(
            [
                [0, 0, 0],
                [segment_length, 0, 0],
                [0, segment_length, 0],
                [0, 0, segment_length],
            ]
        )
        points_transformed = T @ np.vstack([points.T, np.ones(points.shape[0])])
        points_transformed = points_transformed[:3, :].T
        fig = plt.figure(figsize=(20, 12))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the original points and set the axis limits
        ax.plot(
            [0, segment_length],
            [0, 0],
            [0, 0],
            c="r",
            linestyle="--",
            alpha=0.25,
            label="x-axis",
        )
        ax.plot(
            [0, 0],
            [0, segment_length],
            [0, 0],
            c="g",
            linestyle="--",
            alpha=0.25,
            label="y-axis",
        )
        ax.plot(
            [0, 0],
            [0, 0],
            [0, segment_length],
            c="b",
            linestyle="--",
            alpha=0.25,
            label="z-axis",
        )
        # Plot the transformed points and the new axes
        x_axis_transformed = R @ np.array([segment_length, 0, 0]) + t
        y_axis_transformed = R @ np.array([0, segment_length, 0]) + t
        z_axis_transformed = R @ np.array([0, 0, segment_length]) + t
        line_1 = ax.plot(
            [t[0], x_axis_transformed[0]],
            [t[1], x_axis_transformed[1]],
            [t[2], x_axis_transformed[2]],
            c="r",
        )
        line_2 = ax.plot(
            [t[0], y_axis_transformed[0]],
            [t[1], y_axis_transformed[1]],
            [t[2], y_axis_transformed[2]],
            c="g",
        )
        line_3 = ax.plot(
            [t[0], z_axis_transformed[0]],
            [t[1], z_axis_transformed[1]],
            [t[2], z_axis_transformed[2]],
            c="b",
        )
        ax.grid(False)

        def update_image(idx):
            T = all_transform_matrices[idx]
            R, t = T[:3, :3], T[:3, 3]
            # Apply the transformation to the points
            points_transformed = T @ np.vstack([points.T, np.ones(points.shape[0])])
            points_transformed = points_transformed[:3, :].T

            x_axis_transformed = R @ np.array([segment_length, 0, 0]) + t
            y_axis_transformed = R @ np.array([0, segment_length, 0]) + t
            z_axis_transformed = R @ np.array([0, 0, segment_length]) + t

            all_xs = [
                t[0],
                x_axis_transformed[0],
                y_axis_transformed[0],
                z_axis_transformed[0],
                0,
                segment_length,
            ]
            all_ys = [
                t[1],
                x_axis_transformed[1],
                y_axis_transformed[1],
                z_axis_transformed[1],
                0,
                segment_length,
            ]
            all_zs = [
                t[2],
                x_axis_transformed[2],
                y_axis_transformed[2],
                z_axis_transformed[2],
                0,
                segment_length,
            ]
            all_xyzs = all_xs + all_ys + all_zs

            boundary = 0.005

            line_1[0].set_data_3d(
                [t[0], x_axis_transformed[0]],
                [t[1], x_axis_transformed[1]],
                [t[2], x_axis_transformed[2]],
            )
            line_2[0].set_data_3d(
                [t[0], y_axis_transformed[0]],
                [t[1], y_axis_transformed[1]],
                [t[2], y_axis_transformed[2]],
            )
            line_3[0].set_data_3d(
                [t[0], z_axis_transformed[0]],
                [t[1], z_axis_transformed[1]],
                [t[2], z_axis_transformed[2]],
            )
            lims = max(abs(min(all_xyzs) - boundary), abs(max(all_xyzs) + boundary))
            ax.set_xlim(-lims, lims)
            ax.set_ylim(-lims, lims)
            ax.set_zlim(-lims, lims)
            return None

        anim = FuncAnimation(
            fig,
            update_image,
            frames=len(all_transform_matrices),
            interval=1000 / self._fps,
        )
        output_path = self.export_path / REL_ACTIONS_VIDEO_NAME.format(
            index=self.timestamp
        )
        writer = FFMpegWriter(fps=self._fps, codec="libsvtav1", bitrate=-1, metadata={})
        fig.set_size_inches(4, 4, True)
        anim.save(output_path, writer=writer, dpi=64)
        plt.close()
        end_time = time.perf_counter()
        logger.info(
            f"Saved relative action video to {output_path} in {end_time - start_time}s"
        )

    def process_reconstructed_pcd(self):
        raise NotImplementedError("Need to implement reconstructed PCD")


def filter_r3d_files_to_process(r3d_paths_file):
    with open(r3d_paths_file, "r") as f:
        r3d_paths = json.load(f)

    to_process = []
    # We will filter out the ones that have already been processed.
    for path in r3d_paths:
        assert path.endswith(".zip")
        if os.path.exists(path[:-4]) and (
            os.path.exists(os.path.join(path[:-4], COMPLETION_FILENAME))
            or os.path.exists(os.path.join(path[:-4], ABANDONED_FILENAME))
            or not os.path.exists(os.path.join(path[:-4], IMG_COMPLETION_FILENAME))
        ):
            continue

        to_process.append(path)

    return to_process


def process_r3d_file(file_path):
    logger.info(f"Processing {file_path}")
    processor = VideoProcessor(
        root_path=file_path[:-4],
        export_path=file_path[:-4],
        video_modes=(VideoModes.REL_ACTIONS, VideoModes.RGB, VideoModes.DEPTH),
        skip_frames=5,
        fps=6,
    )
    try:
        processor.process()
        logger.info(f"Finished processing {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        traceback.print_exc()
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
    "--num_workers",
    type=int,
    required=True,
    help="Number of workers to use to process the r3d files.",
)
parser.add_argument(
    "--start_index",
    type=int,
    required=True,
    help="Index to start processing the r3d files from.",
)
parser.add_argument(
    "--end_index",
    type=int,
    required=True,
    help="Index to end processing the r3d files at.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    r3d_paths_file = args.r3d_paths_file
    num_workers = args.num_workers
    start_index = args.start_index
    end_index = args.end_index

    # Filter out the r3d files that have already been processed.
    r3d_paths = filter_r3d_files_to_process(r3d_paths_file)
    if end_index == -1:
        end_index = len(r3d_paths)
    logger.info(f"Number of video files left to process: {len(r3d_paths)}")
    if args.count_only:
        exit()
    r3d_paths = r3d_paths[start_index:end_index]

    # Process the r3d files.
    with Pool(num_workers) as p:
        p.map(
            partial(process_r3d_file),
            tqdm(r3d_paths, desc="Creating video files"),
        )

    logger.info("Finished processing all video files.")
