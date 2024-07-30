import os
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T


def schedule_init(controller, max_h=0.055, max_base=0.08):
    def schedule(sched_no):
        if sched_no <= 2:
            base = -max_base
            if sched_no == 1:
                h = controller.h + max_h * 2 / 3
            else:
                h = controller.h - max_h * 2 / 3
        elif sched_no <= 5:
            base = -max_base / 3
            if sched_no == 3:
                h = controller.h + max_h
            elif sched_no == 4:
                h = controller.h + max_h / 3.5
            else:
                h = controller.h - max_h
        elif sched_no <= 8:
            base = max_base / 3
            if sched_no == 6:
                h = controller.h + max_h
            elif sched_no == 7:
                h = controller.h - max_h / 3.5
            else:
                h = controller.h - max_h
        elif sched_no <= 10:
            base = max_base
            if sched_no == 9:
                h = controller.h + max_h * 2 / 3
            else:
                h = controller.h - max_h * 2 / 3
        return base, h

    return schedule


class AsyncImageActionSaver:
    def __init__(self, folder=None):
        if folder is None:
            # set folder to current directory
            folder = Path.cwd()

        # create folder with timestamp as name
        self.folder = folder
        self.save_dir = None
        self.count = 0
        self.img_threads = []
        self.action_threads = []
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def save_image_async(self, image_data, file_path):
        cv2.imwrite(str(file_path), image_data)

    def save_action_async(self, action, file_path, prev_thread=None):
        # load action list file if it already exists and append to the list
        # else create a new file and write the action list
        if prev_thread is not None:
            prev_thread.join()
        if os.path.exists(file_path):
            action_list = np.load(file_path)
            action_list = np.append(action_list, action)
            np.save(file_path, action_list)
        else:
            action_list = np.array(action)
            np.save(file_path, action_list)

    def create_save_dir_if_not_exists(self):
        if self.save_dir is None:
            # create folder with timestamp as name
            self.save_dir = self.folder / Path(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
            os.makedirs(self.save_dir)
            self.save_nbhrs_dir = self.save_dir / Path("neighbors")
            os.makedirs(self.save_nbhrs_dir)
            self.save_depths_dir = self.save_dir / Path("depths")
            os.makedirs(self.save_depths_dir)

    def save_image(self, image, nbhr=-1):
        self.create_save_dir_if_not_exists()
        # save cv2 image with name as count
        if nbhr == -1:
            file_path = self.save_dir / Path(str(self.count) + ".jpg")
            self.count += 1
        else:
            file_path = self.save_nbhrs_dir / Path(
                str(self.count - 1) + "_nbhr_" + str(nbhr + 1) + ".jpg"
            )
        thread = Thread(target=self.save_image_async, args=(image, file_path))
        self.img_threads.append(thread)
        thread.start()

    def save_action(self, action_list):
        # create thread to save action list, but make sure the treads are executed in order
        self.create_save_dir_if_not_exists()
        file_path = self.save_dir / Path("action_list.npy")
        prev_thread = None if len(self.action_threads) == 0 else self.action_threads[-1]
        thread = Thread(
            target=self.save_action_async,
            args=(action_list, file_path, prev_thread),
        )
        self.action_threads.append(thread)
        thread.start()

    def finish(self):
        for thread in self.img_threads:
            thread.join()
        for thread in self.action_threads:
            thread.join()
        self.img_threads = []
        self.action_threads = []

        self.count = 0
        self.save_dir = None


class AsyncImageDepthActionSaver(AsyncImageActionSaver):
    def __init__(self, folder=None):
        super().__init__(folder)
        self.depth_threads = []

    def save_depth_async(self, depth, file_path):
        np.save(str(file_path), depth)

    def save_depth(self, depth, nbhr=-1):
        self.create_save_dir_if_not_exists()

        file_path = self.save_depths_dir / Path(str(self.count - 1) + ".npy")

        thread = Thread(target=self.save_depth_async, args=(depth, file_path))
        self.depth_threads.append(thread)
        thread.start()

    def finish(self):
        super().finish()
        for thread in self.depth_threads:
            thread.join()
        self.depth_threads = []


class ImageActionBufferManager:
    def __init__(self, buffer_size=4, async_saver=None, act_dim=7):
        self.buffer_size = buffer_size
        self.act_dim = act_dim
        self.image_buffer = deque(maxlen=self.buffer_size)
        self.action_buffer = deque(maxlen=self.buffer_size - 1)
        self.action_list = []
        self.async_saver = async_saver
        self.to_tensor = T.ToTensor()

    def add_image(self, img):
        if self.async_saver is not None:
            self.async_saver.save_image(img)
        tensor = self.img_process(img)
        self.image_buffer.append(tensor)

    def add_action(self, action):
        if self.async_saver is not None:
            self.async_saver.save_action(action)
        self.action_buffer.append(action)
        self.action_list.append(action)

    def get_input_tensor_sequence(self):
        img_seq = torch.stack([img for img in self.image_buffer])
        base_act = torch.Tensor([-0.0] * self.act_dim)
        act_seq = torch.stack([act for act in self.action_buffer] + [base_act])
        return img_seq, act_seq

    def img_process(self, img):
        if type(img) is np.ndarray:
            # convert cv2 image to PIL image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        img = self.to_tensor(img)
        return img


class ImageDepthActionBufferManager(ImageActionBufferManager):
    def __init__(
        self,
        buffer_size=4,
        async_saver=None,
        depth_cfg=dict(log2_scale=1.0, log2_x_shift=0.0, log2_y_shift=0.0, n_bins=10),
        act_dim=7,
    ):
        super().__init__(buffer_size, async_saver, act_dim)

        self.depth_buffer = deque(maxlen=self.buffer_size)

        self.depth_transform = T.ToTensor()

        if depth_cfg is None:
            depth_cfg = dict(
                log2_scale=1.0, log2_x_shift=0.0, log2_y_shift=0.0, n_bins=10
            )
        log2_scale = depth_cfg["log2_scale"]
        log2_x_shift = depth_cfg["log2_x_shift"]
        log2_y_shift = depth_cfg["log2_y_shift"]
        n_bins = depth_cfg["n_bins"]
        self.bin_pixels = lambda x: (
            (
                log2_scale * torch.log2(x.clamp(min=log2_x_shift) - log2_x_shift)
                + log2_y_shift
            ).floor()
        ).clamp(0, n_bins - 1)

    def add_depth(self, depth):
        if self.async_saver is not None:
            self.async_saver.save_depth(depth)
        tensor = self.depth_process(depth)
        self.depth_buffer.append(tensor)

    def get_input_tensor_sequence(self):
        img_seq = torch.stack([img for img in self.image_buffer])
        depth_seq = torch.stack([depth for depth in self.depth_buffer])
        base_act = torch.Tensor([-1] * self.act_dim)
        act_seq = torch.stack([act for act in self.action_buffer] + [base_act])
        return img_seq, depth_seq, act_seq

    def depth_process(self, depth):
        depth = self.depth_transform(depth)
        return depth
