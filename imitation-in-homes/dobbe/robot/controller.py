import time

import cv2
import numpy as np
import rospy
import torch
from scipy.spatial.transform import Rotation as R

import wandb
from utils.action_transforms import *

from .publisher import ImitiationPolicyPublisher
from .subscriber import ImagePolicySubscriber
from .utils import (
    AsyncImageActionSaver,
    AsyncImageDepthActionSaver,
    ImageActionBufferManager,
    ImageDepthActionBufferManager,
    schedule_init,
)


def get_home_param(h=0.5, y=0.02, x=0.0, yaw=0.0, pitch=0.0, roll=0.0, gripper=1.0):
    """
    Returns a list of home parameters
    """
    return [h, y, x, yaw, pitch, roll, gripper]


schedule = None


class Controller:
    def __init__(self, cfg=None, frequency: int = 10):
        global schedule
        publisher = ImitiationPolicyPublisher()
        subscriber = ImagePolicySubscriber()

        self.publisher = publisher
        self.subscriber = subscriber
        self.frequency = frequency

        self.subscriber.register_for_uid(self.publisher)
        self.subscriber.register_for_uid(self)

        self.cfg = cfg
        self.use_depth = cfg["use_depth"]

        if not self.use_depth:
            self.async_saver = AsyncImageActionSaver(cfg["image_save_dir"])
        else:
            self.async_saver = AsyncImageDepthActionSaver(cfg["image_save_dir"])

        self.image_action_buffer_manager = self.create_buffer_manager()

        self.device = cfg["device"]
        schedule = schedule_init(
            self,
            max_h=cfg["robot_params"]["max_h"],
            max_base=cfg["robot_params"]["max_base"],
        )

        self.run_n = -1
        self.step_n = 0
        self.schedul_no = -1
        self.h = cfg["robot_params"]["h"]

        self.abs_gripper = cfg["robot_params"]["abs_gripper"]
        self.gripper = 1.0
        self.rot_unit = cfg["robot_params"]["rot_unit"]

    def setup_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.model.eval()

    def create_buffer_manager(self):
        if self.use_depth:
            return ImageDepthActionBufferManager(
                self.cfg["image_buffer_size"],
                self.async_saver,
                self.cfg["dataset"]["train"]["config"].get("depth_cfg"),
            )
        else:
            return ImageActionBufferManager(
                self.cfg["image_buffer_size"], self.async_saver
            )

    def action_tensor_to_matrix(self, action_tensor):
        affine = np.eye(4)
        if self.rot_unit == "euler":
            r = R.from_euler("xyz", action_tensor[3:6], degrees=False)
        elif self.rot_unit == "axis":
            r = R.from_rotvec(action_tensor[3:6])
        else:
            raise NotImplementedError
        affine[:3, :3] = r.as_matrix()
        affine[:3, -1] = action_tensor[:3]

        return affine

    def matrix_to_action_tensor(self, matrix):
        r = R.from_matrix(matrix[:3, :3])
        action_tensor = np.concatenate(
            (matrix[:3, -1], r.as_euler("xyz", degrees=False))
        )
        return action_tensor

    def cam_to_robot_frame(self, matrix):
        return invert_permutation_transform(matrix)

    def _update_log_keys(self, logs):
        new_logs = {}
        for k in logs.keys():
            new_logs[k + "_" + str(self.run_n)] = logs[k]

        return new_logs

    def _run_policy(self, run_for=1):
        rate = rospy.Rate(10)
        while run_for > 0:
            rate.sleep()

            cv2_img = self.subscriber.get_image()
            self.image_action_buffer_manager.add_image(cv2_img)

            with torch.no_grad():
                input_tensor_sequence = (
                    self.image_action_buffer_manager.get_input_tensor_sequence()
                )

                input_tensor_sequence = (
                    input_tensor_sequence[0].to(self.device).unsqueeze(0),
                    input_tensor_sequence[1].to(self.device).unsqueeze(0),
                )

                action_tensor, logs = self.model.step(input_tensor_sequence)
                if "indices" in logs:
                    indices = logs["indices"].squeeze()
                    for nbhr, idx in enumerate(indices):
                        img = self.model.train_dataset[idx]
                        img = (
                            (img[0][0]).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        )
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        self.async_saver.save_image(img, nbhr=nbhr)
                action_tensor = action_tensor.squeeze(0).cpu()
                self.image_action_buffer_manager.add_action(action_tensor)
                action_tensor = action_tensor.squeeze().numpy()

            action_matrix = self.action_tensor_to_matrix(action_tensor)
            action_robot_matrix = self.cam_to_robot_frame(action_matrix)
            action_robot = self.matrix_to_action_tensor(action_robot_matrix)

            gripper = action_tensor[-1]

            if not self.abs_gripper:
                self.gripper = self.gripper + gripper
                gripper = self.gripper
            self.publisher.publish_action(action_robot, gripper)

            wandb.log(self._update_log_keys(logs), step=self.step_n)
            run_for -= 1
            self.step_n += 1

    def _run_policy_depth(self, run_for=1):
        rate = rospy.Rate(10)
        while run_for > 0:
            rate.sleep()

            cv2_img, np_depth = self.subscriber.get_image_and_depth()
            self.image_action_buffer_manager.add_image(cv2_img)
            self.image_action_buffer_manager.add_depth(np_depth)

            with torch.no_grad():
                input_tensor_sequence = (
                    self.image_action_buffer_manager.get_input_tensor_sequence()
                )

                input_tensor_sequence = (
                    input_tensor_sequence[0].to(self.device).unsqueeze(0),
                    input_tensor_sequence[1].to(self.device).unsqueeze(0),
                    input_tensor_sequence[2].to(self.device).unsqueeze(0),
                )

                action_tensor, logs = self.model.step(input_tensor_sequence)
                action_tensor = action_tensor.squeeze(0).cpu()
                self.image_action_buffer_manager.add_action(action_tensor)
                action_tensor = action_tensor.squeeze().numpy()

            action_matrix = self.action_tensor_to_matrix(action_tensor)
            action_robot_matrix = self.cam_to_robot_frame(action_matrix)
            action_robot = self.matrix_to_action_tensor(action_robot_matrix)

            gripper = action_tensor[-1]

            if not self.abs_gripper:
                self.gripper = self.gripper + gripper
                gripper = self.gripper
            self.publisher.publish_action(action_robot, gripper)

            wandb.log(self._update_log_keys(logs), step=self.step_n)
            run_for -= 1
            self.step_n += 1

    def _run(self, run_for=1):
        if not self.use_depth:
            self._run_policy(run_for=run_for)
        else:
            self._run_policy_depth(run_for=run_for)

    def reset_experiment(self):
        self.async_saver.finish()
        self.run_n += 1
        self.step_n = 0
        self.gripper = 1.0
        self.model.reset()
        self.image_action_buffer_manager = self.create_buffer_manager()

    def _process_instruction(self, instruction):
        global schedule
        if instruction.lower() == "h":
            self.publisher.publish([1], "home_publisher")
            self.reset_experiment()

        elif instruction.lower() == "r":
            h = input("Enter height:")
            self.h = float(h)
            self.publisher.publish(get_home_param(h=self.h), "home_params_publisher")

        elif instruction.lower() == "s":
            sched_no = input("Enter schedule number:")
            base, h = schedule(int(sched_no))
            print(h, base)
            self.publisher.publish(get_home_param(h=h, x=base), "home_params_publisher")
            self.schedul_no = int(sched_no)

        elif instruction.lower() == "n":
            self.schedul_no += 1
            base, h = schedule(self.schedul_no)
            print(h, base)
            self.publisher.publish(get_home_param(h=h, x=base), "home_params_publisher")
        elif len(instruction) == 0:
            self.run_for = 1
            self._run(self.run_for)
        elif instruction.isdigit():
            self.run_for = int(instruction)
            self._run(self.run_for)
        elif instruction.lower() == "q":
            self.async_saver.finish()
        else:
            # raise warning
            print("Invalid instruction")

    def _wait_for_publisher_subscriber(self):
        print("Waiting for publisher and subscriber to be ready..")
        while hasattr(self.publisher, "uid") is False:
            time.sleep(1)
        print("Publisher and subscriber ready..")

    def run(self):
        self._wait_for_publisher_subscriber()
        self.publisher.publish(get_home_param(h=self.h), "home_params_publisher")
        rate = rospy.Rate(10)

        while True:
            rate.sleep()
            instruction = input("Enter instruction:")
            if instruction.lower() == "q":
                instruction = self._process_instruction(instruction)
                break
            instruction = self._process_instruction(instruction)
