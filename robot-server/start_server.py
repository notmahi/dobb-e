import argparse
import logging
import multiprocessing
import signal
import sys
import time
from multiprocessing import Process
from typing import Dict, Optional

import cv2

from camera import ImagePublisher, R3DApp
from robot import HelloRobot, Listener

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def camera_process(app: R3DApp):
    camera_publisher = ImagePublisher(app)
    camera_publisher.publish_image_from_camera()


def robot_process(hello_robot_params: Optional[Dict]):
    logging.info("Robot process started")
    if hello_robot_params is None:
        hello_robot = HelloRobot()
    else:
        hello_robot = HelloRobot(**hello_robot_params)
    listner = Listener(hello_robot)
    listner.start()


def stream_manager():
    app = R3DApp()
    while app.stream_stopped:
        try:
            app.connect_to_device(dev_idx=0)
        except RuntimeError as e:
            logging.warning(e)
            logging.warning(
                "Retrying to connect to device with id 0, make sure the device is connected and id is correct..."
            )
            time.sleep(2)

    try:
        camera_process(app)
    except cv2.error as e:
        logging.warning(e)
        logging.warning(
            "The device was connected but the stream didn't start, trying to reconnect..."
        )
        time.sleep(2)
        stream_manager()

    while not app.stream_stopped:
        time.sleep(2)

    stream_manager()


# Default values for the robot controller
GRIPPER_MAX = 47.0
GRIPPER_MIN = 0.0
GRIPPER_THRESHOLD = 17.0
GRIPPER_TIGHT = -25.0
GRIPPER_POST_GRASP_THRESHOLD = 14.5

parser = argparse.ArgumentParser(
    prog="hello-stretch-server",
    description=(
        "Communicates with the robot policy to communicate the observation to the robot, "
        "and executes the action returned by the policy on the robot.\n"
        "Note that while the gripper arguments here are specified in stepper units, the policy "
        "must output gripper values in a (0, 1) range."
    ),
    epilog="Contact us in https://dobb-e.com for more information.",
)

parser.add_argument(
    "-g",
    "--gripper_threshold",
    type=float,
    help=(
        "Gripper thresholding value in stepper units, between gripper_min (default 0) and gripper_max (default 47)"
        "Above this value, the gripper will remain fully open. Below this value, the gripper will close to gripper_tight."
    ),
    default=GRIPPER_THRESHOLD,
)

parser.add_argument(
    "-s",
    "--unsticky_gripper",
    action="store_false",
    help=(
        "Turn off sticky gripper; without it gripper will open and close every time threshold is passed. "
        "With sticky grasp, grasping an object means robot won't open grasp again afterwards."
        "If you turn this on, you may want to adjust gripper_threshold_post_grasp as well."
    ),
)
parser.add_argument(
    "-m",
    "--gripper_max",
    type=float,
    help=(
        "Maximum gripper opening value in stepper units"
        "Defaults to 47, set it to higher to open the grip more in the beginning."
    ),
    default=GRIPPER_MAX,
)
parser.add_argument(
    "-n",
    "--gripper_min",
    type=float,
    help=(
        "Minimum gripper opening value in stepper units. "
        "If your robot is calibrated it should be 0 (default) and shouldn't be changed."
    ),
    default=GRIPPER_MIN,
)
parser.add_argument(
    "-t",
    "--gripper_tight",
    type=float,
    help=(
        "How tightly to close the gripper once it passes the threshold, value in stepper units. "
        "Defaults to -25, set it > -25 to grip looser and < -25 to grip tighter."
    ),
    default=GRIPPER_TIGHT,
)
parser.add_argument(
    "-p",
    "--gripper_threshold_post_grasp",
    type=float,
    help=(
        "Gripper threshold value in stepper units once the grasp has been executed. "
        "Above this value, the gripper will reopen, below this value, the gripper will remain closed. "
        "Sticky gripper must be turned off for this to take effect."
    ),
    default=GRIPPER_POST_GRASP_THRESHOLD,
)


if __name__ == "__main__":
    args = parser.parse_args()
    params = vars(args)
    logging.debug(params)
    hello_robot_params = dict(
        stretch_gripper_max=params["gripper_max"],
        stretch_gripper_min=params["gripper_min"],
        stretch_gripper_tight=params["gripper_tight"],
        gripper_threshold=params["gripper_threshold"],
        sticky_gripper=(not params["unsticky_gripper"]),
        gripper_threshold_post_grasp=params["gripper_threshold_post_grasp"],
    )
    logger = multiprocessing.log_to_stderr(logging.INFO)
    robot_thread = Process(target=robot_process, args=(hello_robot_params,))
    robot_thread.start()

    def signal_handler(sig, frame):
        robot_thread.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    stream_manager()
