import logging
import math
import os
import time
from typing import Union

import PyKDL
import rospy
import stretch_body.robot
from scipy.spatial.transform import Rotation as R
from urdf_parser_py.urdf import URDF

from .utils import kdl_tree_from_urdf_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OVERRIDE_STATES = {}


class HelloRobot:
    """
    This class is a wrapper for the Hello Robot Stretch RE1 robot.
    It provides a simple interface to control the robot and move the robot to a desired position.

    Parameters:
    -----------
    urdf_file: str
        The path to the urdf file of the robot. Default is "./urdf/stretch_nobase_raised.urdf"
    stretch_gripper_max: int | float
        The maximum opening on the gripper, using stepper motor units. The value should come
        out of Stretch robot calibration, and on our robot it is 47.
    stretch_gripper_min: int | float
        The minimum opening on the gripper, using stepper motor units. The value should come
        out of Stretch robot calibration, and on our robot it is 0. On a calibrated robot it
        should be around 0 (fingertips barely touching).
    gripper_threshold: int | float
        The gripper threshold at which the robot should close its gripper. Should be a value
        between [stretch_gripper_min, stretch_gripper_max].
    stretch_gripper_tight: int | float
        The value (in stepper units) to which the gripper should close when it is below the
        gripper_threshold. This is used to close the gripper tighter when the robot is
        grasping an object. Should be a value below stretch_gripper_min generally.
    sticky_gripper: bool
        If True, the gripper will stay closed after it has gripped an object. If False, the
        gripper may open after it has gripped an object.
    gripper_threshold_post_grasp: int | float
        The gripper threshold at which the robot should open its gripper after it has grasped
        an object. Should be a value between [stretch_gripper_min, stretch_gripper_max].
    """

    def __init__(
        self,
        urdf_file: str = "stretch_nobase_raised.urdf",
        stretch_gripper_max: Union[float, int] = 47,
        stretch_gripper_min: Union[float, int] = 0,
        gripper_threshold: Union[float, int] = 18,
        stretch_gripper_tight: Union[float, int] = -35,
        sticky_gripper: bool = True,
        gripper_threshold_post_grasp: Union[float, int] = 14.5,
    ):
        self.logger = logging.Logger("hello_robot")
        self.logger.setLevel(logging.INFO)

        self.STRETCH_GRIPPER_MAX = stretch_gripper_max
        self.STRETCH_GRIPPER_MIN = stretch_gripper_min
        self.STRETCH_GRIPPER_TIGHT = stretch_gripper_tight
        self._has_gripped = False
        self._sticky_gripper = sticky_gripper
        self.urdf_file = urdf_file

        self.urdf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "urdf", self.urdf_file
        )
        self.GRIPPER_THRESHOLD = gripper_threshold
        self.GRIPPER_THRESHOLD_POST_GRASP = (
            gripper_threshold_post_grasp or gripper_threshold
        )

        # Initializing ROS node
        self.joint_list = [
            "joint_fake",
            "joint_lift",
            "joint_arm_l3",
            "joint_arm_l2",
            "joint_arm_l1",
            "joint_arm_l0",
            "joint_wrist_yaw",
            "joint_wrist_pitch",
            "joint_wrist_roll",
        ]
        try:
            rospy.init_node("hello_robot_node")
        except:
            self.logger.warning("Node already initialized")

        self.robot = stretch_body.robot.Robot()
        self.startup()

        # Initializing the robot base position
        self.base_x = self.robot.base.status["x"]
        self.base_y = self.robot.base.status["y"]

        time.sleep(2)

        # Constraining the robots movement
        self.clamp = lambda n, minn, maxn: max(min(maxn, n), minn)

        # Joint dictionary for Kinematics
        self.setup_kdl()
        self.set_home_position()

    def startup(self, home=False):
        self.robot.startup()

        self.robot.arm.motor.enable_sync_mode()
        self.robot.base.left_wheel.enable_sync_mode()
        self.robot.base.right_wheel.enable_sync_mode()
        self.robot.lift.motor.enable_sync_mode()
        if home:
            self.home()

    def move_to_position(
        self,
        lift_pos=0.5,
        arm_pos=0.02,
        base_trans=0.0,
        wrist_yaw=0.0,
        wrist_pitch=0.0,
        wrist_roll=0.0,
        gripper_pos=None,
    ):
        self.CURRENT_STATE = (
            self.STRETCH_GRIPPER_MAX
            if gripper_pos is None
            else gripper_pos * (self.STRETCH_GRIPPER_MAX - self.STRETCH_GRIPPER_MIN)
            + self.STRETCH_GRIPPER_MIN
        )

        self.robot.lift.move_to(lift_pos)
        self.robot.end_of_arm.move_to("stretch_gripper", self.CURRENT_STATE)
        self.robot.push_command()

        while (
            self.robot.get_status()["arm"]["pos"] > arm_pos + 0.002
            or self.robot.get_status()["arm"]["pos"] < arm_pos - 0.002
        ):
            self.robot.arm.move_to(arm_pos)
            self.robot.push_command()

        self.robot.end_of_arm.move_to("wrist_yaw", wrist_yaw)
        PITCH_VAL = wrist_pitch
        self.robot.end_of_arm.move_to("wrist_pitch", PITCH_VAL)
        # NOTE: code below is to fix the pitch drift issue in current hello-robot.
        OVERRIDE_STATES["wrist_pitch"] = PITCH_VAL
        self.robot.end_of_arm.move_to("wrist_roll", wrist_roll)
        self.robot.base.translate_by(base_trans)
        self.logger.debug("moving to position 3")
        self.robot.push_command()
        self.logger.debug("moving to position 4")

    def set_home_position(
        self,
        lift=0.4,
        arm=0.02,
        base=0.0,
        wrist_yaw=0.0,
        wrist_pitch=0.0,
        wrist_roll=0.0,
        gripper=1.0,
    ):
        self.home_lift = lift
        self.home_arm = arm
        self.home_wrist_yaw = wrist_yaw
        self.home_wrist_pitch = wrist_pitch
        self.home_wrist_roll = wrist_roll
        self.home_gripper = gripper
        self.home_base = base

    def home(self):
        self.not_grasped = True
        self._has_gripped = False
        self.move_to_position(
            self.home_lift,
            self.home_arm,
            self.home_base,
            self.home_wrist_yaw,
            self.home_wrist_pitch,
            self.home_wrist_roll,
            self.home_gripper,
        )

    def setup_kdl(self):
        self.joints = {"joint_fake": 0}

        robot_model = URDF.from_xml_file(self.urdf_path)
        kdl_tree = kdl_tree_from_urdf_model(robot_model)
        self.arm_chain = kdl_tree.getChain("base_link", "link_raised_gripper")
        self.joint_array = PyKDL.JntArray(self.arm_chain.getNrOfJoints())

        # Forward kinematics
        self.fk_p_kdl = PyKDL.ChainFkSolverPos_recursive(self.arm_chain)
        # Inverse Kinematics
        self.ik_v_kdl = PyKDL.ChainIkSolverVel_pinv(self.arm_chain)
        self.ik_p_kdl = PyKDL.ChainIkSolverPos_NR(
            self.arm_chain, self.fk_p_kdl, self.ik_v_kdl
        )

    def updateJoints(self):
        origin_dist = math.sqrt(
            (self.base_y - self.robot.base.status["y"]) ** 2
            + (self.base_x - self.robot.base.status["x"]) ** 2
        )

        self.joints["joint_fake"] = origin_dist
        self.joints["joint_lift"] = self.robot.lift.status["pos"]
        armPos = self.robot.arm.status["pos"]
        self.joints["joint_arm_l3"] = armPos / 4.0
        self.joints["joint_arm_l2"] = armPos / 4.0
        self.joints["joint_arm_l1"] = armPos / 4.0
        self.joints["joint_arm_l0"] = armPos / 4.0

        self.joints["joint_wrist_yaw"] = self.robot.end_of_arm.status["wrist_yaw"][
            "pos"
        ]
        self.joints["joint_wrist_roll"] = self.robot.end_of_arm.status["wrist_roll"][
            "pos"
        ]
        self.joints["joint_wrist_pitch"] = OVERRIDE_STATES.get(
            "wrist_pitch", self.robot.end_of_arm.status["wrist_pitch"]["pos"]
        )
        self.joints["joint_stretch_gripper"] = self.robot.end_of_arm.status[
            "stretch_gripper"
        ]["pos"]

    def get_threshold(self):
        return (
            self.GRIPPER_THRESHOLD
            if not self._has_gripped
            else self.GRIPPER_THRESHOLD_POST_GRASP
        )

    def move_to_joints(self, joints, gripper):
        # Move the robot to a desired joint configuration
        # update the robot joints to the new values from 'joints'

        # You can manipulate the joints dictionary, for example to add a "wall" on the right side of the robot
        # joints['joint_fake'] = self.clamp(joints['joint_fake'], 0.0002, 0.20)

        self.robot.base.translate_by(
            joints["joint_fake"] - self.joints["joint_fake"], 5
        )
        self.robot.arm.move_to(
            joints["joint_arm_l3"]
            + joints["joint_arm_l2"]
            + joints["joint_arm_l1"]
            + joints["joint_arm_l0"]
        )

        self.robot.lift.move_to(joints["joint_lift"])

        # yaw, pitch, roll limits
        self.robot.end_of_arm.move_to(
            "wrist_yaw", self.clamp(joints["joint_wrist_yaw"], -0.4, 1.7)
        )
        self.robot.end_of_arm.move_to(
            "wrist_pitch", self.clamp(joints["joint_wrist_pitch"], -0.8, 0.2)
        )

        # NOTE: Code below is to fix the pitch drift issue in current hello-robot.
        OVERRIDE_STATES["wrist_pitch"] = joints["joint_wrist_pitch"]
        self.robot.end_of_arm.move_to(
            "wrist_roll", self.clamp(joints["joint_wrist_roll"], -1.57, 1.57)
        )
        self.logger.debug("Gripper state before update:", self.CURRENT_STATE)
        self.logger.debug("Gripper instruction from the policy:", gripper[0])

        # gripper[0] value ranges from 0 to 1, 0 being closed and 1 being open.
        # This code maps the gripper value to the range of the gripper joint
        self.CURRENT_STATE = (
            gripper[0] * (self.STRETCH_GRIPPER_MAX - self.STRETCH_GRIPPER_MIN)
            + self.STRETCH_GRIPPER_MIN
        )

        self.logger.debug("Gripper state after update:", self.CURRENT_STATE)
        self.robot.end_of_arm.move_to("stretch_gripper", self.CURRENT_STATE)
        self.logger.debug("Gripper state after update:", self.GRIPPER_THRESHOLD)

        # Map values below certain threshold to negative values to close the gripper much tighter
        if self.CURRENT_STATE < self.get_threshold() or (
            self._sticky_gripper and self._has_gripped
        ):
            self.robot.end_of_arm.move_to("stretch_gripper", self.STRETCH_GRIPPER_TIGHT)
            self._has_gripped = True
        else:
            self.robot.end_of_arm.move_to("stretch_gripper", self.STRETCH_GRIPPER_MAX)
        self.robot.push_command()

    def move_to_pose(self, translation_tensor, rotational_tensor, gripper):
        translation = [
            translation_tensor[0],
            translation_tensor[1],
            translation_tensor[2],
        ]
        rotation = rotational_tensor

        # move logic
        self.updateJoints()

        for joint_index in range(self.joint_array.rows()):
            self.joint_array[joint_index] = self.joints[self.joint_list[joint_index]]

        curr_pose = PyKDL.Frame()
        del_pose = PyKDL.Frame()
        self.fk_p_kdl.JntToCart(self.joint_array, curr_pose)

        rot_matrix = R.from_euler("xyz", rotation, degrees=False).as_matrix()

        del_rot = PyKDL.Rotation(
            PyKDL.Vector(rot_matrix[0][0], rot_matrix[1][0], rot_matrix[2][0]),
            PyKDL.Vector(rot_matrix[0][1], rot_matrix[1][1], rot_matrix[2][1]),
            PyKDL.Vector(rot_matrix[0][2], rot_matrix[1][2], rot_matrix[2][2]),
        )
        del_trans = PyKDL.Vector(translation[0], translation[1], translation[2])
        del_pose.M = del_rot
        del_pose.p = del_trans
        goal_pose_new = curr_pose * del_pose

        seed_array = PyKDL.JntArray(self.arm_chain.getNrOfJoints())
        self.ik_p_kdl.CartToJnt(seed_array, goal_pose_new, self.joint_array)

        ik_joints = {}

        for joint_index in range(self.joint_array.rows()):
            ik_joints[self.joint_list[joint_index]] = self.joint_array[joint_index]

        self.move_to_joints(ik_joints, gripper)

        self.robot.push_command()

        self.updateJoints()
        for joint_index in range(self.joint_array.rows()):
            self.joint_array[joint_index] = self.joints[self.joint_list[joint_index]]
