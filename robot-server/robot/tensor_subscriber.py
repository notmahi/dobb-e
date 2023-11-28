import numpy as np
import rospy
from std_msgs.msg import Float64MultiArray

TRANSLATIONAL_PUBLISHER_TOPIC = "/translation_tensor"
ROTATIONAL_PUBLISHER_TOPIC = "/rotational_tensor"
GRIPPER_PUBLISHER_TOPIC = "/gripper_tensor"
HOME_PUBLISHER_TOPIC = "/home_tensor"
HOME_PARAMS_TOPIC = "/home_params"


class TensorSubscriber(object):
    def __init__(self):
        try:
            rospy.init_node("tensor_receiver")
        except:
            pass

        rospy.Subscriber(
            TRANSLATIONAL_PUBLISHER_TOPIC,
            Float64MultiArray,
            self._callback_translation_data,
            queue_size=1,
        )
        rospy.Subscriber(
            ROTATIONAL_PUBLISHER_TOPIC,
            Float64MultiArray,
            self._callback_rotation_data,
            queue_size=1,
        )
        rospy.Subscriber(
            GRIPPER_PUBLISHER_TOPIC,
            Float64MultiArray,
            self._callback_gripper_data,
            queue_size=1,
        )
        rospy.Subscriber(
            HOME_PUBLISHER_TOPIC,
            Float64MultiArray,
            self._callback_home_data,
            queue_size=1,
        )
        rospy.Subscriber(
            HOME_PARAMS_TOPIC,
            Float64MultiArray,
            self._callback_home_params,
            queue_size=1,
        )

        self.translation_tensor = None
        self.rotational_tensor = None
        self.gripper_tensor = None
        self.home_tensor = None
        self.tr_data_offset = None
        self.rot_data_offset = None
        self.gr_data_offset = None
        self.home_data_offset = None
        self.home_params_offset = None

    def _callback_translation_data(self, data):
        self.translation_tensor = list(data.data)
        self.tr_data_offset = data.layout.data_offset

    def _callback_rotation_data(self, data):
        self.rotational_tensor = list(data.data)
        self.rot_data_offset = data.layout.data_offset

    def _callback_gripper_data(self, data):
        self.gripper_tensor = list(data.data)
        self.gr_data_offset = data.layout.data_offset

    def _callback_home_data(self, data):
        self.home_tensor = list(data.data)
        self.home_data_offset = data.layout.data_offset

    def _callback_home_params(self, data):
        self.home_params = list(data.data)
        self.home_params_offset = data.layout.data_offset
