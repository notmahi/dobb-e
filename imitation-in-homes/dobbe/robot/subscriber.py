import time
from copy import copy
from warnings import warn as Warnings

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image_msg
from std_msgs.msg import Float32MultiArray, Int64

IMAGE_SUBSCRIBER_TOPIC = "/gopro_image"
DEPTH_SUBSCRIBER_TOPIC = "/gopro_depth"
SEQ_SUBSCRIBER_TOPIC = "/gopro_seq"


PING_TOPIC = "run_model_ping"


class ImagePolicySubscriber:
    def __init__(self):
        self.uid = -1
        self.prev_uid = -1
        self._registered_objects = []
        # Initializing a rosnode
        try:
            rospy.init_node("image_subscriber")
        except rospy.exceptions.ROSException:
            Warnings.warn("ROS node already initialized")
            pass

        self.bridge = CvBridge()

        # Getting images from the rostopic
        self.image = None
        # Subscriber for images
        rospy.Subscriber(
            IMAGE_SUBSCRIBER_TOPIC, Image_msg, self._callback_image, queue_size=1
        )

        rospy.Subscriber(
            DEPTH_SUBSCRIBER_TOPIC,
            Float32MultiArray,
            self._callback_depth,
            queue_size=1,
        )

        rospy.Subscriber(PING_TOPIC, Int64, self._callback_ping, queue_size=1)
        print("Image subscriber initialized")

    def register_for_uid(self, obj):
        print("Registered object")
        self._registered_objects.append(obj)

    def _callback_image(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def _callback_depth(self, data):
        try:
            shape = [dim.size for dim in data.layout.dim]
            np_depth = np.array(data.data).reshape(shape)
            self.depth = np_depth

        except CvBridgeError as e:
            print(e)

    def _callback_ping(self, data):
        self.uid = int(data.data)
        for obj in self._registered_objects:
            obj.uid = self.uid

    def _wait_for_image_ping(self):
        while self.image is None or self.uid == self.prev_uid:
            time.sleep(0.1)
            pass

    def get_image(self):
        self._wait_for_image_ping()
        self.prev_uid = copy(self.uid)
        return self.image

    def get_image_and_depth(self):
        self._wait_for_image_ping()
        self.prev_uid = copy(self.uid)
        return self.image, self.depth
