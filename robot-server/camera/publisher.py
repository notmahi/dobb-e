import logging

import cv2
import numpy as np
import rospy
from imgcat import imgcat

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

from cv_bridge import CvBridge, CvBridgeError
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int32, MultiArrayDimension

from .demo import R3DApp

NODE_NAME = "gopro_node"
IMAGE_PUBLISHER_NAME = "/gopro_image"
DEPTH_PUBLISHER_NAME = "/gopro_depth"
SEQ_PUBLISHER_NAME = "/gopro_seq"


# @converts_to_message(Float32MultiArray))
def convert_numpy_array_to_float32_multi_array(matrix):
    # Create a Float64MultiArray object
    data_to_send = Float32MultiArray()

    # Set the layout parameters
    data_to_send.layout.dim.append(MultiArrayDimension())
    data_to_send.layout.dim[0].label = "rows"
    data_to_send.layout.dim[0].size = len(matrix)
    data_to_send.layout.dim[0].stride = len(matrix) * len(matrix[0])

    data_to_send.layout.dim.append(MultiArrayDimension())
    data_to_send.layout.dim[1].label = "columns"
    data_to_send.layout.dim[1].size = len(matrix[0])
    data_to_send.layout.dim[1].stride = len(matrix[0])

    # Flatten the matrix into a list
    data_to_send.data = matrix.flatten().tolist()

    return data_to_send


class ImagePublisher(object):
    def __init__(self, app: R3DApp, rate: int = 10):
        # Initializing camera
        self.app = app
        self.rate = rate
        # Initializing ROS node
        try:
            rospy.init_node(NODE_NAME)
        except rospy.exceptions.ROSException as e:
            logging.warning(e)
            logging.warning("ROS node already initialized")
        self.bridge = CvBridge()
        self.image_publisher = rospy.Publisher(
            IMAGE_PUBLISHER_NAME, Image, queue_size=1
        )
        self.depth_publisher = rospy.Publisher(
            DEPTH_PUBLISHER_NAME, Float32MultiArray, queue_size=1
        )
        self.seq_publisher = rospy.Publisher(SEQ_PUBLISHER_NAME, Int32, queue_size=1)
        self._seq = 0

    def publish_image_from_camera(self):
        rate = rospy.Rate(self.rate)
        while True:
            image, depth, pose = self.app.start_process_image()
            image = np.moveaxis(image, [0], [1])[..., ::-1, ::-1]
            image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            depth = np.ascontiguousarray(np.rot90(depth, -1)).astype(np.float64)

            # Creating a CvBridge and publishing the data to the rostopic
            try:
                self.image_message = self.bridge.cv2_to_imgmsg(image, "bgr8")
            except CvBridgeError as e:
                logging.warning(e)

            depth_data = convert_numpy_array_to_float32_multi_array(depth)
            self.image_publisher.publish(self.image_message)
            self.depth_publisher.publish(depth_data)
            self.seq_publisher.publish(Int32(self._seq))
            self._seq += 1

            # Stopping the camera
            if cv2.waitKey(1) == 27:
                break
            if self.app.stream_stopped:
                logging.info("Breaking")
                break

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = R3DApp()
    app.connect_to_device(dev_idx=0)
    logging.info("Connected")
    camera_publisher = ImagePublisher(app)
    camera_publisher.publish_image_from_camera()
