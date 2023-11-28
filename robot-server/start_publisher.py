from camera import R3DApp, ImagePublisher
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    app = R3DApp()
    app.connect_to_device(dev_idx=0)
    logging.info("Connected")
    camera_publisher = ImagePublisher(app)
    camera_publisher.publish_image_from_camera()
