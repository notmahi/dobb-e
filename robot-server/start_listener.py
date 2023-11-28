import argparse

from robot import HelloRobot, listener

PING_TOPIC_NAME = "run_model_ping"


parser = argparse.ArgumentParser()
parser.add_argument("--lift", type=float, default=0.5, help="position of robot lift")
parser.add_argument("--arm", type=str, default=0.02, help="arm position")
parser.add_argument("--base", type=float, default=0.0, help="position of robot base")
parser.add_argument(
    "--yaw", type=float, default=0.0, help="position of robot wrist yaw"
)
parser.add_argument(
    "--pitch", type=float, default=0.0, help="position of robot wrist pitch"
)
parser.add_argument(
    "--roll", type=float, default=0.0, help="position of robot wrist roll"
)
parser.add_argument(
    "--gripper", type=float, default=1.0, help="position of robot gripper"
)

args = parser.parse_args()
params = vars(args)

if __name__ == "__main__":
    hello_robot = HelloRobot()
    hello_robot.set_home_position(
        params["lift"],
        params["arm"],
        params["base"],
        params["yaw"],
        params["pitch"],
        params["roll"],
        params["gripper"],
    )

    listener(hello_robot)
