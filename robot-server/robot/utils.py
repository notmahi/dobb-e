import logging

import numpy as np
import PyKDL

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def euler_to_quat(r, p, y):
    sr, sp, sy = np.sin(r / 2.0), np.sin(p / 2.0), np.sin(y / 2.0)
    cr, cp, cy = np.cos(r / 2.0), np.cos(p / 2.0), np.cos(y / 2.0)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def urdf_joint_to_kdl_joint(jnt):
    kdl = PyKDL
    origin_frame = urdf_pose_to_kdl_frame(jnt.origin)
    if jnt.joint_type == "fixed":
        return kdl.Joint(jnt.name, getattr(kdl.Joint, "None"))
    axis = kdl.Vector(*jnt.axis)
    if jnt.joint_type == "revolute":
        return kdl.Joint(
            jnt.name, origin_frame.p, origin_frame.M * axis, kdl.Joint.RotAxis
        )
    if jnt.joint_type == "continuous":
        return kdl.Joint(
            jnt.name, origin_frame.p, origin_frame.M * axis, kdl.Joint.RotAxis
        )
    if jnt.joint_type == "prismatic":
        return kdl.Joint(
            jnt.name, origin_frame.p, origin_frame.M * axis, kdl.Joint.TransAxis
        )
    logging.warning("Unknown joint type: %s." % jnt.joint_type)
    return kdl.Joint(jnt.name, kdl.Joint.Fixed)


def urdf_pose_to_kdl_frame(pose):
    kdl = PyKDL
    pos = [0.0, 0.0, 0.0]
    rot = [0.0, 0.0, 0.0]
    if pose is not None:
        if pose.position is not None:
            pos = pose.position
        if pose.rotation is not None:
            rot = pose.rotation
    return kdl.Frame(kdl.Rotation.Quaternion(*euler_to_quat(*rot)), kdl.Vector(*pos))


def urdf_inertial_to_kdl_rbi(i):
    kdl = PyKDL
    origin = urdf_pose_to_kdl_frame(i.origin)
    rbi = kdl.RigidBodyInertia(
        i.mass,
        origin.p,
        kdl.RotationalInertia(
            i.inertia.ixx,
            i.inertia.iyy,
            i.inertia.izz,
            i.inertia.ixy,
            i.inertia.ixz,
            i.inertia.iyz,
        ),
    )
    return origin.M * rbi


# Returns a PyKDL.Tree generated from a urdf_parser_py.urdf.URDF object.
def kdl_tree_from_urdf_model(urdf):
    kdl = PyKDL
    root = urdf.get_root()
    tree = kdl.Tree(root)

    def add_children_to_tree(parent):
        if parent in urdf.child_map:
            for joint, child_name in urdf.child_map[parent]:
                child = urdf.link_map[child_name]
                if child.inertial is not None:
                    kdl_inert = urdf_inertial_to_kdl_rbi(child.inertial)
                else:
                    kdl_inert = kdl.RigidBodyInertia()
                kdl_jnt = urdf_joint_to_kdl_joint(urdf.joint_map[joint])
                kdl_origin = urdf_pose_to_kdl_frame(urdf.joint_map[joint].origin)
                kdl_sgm = kdl.Segment(child_name, kdl_jnt, kdl_origin, kdl_inert)
                tree.addSegment(kdl_sgm, parent)
                add_children_to_tree(child_name)

    add_children_to_tree(root)
    return tree


def get_color_logger():
    class CustomFormatter(logging.Formatter):
        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        FORMATS = {
            logging.DEBUG: grey + format + reset,
            logging.INFO: grey + format + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: bold_red + format + reset,
        }

        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    return ch
