import os
import numpy as np
from trackerLab import TRACKERLAB_ASSETS_DIR
from etils import epath
import torch
import json

SIM2SIMLIB_REPO_DIR = os.path.dirname(os.path.abspath(__file__))

def get_assets(robot_name) -> list[str]:
    assets = []
    assets_dir = os.path.join(TRACKERLAB_ASSETS_DIR, "data", robot_name)
    xml_dir = os.path.join(assets_dir, "mjcf")
    assets = epath.Path(xml_dir).glob("*.xml")
    return assets

@torch.jit.script
def slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta
    
    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q

def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

# def get_lab_joint_names(robot_type: str) -> list:
#     """
#     Load the lab joint names for a specific robot type.
    
#     Args:
#         robot_type (str): The type of the robot.
        
#     Returns:
#         list: A list of lab joint names.
#     """
#     file_path = os.path.join(SIM2SIMLIB_REPO_DIR, "configs", f"{robot_type}.json")
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     return data.get("lab_joint_names")


def get_mujoco_joint_names(robot_type: str) -> list:
    """
    Load the MuJoCo joint names for a specific robot type.

    Args:
        robot_type (str): The type of the robot.

    Returns:
        list: A list of MuJoCo joint names.
    """
    file_path = os.path.join(SIM2SIMLIB_REPO_DIR, "configs", f"{robot_type}.json")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get("mujoco_joint_names")