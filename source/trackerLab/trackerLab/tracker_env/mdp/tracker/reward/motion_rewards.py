import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.sensors import ContactSensor

from typing import TYPE_CHECKING

from trackerLab.motion_buffer import MotionBuffer
from trackerLab.utils.torch_utils import euler_from_quaternion

# Motion terms v1

"""
Penalize joint positions that deviate from the default one when no command.
l1 for all joint pos and mul with the z component of the projected gravity
"""
def punish_motion_l1_whb_dof_pos_subset(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    # asset: Articulation = env.scene[asset_cfg.name]
    # diff_angle = asset.data.joint_pos - env.motion_manager.loc_dof_pos
    diff = env.joint_subset - env.motion_manager.loc_dof_pos
    reward = torch.sum(torch.abs(diff), dim=1)
    return reward

def punish_motion_l1_lin_vel(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.root_lin_vel_b - env.motion_manager.loc_root_vel
    reward = torch.sum(torch.abs(diff), dim=1)
    return reward

def punish_motion_l1_ang_vel(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.root_ang_vel_b - env.motion_manager.loc_ang_vel
    reward = torch.sum(torch.abs(diff), dim=1)
    return reward

def punish_base_ang_vel(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """
    Penalize joint positions that deviate from the default one when no command.
    l1 for all joint pos and mul with the z component of the projected gravity
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff = asset.data.root_ang_vel_b[:, 2]
    reward = torch.abs(diff)
    return reward

def reward_motion_base_lin_vel_x(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), vel_scale = 1.0
):
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.root_lin_vel_b[:, 0] - env.motion_manager.loc_root_vel[:, 0] * vel_scale
    reward = torch.exp(- (diff ** 2))
    return reward

def reward_motion_base_ang_vel(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.root_ang_vel_b - env.motion_manager.loc_ang_vel
    reward = torch.exp(- torch.norm(diff, dim=1))
    return reward

def reward_tracking_demo_roll_pitch(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")  
):

    demo_roll, demo_pitch, demo_yaw = euler_from_quaternion(env.motion_manager.loc_root_rot)
    demo_roll_pitch = torch.stack((demo_roll, demo_pitch), dim=1)
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_from_quaternion(asset.data.root_quat_w.roll(shifts=-1, dims=-1))
    cur_roll_pitch = torch.stack((roll, pitch), dim=1)
    rew = torch.exp(- torch.norm(cur_roll_pitch - demo_roll_pitch, dim=1))
    return rew

def reward_tracking_demo_height(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    demo_height = env.motion_manager.loc_height
    cur_height = asset.data.root_pos_w[:, 2]
    rew = torch.exp(- torch.abs(cur_height - demo_height))
    return rew

# Motion Terms V2

def motion_ang_vel_z_world_exp(
    env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(asset.data.root_ang_vel_b[:, 2] - env.motion_manager.loc_ang_vel[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def motion_lin_vel_xy_yaw_frame_exp(
    env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), vel_scale: float = 1.0
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.motion_manager.loc_root_vel[:, :2] * vel_scale - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)

def motion_whb_dof_pos_subset_exp(
    env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of whb dof position commands in the gravity aligned robot frame using exponential kernel."""
    diff_angle = env.joint_subset - env.motion_manager.loc_dof_pos
    diff_angle = torch.sum(torch.abs(diff_angle), dim=1)
    reward = torch.exp(-diff_angle / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward

def motion_whb_dof_pos_subset_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of whb dof position commands in the gravity aligned robot frame using L2 loss."""
    diff_angle = env.joint_subset - env.motion_manager.loc_dof_pos
    diff_angle = torch.sum(torch.abs(diff_angle), dim=1)
    return torch.square(diff_angle)

def motion_roll_pitch_world_exp(
    env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")  
):

    demo_roll, demo_pitch, demo_yaw = euler_from_quaternion(env.motion_manager.loc_root_rot)
    demo_roll_pitch = torch.stack((demo_roll, demo_pitch), dim=1)
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_from_quaternion(asset.data.root_quat_w.roll(shifts=-1, dims=-1))
    cur_roll_pitch = torch.stack((roll, pitch), dim=1)
    rew = torch.exp(- torch.norm(cur_roll_pitch - demo_roll_pitch, dim=1) / std**2)
    return rew
