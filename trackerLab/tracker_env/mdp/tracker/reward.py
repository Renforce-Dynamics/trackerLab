import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import SceneEntityCfg
from trackerLab.motion_buffer import MotionBuffer
from trackerLab.utils.torch_utils import euler_from_quaternion

from isaaclab.sensors import ContactSensor

from typing import TYPE_CHECKING

# from trackerLab.tracker_env.manager_based_tracker_env import ManagerBasedTrackerEnv

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def reward_alive(env):
    """
    Will only return true, for making the policy live as long as possible.
    """
    return torch.ones((env.num_envs, ), device=env.device)

"""
reward motion terms will using the motion manager self cached data which is better for .
"""

"""
Penalize joint positions that deviate from the default one when no command.
l1 for all joint pos and mul with the z component of the projected gravity
"""
def reward_motion_l1_whb_dof_pos_subset(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    # asset: Articulation = env.scene[asset_cfg.name]
    # diff_angle = asset.data.joint_pos - env.motion_manager.loc_dof_pos
    diff_angle = env.joint_subset - env.motion_manager.loc_dof_pos
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    return reward

def reward_motion_exp_whb_dof_pos_subset(
    env, factor = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    diff_angle = env.joint_subset - env.motion_manager.loc_dof_pos
    diff_angle = torch.sum(torch.abs(diff_angle), dim=1)
    return torch.exp(-diff_angle * factor)

def reward_motion_l1_whb_dof_pos(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos - env.motion_manager.loc_dof_pos
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    return reward

def reward_motion_exp_whb_dof_pos(
    env, factor = 0.5, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: Articulation = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos - env.motion_manager.loc_dof_pos
    diff_angle = torch.sum(torch.abs(diff_angle), dim=1)
    return torch.exp(-diff_angle * factor)

def reward_motion_base_lin_vel(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), vel_scale = 1.0
):
    asset: Articulation = env.scene[asset_cfg.name]
    diff = asset.data.root_lin_vel_b - env.motion_manager.loc_root_vel * vel_scale
    reward = torch.exp(- torch.norm(diff, dim=1))
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
    asset: Articulation = env.scene[asset_cfg.name]
    demo_roll_pitch = env.motion_manager.curr_demo_roll_pitch
    roll, pitch, yaw = euler_from_quaternion(asset.data.root_quat_w)
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
    reward = torch.exp(- diff ** 2)
    return reward