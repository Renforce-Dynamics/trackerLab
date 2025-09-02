import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.sensors import ContactSensor

from typing import TYPE_CHECKING

from trackerLab.motion_buffer import MotionBuffer
from trackerLab.utils.torch_utils import euler_from_quaternion

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

def foot_clearance_reward(
    env, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def energy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)

def reward_alive(env):
    """
    Will only return true, for making the policy live as long as possible.
    """
    return torch.ones((env.num_envs, ), device=env.device)

"""
reward motion terms will using the motion manager self cached data which is better for .
"""

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

def motion_whb_dof_pos_subset_exp(env, std: float) -> torch.Tensor:
    """Reward tracking of whb dof position commands in the gravity aligned robot frame using exponential kernel."""
    diff_angle = env.joint_subset - env.motion_manager.loc_dof_pos
    diff_angle = torch.sum(torch.abs(diff_angle), dim=1)
    return torch.exp(-diff_angle / std**2)

def motion_whb_dof_pos_subset_l2(env) -> torch.Tensor:
    """Reward tracking of whb dof position commands in the gravity aligned robot frame using L2 loss."""
    diff_angle = env.joint_subset - env.motion_manager.loc_dof_pos
    diff_angle = torch.sum(torch.abs(diff_angle), dim=1)
    return torch.square(diff_angle)


# 

def body_orientation_l2(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)

def body_force(
    env, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward

# 

def feet_too_near(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_stumble(
    env, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )

