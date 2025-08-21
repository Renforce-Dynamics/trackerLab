# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject

from .utils import quaternion_to_tangent_and_normal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return env.motion_manager.get_subset_real(asset.data.joint_pos[:, asset_cfg.joint_ids])


def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return env.motion_manager.get_subset_real(
        asset.data.joint_vel[:, asset_cfg.joint_ids]
    )

def body_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    
    asset: RigidObject = env.scene[asset_cfg.name]
    ref_body_index = asset.data.body_names.index("base_link")
    return asset.data.body_pos_w[:, ref_body_index][:, 2:3]

def body_quat_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    ref_body_index = asset.data.body_names.index("base_link")
    return quaternion_to_tangent_and_normal(asset.data.body_quat_w[:, ref_body_index])

def body_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    ref_body_index = asset.data.body_names.index("base_link")
    return asset.data.body_lin_vel_w[:, ref_body_index]

def body_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    ref_body_index = asset.data.body_names.index("base_link")
    return asset.data.body_ang_vel_w[:, ref_body_index]

def key_body_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    # TODO: 改掉
    key_body_names = ["l_ankle_pitch_link", "r_ankle_pitch_link", "l_claw_link", "r_claw_link"]
    asset: RigidObject = env.scene[asset_cfg.name]
    key_body_indexes = [asset.data.body_names.index(name) for name in key_body_names]
    key_body_positions = asset.data.body_pos_w[:, key_body_indexes]
    root_positions = asset.data.body_pos_w[:, asset.data.body_names.index("base_link")].unsqueeze(-2)
    return (key_body_positions - root_positions).view(key_body_positions.shape[0], -1)

