# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from trackerLab.tracker_env import ManagerBasedTrackerEnv


class SkillAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: SkillActionCfg
    _env: ManagerBasedTrackerEnv
    """The configuration of the action term."""

    def __init__(self, cfg: SkillActionCfg, env: ManagerBasedTrackerEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        self.policy = torch.load(cfg.policy_path).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # prepare low level actions
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        def last_action():
            # reset the low level actions if the episode was reset
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # remap some of the low level observations to internal observations
        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()

        # add the low level observations to the observation manager
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._env.motion_manager.skill_graph.num_skills

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        env_ids = torch.arange(self.num_envs, device=self.device)
        # One hot actions to skill ids
        skill_ids = torch.argmax(actions, dim=-1).to(self.device)
        self._env.motion_manager.set_skill(env_ids, skill_ids)

    def apply_actions(self):
        """
        Only apply low level actions, since the high level actions are already processed
        """
        if self._counter % self.cfg.low_level_action_decimation == 0:
            # update the low level actions
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy.act_inference(low_level_obs)
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        if self._counter % self.cfg.low_level_action_decimation == 0:
            self._env.motion_manager.compute()
        self._counter += 1


@configclass
class SkillActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`TrackAction` for more details.
    """

    class_type: type[ActionTerm] = SkillAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    low_level_action_decimation: int = 4
    """Decimation factor for low level actions."""
