import torch
import numpy as np
import gymnasium as gym
from trackerLab.managers import AMPManager
from ..manager_based_tracker_env import ManagerBasedTrackerEnv

import trackerLab.tracker_env.mdp.amp as amp_mdp

class ManagerBasedAMPEnv(ManagerBasedTrackerEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_amp_observations = getattr(self.cfg, "num_amp_observations", 2)
        
        # DEBUG for cusrl
        # self.amp_obs_feat = 59
        amp_obs = self.collect_reference_motions(1)
        self.amp_obs_feat = amp_obs.shape[-1]
        self.amp_observation_size = self.num_amp_observations * self.amp_obs_feat
        
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_obs_feat,))
        self.amp_observation_buffer = torch.zeros(
            (self.scene.num_envs, self.num_amp_observations, self.amp_obs_feat), device=self.device
        )
        self.update_local_amp_buffer()
        return
    
    def load_managers(self):
        super().load_managers()
        self.motion_manager = AMPManager(self.cfg.motion, self, self.device)
        self.motion_manager.compute()
        print("[INFO] Motion Manager: ", self.motion_manager)
        
    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self.motion_manager.sample_times(num_samples)
        times = (
            current_times.unsqueeze(-1)
            - self.motion_manager.motion_dt * torch.arange(0, self.num_amp_observations, device=current_times.device)
        ).flatten()
        
        # get motions
        (
            dof_pos,
            dof_vel,
            root_trans,
            root_rot,
            root_lin_vel,
            root_ang_vel,
        ) = self.motion_manager.sample(num_samples=num_samples, times=times)

        amp_observation = amp_mdp.compute_obs(
            dof_pos,
            dof_vel,
            root_trans,
            root_rot,
            root_lin_vel,
            root_ang_vel,
            # body_positions,
        )

        return amp_observation.reshape(self.num_amp_observations, -1)
        # cusrl中需要view到 (1, 2*self.amp_obs_feat)
        # return amp_observation.view(num_samples, -1)
    
    def collect_local_amp_obs(self):
        
        dof_pos = amp_mdp.amp_dof_pos(self)
        dof_vel = amp_mdp.amp_dof_vel(self)
        root_trans = amp_mdp.amp_root_trans(self)
        root_rot = amp_mdp.amp_root_rot(self)
        root_lin_vel = amp_mdp.amp_root_lin_vel(self)
        root_ang_vel = amp_mdp.amp_root_ang_vel(self)
        
        amp_observation = amp_mdp.compute_obs(
            dof_pos,
            dof_vel,
            root_trans,
            root_rot,
            root_lin_vel,
            root_ang_vel,
            # body_positions,
        )
        
        return amp_observation
    
    def update_local_amp_buffer(self):
        # update AMP observation history
        for i in reversed(range(self.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        local_amp_obs = self.collect_local_amp_obs()
        self.amp_observation_buffer[:, 0] = local_amp_obs
        # To avoid removing other extras
        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)
        self.obs_buf["amp"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)

    
    def step(self, action: torch.Tensor):
        super().step(action)
        self.update_local_amp_buffer()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    