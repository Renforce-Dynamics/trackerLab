import torch
import numpy as np
import gymnasium as gym
from trackerLab.managers import AMPManager
from ..manager_based_tracker_env import ManagerBasedTrackerEnv
from isaaclab.utils.math import quat_apply as quat_rotate

class ManagerBasedAMPEnv(ManagerBasedTrackerEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_amp_observations = getattr(self.cfg, "num_amp_observations", 2)
        
        
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

        amp_observation = compute_obs(
            dof_pos,
            dof_vel,
            root_trans,
            root_rot,
            root_lin_vel,
            root_ang_vel,
            # body_positions,
        )

        return amp_observation.reshape(self.num_amp_observations, -1)
    
    def collect_local_amp_obs(self):
        
        dof_pos = self.motion_manager.get_subset_real(self.robot.data.joint_pos)
        dof_vel = self.motion_manager.get_subset_real(self.robot.data.joint_vel)
        root_trans = self.robot.data.root_pos_w
        root_rot = self.robot.data.root_link_quat_w
        root_lin_vel = self.robot.data.root_lin_vel_b
        root_ang_vel = self.robot.data.root_ang_vel_b
        
        amp_observation = compute_obs(
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
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

    
    def step(self, action: torch.Tensor):
        super().step(action)
        self.update_local_amp_buffer()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
    
@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_rotate(q, ref_tangent)
    normal = quat_rotate(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)
    
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    # key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            # (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs