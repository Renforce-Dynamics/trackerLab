import torch
import numpy as np
from trackerLab.managers import MotionManager, MotionManagerCfg, SkillManager, SkillManagerCfg, AMPMotionManager, AMPMotionManagerCfg
from .manager_based_tracker_env import ManagerBasedTrackerEnv
from .manager_based_amp_tracker_env_cfg import ManagerBasedAMPTrackerEnvCfg
import gymnasium as gym
from .manager_based_amp_tracker_env_cfg.amp_mdp.utils import compute_obs

class ManagerBasedAMPTrackerEnv(ManagerBasedTrackerEnv):
    cfg: ManagerBasedAMPTrackerEnvCfg
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # motion的body_names
        # self._body_names = self.motion_manager.lab_joint_names
        self._body_names = self.scene['robot'].data.body_names

        
        # # self.motion_dof_indexes = [ i for i in range(27)]
        self.motion_ref_body_index = self.get_body_index([self.cfg.reference_body])[0]
        # self.motion_key_body_indexes = self.get_body_index(self.cfg.key_body_names)

        self.num_amp_observations = self.cfg.num_amp_observations
        # same with the amp_obs size
        self.num_amp_observation_space = self.cfg.num_amp_observation_space
        
        
        self.amp_observation_size = self.num_amp_observations * self.num_amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.scene.num_envs, self.num_amp_observations, self.num_amp_observation_space), device=self.device
        )
    
    def load_managers(self):
        # prepare the managers
        # -- motion manager
        super().load_managers()
        self.motion_manager = AMPMotionManager(self.cfg.motion, self, self.device)
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
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self.motion_manager.sample(num_samples=num_samples, times=times)
        # compute AMP observation
        # amp_observation = compute_obs(
        #     dof_positions[:, self.motion_dof_indexes],
        #     dof_velocities[:, self.motion_dof_indexes],
        #     body_positions[:, self.motion_ref_body_index],
        #     body_rotations[:, self.motion_ref_body_index],
        #     body_linear_velocities,
        #     body_angular_velocities,
        #     body_positions[:, self.motion_key_body_indexes],
        # )
        amp_observation = compute_obs(
            dof_positions,
            dof_velocities,
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities,
            body_angular_velocities,
            # body_positions,
        )

        return amp_observation.view(-1, self.amp_observation_size)
    
    def amp_step(self, env_returns):
        # update AMP observation history
        for i in reversed(range(self.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = env_returns[0]['amp_obs'].clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}
        env_returns[-1]["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)
        return env_returns
    
    # TODO: 这个函数会不会已有一份
    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes
    
    def step(self, action: torch.Tensor):
        action = torch.tensor(action, device=self.device)
        returns = super().step(action)
        returns = self.amp_step(returns)
        return returns