import os
import time
import mujoco
import mujoco.viewer
from dataclasses import dataclass
from glob import glob
import numpy as np

import torch
from sim2simlib.model.sim2sim_base import Sim2Sim_Base_Model, Sim2Sim_Config
from sim2simlib.motion.motion_manager import MotionManagerSim2sim

    
class Sim2Sim_Motion_Model(Sim2Sim_Base_Model):
    
    motion_manager: MotionManagerSim2sim
    motion_obs_history: dict[str, list[np.ndarray]] = {}
    
    def __init__(self, cfg: Sim2Sim_Config):
        super().__init__(cfg)
        self._init_motion_manager()
        self._init_motion_joint_maps()
        self._init_motion_observation_history()
     
    def _init_motion_manager(self):
        self.motion_manager = MotionManagerSim2sim.from_configclass(
                                cfg=self._cfg.motion_cfg,
                                lab_joint_names=self.policy_joint_names,
                                dt=self._cfg.simulation_dt * self._cfg.control_decimation,
                                device="cpu"
                            )
        self.motion_manager.init_finite_state_machine()
        self.motion_manager.set_finite_state_machine_motion_ids(
            motion_ids=torch.tensor([0], device="cpu", dtype=torch.long))
   
    def _init_motion_joint_maps(self):
        """Initialize mapping from motion joint order to mujoco actuator joint order."""
        
        # motion joint order -> mujoco joint order
        self.motion_joint_names = self.motion_manager.id_caster.shared_subset_lab_names
        self.motion_maps = []
        
        print('[INFO] Motion joint names:', self.motion_joint_names)
        print('[INFO] Actuator joint names:', self.actuators_joint_names)
        
        # Create mapping from motion joint names to mujoco actuator joint indices
        for motion_joint_name in self.motion_joint_names:
            if motion_joint_name in self.actuators_joint_names:
                # Find the index in actuators_joint_names (which corresponds to mujoco joint order)
                mujoco_idx = self.actuators_joint_names.index(motion_joint_name)
                self.motion_maps.append(mujoco_idx)
            else:
                raise ValueError(f"Motion joint name {motion_joint_name} not found in MuJoCo actuator joints.")
        
        print(f"[INFO] Motion maps (motion order -> mujoco actuator order): {self.motion_maps}")
        
        self.motion_maps = [item + self.base_link_id + 7 for item in self.motion_maps]
        
    def _init_motion_observation_history(self):
        """Initialize motion observation history buffers."""
        if self._cfg.observation_cfg.using_motion_obs_history:
            # Get initial motion observations to determine sizes
            initial_obs = self._get_current_motion_observations()
            
            # Initialize history buffers for each motion observation term
            for term, obs_value in initial_obs.items():
                self.motion_obs_history[term] = [obs_value.copy() for _ in range(self._cfg.observation_cfg.motion_obs_his_length)]

    def _get_current_motion_observations(self) -> dict[str, np.ndarray]:
        """Get current motion observations without history processing."""
        is_update = self.motion_manager.step()
        motion_observations = {}
        for term in self._cfg.observation_cfg.motion_observations_terms:
            if hasattr(self.motion_manager, f"{term}"):
                term_data = getattr(self.motion_manager, f"{term}")
                if type(term_data) is torch.Tensor:
                    term_data = term_data.detach().cpu().numpy()[0]
                motion_observations[term] = term_data.astype(np.float32)
            else:
                raise ValueError(f"Motion observation term '{term}' not implemented.")
        if torch.any(is_update):
            self.motion_manager.set_finite_state_machine_motion_ids(
                motion_ids=torch.tensor([0], device="cpu", dtype=torch.long))
        return motion_observations
    
    def _update_motion_observation_history(self):
        """Update motion observation history with current observations."""
        if self._cfg.observation_cfg.using_motion_obs_history:
            current_obs = self._get_current_motion_observations()
            
            for term, obs_value in current_obs.items():
                # Shift history (remove oldest, add newest)
                self.motion_obs_history[term] = self.motion_obs_history[term][1:] + [obs_value.copy()]
        
    
    def get_motion_command(self) -> dict[str, np.ndarray]:
        """Get motion observations with optional history."""
        # Update history before getting observations
        if self._cfg.observation_cfg.using_motion_obs_history:
            self._update_motion_observation_history()
            
        motion_observations = {}
        
        if self._cfg.observation_cfg.using_motion_obs_history:
            # Return historical observations
            for term in self._cfg.observation_cfg.motion_observations_terms:
                if term in self.motion_obs_history:
                    if self._cfg.observation_cfg.motion_obs_flatten:
                        # Flatten history: [t-2, t-1, t] -> concatenated array
                        motion_observations[term] = np.concatenate(self.motion_obs_history[term], axis=-1)
                    else:
                        # Keep as separate timesteps: shape (history_length, obs_dim)
                        motion_observations[term] = np.stack(self.motion_obs_history[term], axis=0)
                else:
                    raise ValueError(f"Motion observation term '{term}' not found in history.")
        else:
            # Return current observations without history
            motion_observations = self._get_current_motion_observations()
        
        return motion_observations
    
    def get_obs(self) -> dict[str, np.ndarray]:
        base_observations = self.get_base_observations()
        motion_command = self.get_motion_command()
        return  motion_command | base_observations

    def motion_fk_view(self):
        """
        Visualize motion tracking using forward kinematics.
        
        This method displays the robot following motion data by directly setting
        joint positions from the motion manager to the MuJoCo simulation.
        """
        counter = 0
        print(f"[INFO] Starting motion forward kinematics visualization...")
        print(f"[INFO] Motion joints: {len(self.motion_joint_names)}, Actuator joints: {len(self.actuators_joint_names)}")
        
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                if counter % self._cfg.control_decimation == 0:
                    # Update motion manager to get new motion data
                    is_update = self.motion_manager.step()
                    
                    # Get motion joint positions (local DOF positions)
                    loc_dof_pos = self.motion_manager.loc_dof_pos.detach().cpu().numpy()[0]
                    
                    # Control qpos
                    self.mj_data.qpos[self.motion_maps] = loc_dof_pos
                    self.mj_data.qpos[:3] = self._cfg.default_pos
                    
                    # Optional: Print motion update info
                    if torch.any(is_update):
                        print(f"[INFO] Motion updated at step {counter}")
                
                # Forward kinematics computation (no dynamics)
                mujoco.mj_forward(self.mj_model, self.mj_data)
                viewer.sync()  

                counter += 1
                time_until_next_step = self.mj_model.opt.timestep * self.slowdown_factor - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
