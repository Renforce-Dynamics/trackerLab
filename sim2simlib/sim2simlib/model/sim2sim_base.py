import os
import time
import mujoco
import mujoco.viewer
from dataclasses import dataclass
from glob import glob
import numpy as np
import torch
import re

from sim2simlib import SIM2SIMLIB_ASSETS_DIR
from sim2simlib.utils.utils import get_gravity_orientation
from sim2simlib.model.dc_motor import DC_Motor, PID_Motor
from sim2simlib.model.config import Sim2Sim_Config, Actions

class Sim2Sim_Base_Model:
    
    def __init__(self, cfg: Sim2Sim_Config):
        self._cfg = cfg
        if cfg.xml_path is None:
            self.xml_path = glob(os.path.join(SIM2SIMLIB_ASSETS_DIR, cfg.robot_name, "mjcf", "*.xml"))[0]
        else:
            self.xml_path = cfg.xml_path
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = cfg.simulation_dt
        self.slowdown_factor = self._cfg.slowdown_factor
        
        #
        self.action = np.zeros(len(cfg.policy_joint_names), dtype=np.float32)
        self.base_link_id = 0
        self.policy_joint_names = self._cfg.policy_joint_names
        self.qpos_maps = []
        self.qvel_maps = []
        self.act_maps = []
        self.cmd = [0, 0, 0]
        
        # Initialize observation history
        self.base_obs_history = {}        
        # 
        self._init_joint_names()
        self._init_default_pos_angles()
        self._init_load_policy()
        self._init_dc_model()
        self._init_observation_history()

    def _init_joint_names(self):
        self.mujoco_joint_names = [self.mj_model.jnt(i).name for i in range(self.mj_model.njnt)]
        self.qpos_strat_ids = [self.mj_model.jnt_qposadr[i] for i in range(self.mj_model.njnt)]
        self.qvel_strat_ids = [self.mj_model.jnt_dofadr[i] for i in range(self.mj_model.njnt)]

        self.actuators_joint_names = self.mujoco_joint_names[1:]
        print('[INFO] MuJoCo joint names:', self.mujoco_joint_names)
        print('[INFO] Policy joint names:', self.policy_joint_names)
        
        # mujoco order -> policy order
        for joint_name in self.policy_joint_names:
            if joint_name in self.mujoco_joint_names:
                idx = self.mujoco_joint_names.index(joint_name)
                self.qpos_maps.append(self.qpos_strat_ids[idx])
                self.qvel_maps.append(self.qvel_strat_ids[idx])
            else:
                raise ValueError(f"Joint name {joint_name} not found in MuJoCo model.")

        print("[INFO] qpos maps:", self.qpos_maps)
        print("[INFO] qvel maps:", self.qvel_maps)

        # policy order -> mujoco order
        for joint_name in self.mujoco_joint_names:
            if joint_name in self.policy_joint_names:
                idx = self.policy_joint_names.index(joint_name)
                self.act_maps.append(idx)

        print("[INFO] Action maps:", self.act_maps)
    
    def _init_default_pos_angles(self):
        self.mj_data.qpos[:3] = self._cfg.default_pos
        if isinstance(self._cfg.default_angles, np.ndarray):
            self.mj_data.qpos[7:] = self._cfg.default_angles
        elif isinstance(self._cfg.default_angles, dict):
            for joint_name, angle in self._cfg.default_angles.items():
                for i, name in enumerate(self.actuators_joint_names):
                    if re.match(joint_name, name):
                        self.mj_data.qpos[i + 7] = angle
                        break
        self.init_qpos = self.mj_data.qpos.copy()
        self.init_angles = self.mj_data.qpos[7:].copy()

    def _init_load_policy(self):
        self.policy = torch.jit.load(self._cfg.policy_path)
        
    def _init_dc_model(self):
        motor_type = self._cfg.motor_cfg.motor_type
        self._cfg.motor_cfg.joint_names = self.actuators_joint_names
        self.dc_motor = motor_type(self._cfg.motor_cfg)         
    
    def _init_observation_history(self):
        """Initialize observation history buffers."""
        if self._cfg.observation_cfg.using_base_obs_history:
            # Get initial observations to determine sizes
            initial_obs = self._get_current_base_observations()
            
            # Initialize history buffers for each observation term
            for term, obs_value in initial_obs.items():
                self.base_obs_history[term] = [obs_value.copy() for _ in range(self._cfg.observation_cfg.base_obs_his_length)]

    def _get_current_base_observations(self) -> dict[str, np.ndarray]:
        """Get current observations without history processing."""
        base_observations = {}
        for term in self._cfg.observation_cfg.base_observations_terms:
            if hasattr(self, f"_obs_{term}"):
                base_observations[term] = getattr(self, f"_obs_{term}")() * self._cfg.observation_cfg.scale[term]
            else:
                raise ValueError(f"Observation term {term} not implemented.")
        return base_observations
    
    def _update_observation_history(self):
        """Update observation history with current observations."""
        if self._cfg.observation_cfg.using_base_obs_history:
            current_obs = self._get_current_base_observations()
            
            for term, obs_value in current_obs.items():
                # Shift history (remove oldest, add newest)
                self.base_obs_history[term] = self.base_obs_history[term][1:] + [obs_value.copy()]         
            
    def get_base_observations(self) -> dict[str, np.ndarray]:
        """Get base observations with optional history."""
        # Update history before getting observations
        if self._cfg.observation_cfg.using_base_obs_history:
            self._update_observation_history()
            
        base_observations = {}
        
        if self._cfg.observation_cfg.using_base_obs_history:
            # Return historical observations
            for term in self._cfg.observation_cfg.base_observations_terms:
                if term in self.base_obs_history:
                    if self._cfg.observation_cfg.base_obs_flatten:
                        # Flatten history: [t-2, t-1, t] -> concatenated array
                        base_observations[term] = np.concatenate(self.base_obs_history[term], axis=-1)
                    else:
                        # Keep as separate timesteps: shape (history_length, obs_dim)
                        base_observations[term] = np.stack(self.base_obs_history[term], axis=0)
                else:
                    raise ValueError(f"Observation term {term} not found in history.")
        else:
            # Return current observations without history
            for term in self._cfg.observation_cfg.base_observations_terms:
                if hasattr(self, f"_obs_{term}"):
                    base_observations[term] = getattr(self, f"_obs_{term}")() * self._cfg.observation_cfg.scale[term]
                else:
                    raise ValueError(f"Observation term {term} not implemented.")
        
        return base_observations
    
    def get_obs(self) -> dict[str, np.ndarray]:
        base_observations = self.get_base_observations()
        return base_observations

    def process_action(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(action, *self._cfg.action_cfg.action_clip)
        action *= self._cfg.action_cfg.scale
        ctrl_action = action[self.act_maps]
        target_joint_pos = ctrl_action + self.init_angles
        return target_joint_pos

    def pid_control(self, target_joint_pos: np.ndarray):
        tau = self.dc_motor.compute(
                joint_pos=self.mj_data.qpos[self.base_link_id + 7:],
                joint_vel=self.mj_data.qvel[self.base_link_id + 6:],
                action=Actions(
                    joint_pos=target_joint_pos,
                    joint_vel=np.zeros_like(target_joint_pos),
                    joint_efforts=np.zeros_like(target_joint_pos)
                )
            )
        self.mj_data.ctrl[:] = tau

    def act(self) -> np.ndarray:
        obs_dict = self.get_obs()
        obs_np = np.concatenate(list(obs_dict.values()), axis=-1).astype(np.float32)
        obs_tensor = torch.from_numpy(obs_np).unsqueeze(0)
        print(obs_tensor)
        action = self.policy(obs_tensor).detach().numpy().squeeze()
        print(action)
        self.action[:] = action
        return action

    def _obs_base_lin_vel(self) -> np.ndarray:
        return self.mj_data.qvel[self.base_link_id : self.base_link_id + 3]
    
    def _obs_base_ang_vel(self) -> np.ndarray:
        return self.mj_data.qvel[self.base_link_id + 3 : self.base_link_id + 6]
    
    def _obs_cmd(self) -> np.ndarray:
        return np.array(self.cmd, dtype=np.float32)
    
    def _obs_gravity_orientation(self) -> np.ndarray:
        return get_gravity_orientation(self.mj_data.qpos[self.base_link_id + 3:self.base_link_id + 7])
    
    def _obs_joint_pos(self) -> np.ndarray:
        print(f"{self.mj_data.qpos=}")
        return (self.mj_data.qpos - self.init_qpos)[self.qpos_maps]
    
    def _obs_joint_vel(self) -> np.ndarray:
        return self.mj_data.qvel[self.qvel_maps]
    
    def _obs_action(self) -> np.ndarray:
        return self.action
    
    def headless_run(self):
        counter = 0
        target_joint_pos = self._cfg.default_angles
        while True:
            step_start = time.time()
            if counter % self._cfg.control_decimation == 0:
                action = self.act()
                target_joint_pos = self.process_action(action)
                    
            self.pid_control(target_joint_pos)
            mujoco.mj_step(self.mj_model, self.mj_data)
            
            counter += 1
            time_until_next_step = self.mj_data.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
    
    def view_run(self):
        counter = 0
        target_joint_pos = self._cfg.default_angles
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                if counter % self._cfg.control_decimation == 0:
                    action = self.act()
                    target_joint_pos = self.process_action(action)
                    
                self.pid_control(target_joint_pos)
                mujoco.mj_step(self.mj_model, self.mj_data)
                viewer.sync()  

                counter += 1
                
                time_until_next_step = self.mj_model.opt.timestep*self.slowdown_factor - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
