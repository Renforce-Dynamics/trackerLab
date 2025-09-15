import os
import time
import mujoco
import mujoco.viewer
from dataclasses import dataclass
from glob import glob
import numpy as np
import rich
import torch
from sim2simlib.model.sim2sim_base import Sim2SimBaseModel, Sim2Sim_Config
from sim2simlib.motion import MotionManagerSim2sim
from trackerLab.motion_buffer.motion_lib.transforms.hdf5_loader import load_trajectories_from_hdf5    

class Sim2SimReplayModel(Sim2SimBaseModel):
    
    motion_manager: MotionManagerSim2sim
    motion_obs_history: dict[str, list[np.ndarray]] = {}
    
    def __init__(self, cfg: Sim2Sim_Config):
        super().__init__(cfg)
        self.trajs = load_trajectories_from_hdf5()
        self.traj_id = 0
        self.traj_prt = 0
        self.mj_model.opt.gravity[:] = 0, 0, 0
        
    @property
    def cur_traj_len(self):
        return self.trajs[self.traj_id]["actions"].shape[0]
    
    @property
    def traj_num(self):
        return len(self.trajs)
        
    def update_traj_prt(self):
        self.traj_prt += 1
        if self.traj_prt >= self.cur_traj_len:
            self.traj_prt = 0
            # self.traj_id += 1
            # self.traj_id = self.traj_id % self.traj_num
        
    def act(self) -> np.ndarray:
        action = self.trajs[self.traj_id]["actions"][self.traj_prt]
        # self.debug("[DEBUG] action:", action)
        self.last_action[:] = action
        self.update_traj_prt()
        return action
    
    def apply_action(self, joint_pos_action: np.ndarray):
        super().apply_action(joint_pos_action)
        self.mj_data.qpos[:3] = self._cfg.default_pos
        self.mj_data.qpos[3:7] = (1, 0, 0, 0)
    
    