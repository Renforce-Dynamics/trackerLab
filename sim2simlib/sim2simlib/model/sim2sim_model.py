import os
import mujoco
from dataclasses import dataclass
from glob import glob
import numpy as np

import torch
from sim2simlib.model.sim2sim_base import Sim2Sim_Base_Model, Sim2Sim_Config
from sim2simlib.sim2sim_manager import Motion_Manager

    
class Sim2Sim_Model(Sim2Sim_Base_Model):
    
    motion_manager: Motion_Manager
    
    def __init__(self, cfg: Sim2Sim_Config):
        super().__init__(cfg)
        self._init_motion_manager()

     
    def _init_motion_manager(self):
        self.motion_manager = Motion_Manager(self._cfg.motion_cfg)
        self.motion_manager.init_finite_state_machine()
        self.motion_manager.set_finite_state_machine_motion_ids(
            motion_ids=torch.tensor([0], device="cpu", dtype=torch.long))
    
    def get_motion_command(self) -> dict[str, np.ndarray]:
        motion_observations = {}
        for term in self._cfg.observation_cfg.motion_observations_terms:
            if hasattr(self.motion_manager, f"{term}"):
                motion_observations[term] = getattr(self.motion_manager, f"{term}")
            else:
                raise ValueError(f"Motion observation term {term} not implemented.")
        return motion_observations
    
    def get_obs(self):
        base_observations = self.get_base_observations()
        motion_command = self.get_motion_command()

        is_update = self.motion_manager.step()
        if torch.any(is_update):
            print("Motion updated.")
            self.motion_manager.set_finite_state_machine_motion_ids(
                motion_ids=torch.tensor([1], device="cpu", dtype=torch.long))
        