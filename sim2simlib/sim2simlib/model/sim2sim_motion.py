import os
import time
import mujoco
import mujoco.viewer
from dataclasses import dataclass
from glob import glob
import numpy as np

import torch
from sim2simlib.model.sim2sim_base import Sim2Sim_Base_Model, Sim2Sim_Config
from sim2simlib.motion.sim2sim_manager import Motion_Manager

    
class Sim2Sim_Motion_Model(Sim2Sim_Base_Model):
    
    motion_manager: Motion_Manager
    
    def __init__(self, cfg: Sim2Sim_Config):
        super().__init__(cfg)
        self._init_motion_manager()
     
    def _init_motion_manager(self):
        self.motion_manager = Motion_Manager(
                                motion_buffer_cfg=self._cfg.motion_cfg,
                                lab_joint_names=self.policy_joint_names,
                                robot_type=self._cfg.robot_name,
                                dt=self._cfg.simulation_dt,
                                device="cpu"
                            )
        self.motion_manager.init_finite_state_machine()
        self.motion_manager.set_finite_state_machine_motion_ids(
            motion_ids=torch.tensor([0], device="cpu", dtype=torch.long))
    
    def get_motion_command(self) -> dict[str, np.ndarray]:
        motion_observations = {}
        for term in self._cfg.observation_cfg.motion_observations_terms:
            if hasattr(self.motion_manager, f"{term}"):
                term_data = getattr(self.motion_manager, f"{term}")
                if type(term_data) is torch.Tensor:
                    term_data = term_data.detach().cpu().numpy()[0]
                motion_observations[term] = term_data.astype(np.float32)
            else:
                raise ValueError(f"Motion observation term {term} not implemented.")
        return motion_observations
    
    def get_obs(self) -> dict[str, np.ndarray]:
        is_update = self.motion_manager.step()
        base_observations = self.get_base_observations()
        motion_command = self.get_motion_command()

        if torch.any(is_update):
            print("Motion updated.")
            self.motion_manager.set_finite_state_machine_motion_ids(
                motion_ids=torch.tensor([1], device="cpu", dtype=torch.long))

        return  motion_command | base_observations

    def motion_view(self):
        counter = 0
        target_joint_pos = self._cfg.default_angles
        self.mj_model.opt.gravity[:] = [0, 0, 0]
        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                if counter % 4 == 0:
                    is_update = self.motion_manager.step()
                    loc_dof_pos = self.motion_manager.loc_dof_pos.detach().cpu().numpy()[0]
                    loc_dof_pos = np.concatenate([np.array([0], dtype=np.float32), loc_dof_pos])
                    print("loc_dof_pos:", loc_dof_pos)
                    target_joint_pos = self._cfg.default_angles + loc_dof_pos
                    self.mj_data.qpos[self.base_link_id+7:] = target_joint_pos
                    self.mj_data.qpos[:3] = self._cfg.default_pos
                    
                mujoco.mj_forward(self.mj_model, self.mj_data)
                viewer.sync()  

                counter += 1
                time_until_next_step = self.mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
