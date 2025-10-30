import torch
import numpy as np

import os
from trackerLab.joint_id_caster import JointIdCaster
from trackerLab.motion_buffer.motion_buffer import MotionBuffer
from trackerLab.motion_buffer.motion_lib import MotionLib
from trackerLab.motion_buffer.motion_buffer_cfg import MotionBufferCfg
from trackerLab.managers.motion_manager import MotionManagerCfg

from deploylib.utils.motion import slerp, lerp

class DeployManager(object):
    
    motion_buffer: MotionBuffer
    id_caster: JointIdCaster
    motion_lib: MotionLib
    
    def __init__(
            self, motion_buffer_cfg: MotionBufferCfg, 
            motion_align_cfg: dict,
            lab_joint_names, robot_type, dt, device
        ):
        self.motion_buffer_cfg = motion_buffer_cfg
        self.lab_joint_names = lab_joint_names
        self.robot_type = robot_type
        self.dt = dt
        self.device = device
        
        self.id_caster = JointIdCaster(device, lab_joint_names, robot_type=robot_type, motion_align_cfg=motion_align_cfg)
        self.motion_buffer = MotionBuffer(motion_buffer_cfg, num_envs=1, dt=dt, device=device, id_caster=self.id_caster)
        self.motion_lib = self.motion_buffer._motion_lib
        
        self.gym2lab_dof_ids = self.id_caster.gym2lab_dof_ids
        self.lab2gym_dof_ids = self.id_caster.lab2gym_dof_ids
        
        self.interp_config = [
            ("loc_trans_base", "lerp"),
            ("loc_root_rot",   "slerp"),
            ("loc_root_pos",   "lerp"),
            ("loc_dof_pos",    "slerp"),
            ("loc_dof_vel",    "lerp"),
            ("loc_root_vel",   "lerp"),
            ("loc_ang_vel",    "lerp"),
            ("loc_local_rot",  "slerp"),
            ("loc_root_vel_global", "lerp"),
        ]
        

    @classmethod
    def from_configclass(cls, cfg: MotionManagerCfg, lab_joint_names, dt, device):
        return cls(cfg.motion_buffer_cfg, cfg.motion_align_cfg, lab_joint_names, cfg.robot_type, dt, device)

    def init_finite_state_machine(self):
        self.motion_buffer._motion_ids = torch.zeros_like(self.motion_buffer._motion_ids, dtype=torch.long, device=self.device)
        self.motion_buffer._motion_times = torch.zeros_like(self.motion_buffer._motion_times, dtype=torch.float, device=self.device)        
        
    def set_finite_state_machine_motion_ids(self, motion_ids):
        """
        Set the motion ids for the finite state machine.
        """
        assert motion_ids.shape[0] == self.motion_buffer._motion_ids.shape[0], "Motion ids shape mismatch."
        self.motion_buffer._motion_ids = motion_ids.to(self.device, dtype=torch.long)
        self.motion_buffer._motion_times = torch.zeros_like(self.motion_buffer._motion_times, dtype=torch.float, device=self.device)
        
    def add_finite_state_machine_motion_ids(self):
        self.motion_buffer._motion_ids += 1
        self.motion_buffer._motion_ids %= self.motion_lib.num_motions()
        
    def step(self):
        """
        Step the motion buffer and update the motion library.
        """
        is_update = self.motion_buffer.update_motion_times()
        self.loc_gen_state(self.motion_buffer._motion_times, self.motion_buffer._motion_ids)
        return is_update

    loc_trans_base: torch.Tensor = None
    loc_root_pos: torch.Tensor = None # This is demo given
    loc_dof_pos: torch.Tensor = None
    loc_dof_vel: torch.Tensor = None
    loc_root_rot: torch.Tensor = None
    loc_ang_vel: torch.Tensor = None
    loc_root_vel_global: torch.Tensor = None
    
    loc_init_root_pos: torch.Tensor = None
    loc_init_demo_root_pos: torch.Tensor = None
    
    @property
    def loc_height(self):
        return self.loc_root_pos[:, 2]

    def calc_loc_terms(self, frame):
        return {
            "loc_trans_base": self.motion_lib.ltbs[frame],
            "loc_root_rot":   self.motion_lib.grs[frame, 0],
            "loc_root_pos":   self.motion_lib.gts[frame, 0],
            "loc_local_rot":  self.motion_lib.lrs[frame],
            "loc_dof_vel":    self.motion_lib.dvs[frame],
            "loc_dof_pos":    self.motion_lib.dps[frame],
            "loc_root_vel":   self.motion_lib.lvbs[frame],
            "loc_ang_vel":    self.motion_lib.avbs[frame],
            "loc_root_vel_global": self.motion_lib.grvs[frame],
        }

    def interpolate_term(self, term0, term1, blend, mode):
        if term0 is None or term1 is None:
            return term0
        if mode == "lerp":
            return lerp(term0, term1, blend)
        elif mode == "slerp":
            return slerp(term0, term1, blend)
        elif mode == "mean":
            return 0.5 * (term0 + term1)
        else:
            raise ValueError(f"Unsupported interpolation mode: {mode}")

    def loc_gen_state(self, time, motion_ids):
        f0, f1, blend = self.motion_lib.get_frame_idx(motion_ids, time)
        blend = blend.unsqueeze(-1)

        terms_0 = self.calc_loc_terms(f0)
        terms_1 = self.calc_loc_terms(f1)

        results = {}
        for name, mode in self.interp_config:
            results[name] = self.interpolate_term(
                terms_0[name], terms_1[name], blend, mode
            )

        dof_pos, dof_vel = self.motion_buffer.reindex_dof_pos_vel(
            results["loc_dof_pos"], results["loc_dof_vel"]
        )
        results["loc_dof_pos"] = dof_pos[:, self.gym2lab_dof_ids]
        results["loc_dof_vel"] = dof_vel[:, self.gym2lab_dof_ids]

        for k, v in results.items():
            setattr(self, k, v)

        return results
        