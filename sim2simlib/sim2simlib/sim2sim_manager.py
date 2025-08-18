import torch
import numpy as np

import os
from trackerLab.managers.joint_id_caster import JointIdCaster
from trackerLab.motion_buffer.motion_buffer import MotionBuffer
from trackerLab.motion_buffer.motion_lib import MotionLib
from trackerLab.motion_buffer.motion_buffer_cfg import MotionBufferCfg

from utils import slerp

class Motion_Manager(object):
    
    motion_buffer: MotionBuffer
    id_caster: JointIdCaster
    motion_lib: MotionLib
    
    def __init__(
            self, motion_buffer_cfg: MotionBufferCfg, 
            lab_joint_names, robot_type, dt, device
        ):
        self.motion_buffer_cfg = motion_buffer_cfg
        self.lab_joint_names = lab_joint_names
        self.robot_type = robot_type
        self.dt = dt
        self.device = device
        
        self.id_caster = JointIdCaster(device, lab_joint_names, robot_type=robot_type)
        self.motion_buffer = MotionBuffer(motion_buffer_cfg, num_envs=1, dt=dt, device=device, id_caster=self.id_caster)
        self.motion_lib = self.motion_buffer._motion_lib
        
        self.gym2lab_dof_ids = self.id_caster.gym2lab_dof_ids
        self.lab2gym_dof_ids = self.id_caster.lab2gym_dof_ids
        pass

    def init_finite_state_machine(self):
        self.motion_buffer._motion_ids = torch.zeros_like(self.motion_buffer._motion_times, dtype=torch.long, device=self.device)
        self.motion_buffer._motion_times = torch.zeros_like(self.motion_buffer._motion_times, dtype=torch.float, device=self.device)        
        
    def set_finite_state_machine_motion_ids(self, motion_ids):
        """
        Set the motion ids for the finite state machine.
        """
        assert motion_ids.shape[0] == self.motion_buffer._motion_ids.shape[0], "Motion ids shape mismatch."
        self.motion_buffer._motion_ids = motion_ids.to(self.device, dtype=torch.long)
        self.motion_buffer._motion_times = torch.zeros_like(self.motion_buffer._motion_times, dtype=torch.float, device=self.device)
        
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
    
    loc_init_root_pos: torch.Tensor = None
    loc_init_demo_root_pos: torch.Tensor = None
    
    @property
    def loc_height(self):
        return self.loc_root_pos[:, 2]

    def calc_loc_terms(self, frame):
        """
        Calc terms at certain frame.
        """
        loc_trans_base  = self.motion_lib.trans_base[frame]
        loc_root_rot    = self.motion_lib.grs[frame, 0]
        loc_root_pos    = self.motion_lib.gts[frame, 0]
        loc_local_rot   = self.motion_lib.lrs[frame]
        loc_dof_vel     = self.motion_lib.dvs[frame]
        loc_dof_pos     = self.motion_lib.dof_pos[frame]
        loc_root_vel    = self.motion_lib.vels_base[frame]
        loc_ang_vel     = self.motion_lib.ang_vels_base[frame]
        return loc_trans_base, loc_root_rot, loc_root_pos, \
            loc_dof_pos, loc_dof_vel, loc_root_vel, loc_ang_vel, loc_local_rot
    
    def loc_gen_state(self, time, motion_ids):
        f0l, f1l, blend = self.motion_lib.get_frame_idx(motion_ids, time)
        
        terms_0, terms_1 = self.calc_loc_terms(f0l), self.calc_loc_terms(f1l)
        
        terms = []
        for term0, term1 in zip(terms_0, terms_1):
            if term0 is not None:
                terms.append((term0 + term1)/2)
            else:
                terms.append(term0)
        
        self.loc_trans_base, _, self.loc_root_pos, \
            _, loc_dof_vel, self.loc_root_vel, self.loc_ang_vel, _ = terms
        
        blend = blend.unsqueeze(-1)
        self.loc_root_rot = slerp(terms_0[1], terms_1[1], blend)
        
        blend = blend.unsqueeze(-1)
        loc_local_rot = slerp(terms_0[7], terms_1[7], blend)
        loc_dof_pos = self.motion_lib._local_rotation_to_dof(loc_local_rot)
        
        loc_dof_pos, loc_dof_vel = self.motion_buffer.reindex_dof_pos_vel(loc_dof_pos, loc_dof_vel)
        self.loc_dof_pos, self.loc_dof_vel = loc_dof_pos[:, self.gym2lab_dof_ids], loc_dof_vel[:, self.gym2lab_dof_ids]
        