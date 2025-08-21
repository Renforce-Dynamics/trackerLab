import os
import torch
from .utils.jit_func import reindex_motion_dof
from .motion_lib import MotionLib
from .motion_buffer_cfg import MotionBufferCfg

from trackerLab.joint_id_caster import JointIdCaster

class MotionBuffer(object):
    """
    Interface for the motion buffer. Which will load the motion data and output the motion data in the gym order.
    The motion data loaded in buffer is shaped as [num_envs, motion_length, motion_dim].
    Or rather, in the variable format is [num_envs, cfg.n_demo_steps, n_demo].
    n_demo is naturally a constant value.
    """

    def __init__(self, cfg: MotionBufferCfg, num_envs, dt, device, id_caster: JointIdCaster=None):
        super(MotionBuffer, self).__init__()
        self.device = device
        self.cfg: MotionBufferCfg = cfg
        self.num_envs = num_envs
        self.dt = dt

        self.id_caster = id_caster

        self.init_motions()
        self.init_motion_buffers()
        
    # function
    reindex_motion_dof = reindex_motion_dof
        
    def init_motions(self):
        self._motion_lib = MotionLib(motion_file=self.cfg.motion.motion_name,
                                     id_caster=self.id_caster,
                                     device=self.device, 
                                     regen_pkl=self.cfg.motion.regen_pkl)

    def init_motion_buffers(self):
        """
        Initialize the motion buffers. For this buffer will have following key components:
        - motion_ids: the ids of the motions
        - motion_times: the times of the motions, will init in rand [0, motion_length)
        - motion_lengths: the lengths of the motions
        - motion_difficulty: the difficulty of the motions
        - motion_features: (not used) the features of the motions
        - motion_dt: the dt of the motions, equal to the simulation dt (self.dt)
        - motion_num_future_steps: the number of future steps of the motions
        - motion_demo_offsets: the offsets of the motions
        - dof_term_threshold: the threshold for the dof termination
        - keybody_term_threshold: the threshold for the key body termination
        - yaw_term_threshold: the threshold for the yaw termination
        - height_term_threshold: the threshold for the height termination
        - step_inplace_ids: (not used) the ids for the step in place motions
        """
        num_motions = self._motion_lib.num_motions()
        self._motion_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._motion_ids = torch.remainder(self._motion_ids, num_motions)
        self._max_motion_difficulty = 9
        self._motion_times = self._motion_lib.sample_time(self._motion_ids)
        self._motion_lengths = self._motion_lib.get_motion_length(self._motion_ids)
        self._motion_difficulty = self._motion_lib.get_motion_difficulty(self._motion_ids)
        self._motion_dt = self.dt

    # resample for reset
    def resample_motion_times(self, env_ids):
        return self._motion_lib.sample_time(self._motion_ids[env_ids])
    
    def update_motion_ids(self, env_ids):
        """
        This only used in reset.
        """
        self._motion_times[env_ids] = self.resample_motion_times(env_ids)
        self._motion_lengths[env_ids] = self._motion_lib.get_motion_length(self._motion_ids[env_ids])
        self._motion_difficulty[env_ids] = self._motion_lib.get_motion_difficulty(self._motion_ids[env_ids])

    # update functions
    def update_motion_times(self):
        self._motion_times += self._motion_dt
        update_flag = self._motion_times >= self._motion_lengths
        self._motion_times[update_flag] = 0.
        return update_flag

    def reindex_dof_pos_vel(self, dof_pos, dof_vel):
        valid_dof_body_ids = self.id_caster.valid_dof_body_ids
        dof_indices_sim = self.id_caster.dof_indices_sim
        dof_indices_motion = self.id_caster.dof_indices_motion
        dof_pos = reindex_motion_dof(dof_pos, dof_indices_sim, dof_indices_motion, valid_dof_body_ids)
        dof_vel = reindex_motion_dof(dof_vel, dof_indices_sim, dof_indices_motion, valid_dof_body_ids)
        return dof_pos, dof_vel


    