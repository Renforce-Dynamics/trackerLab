import os
import torch
from .utils.jit_func import build_demo_observations, reindex_motion_dof, build_demo_observations_key_pos
from .motion_lib import MotionLib
from .motion_buffer_cfg import MotionBufferCfg

from trackerLab.managers.joint_id_caster import JointIdCaster

from poselib import POSELIB_DATA_DIR

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

        self.init_key_mapping(cfg)
        self.init_motions(cfg)
        self.init_motion_buffers(cfg)
        
    # function
    reindex_motion_dof = reindex_motion_dof

    dof_body_ids: torch.Tensor
    dof_offsets: torch.Tensor
    valid_dof_body_ids: torch.Tensor
    dof_indices_sim: torch.Tensor
    dof_indices_motion: torch.Tensor
    def init_key_mapping(self, cfg: MotionBufferCfg):
        gym_joint_names, dof_body_ids, dof_offsets, valid_dof_body_ids, dof_indices_sim, dof_indices_motion = \
            self.id_caster.init_gym_motion_offset()
        _, self.dof_body_ids, self.dof_offsets, self.valid_dof_body_ids, self.dof_indices_sim, self.dof_indices_motion \
            = gym_joint_names, dof_body_ids, dof_offsets, valid_dof_body_ids, dof_indices_sim, dof_indices_motion


    def init_motions(self, cfg: MotionBufferCfg):
        if cfg.motion.motion_type == "single":
            motion_file = os.path.join(POSELIB_DATA_DIR, "retarget_npy", f"{cfg.motion.motion_name}.npy")
        else:
            assert cfg.motion.motion_type == "yaml"
            motion_file = os.path.join(POSELIB_DATA_DIR, "configs",f"{cfg.motion.motion_name}")
        
        self._load_motion(motion_file)

    def init_motion_buffers(self, cfg: MotionBufferCfg):
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
        # self._motion_num_future_steps = self.cfg.n_demo_steps
        # self._motion_demo_offsets = \
        #     torch.arange(0, self.cfg.n_demo_steps * self.cfg.interval_demo_steps, 
        #                  self.cfg.interval_demo_steps, device=self.device)
        # self._in_place_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _load_motion(self, motion_file):
        # assert(self._dof_offsets[-1] == self.num_dof + 2)  # +2 for hand dof not used
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self.dof_body_ids,
                                     dof_offsets=self.dof_offsets,
                                     device=self.device, 
                                     regen_pkl=self.cfg.regen_pkl)
    
    # Get values funcs
    # def get_motion_state(self, motion_ids, motion_times, get_lbp=False):
    #     root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos = \
    #         self._motion_lib.get_motion_state(motion_ids, motion_times, get_lbp)
    #     # dof_pos, dof_vel = self.reindex_dof_pos_vel(dof_pos, dof_vel)
    #     return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos
    
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
        self._motion_times[self._motion_times >= self._motion_lengths] = 0.

    def reindex_dof_pos_vel(self, dof_pos, dof_vel):
        dof_pos = reindex_motion_dof(dof_pos, self.dof_indices_sim, self.dof_indices_motion, self.valid_dof_body_ids)
        dof_vel = reindex_motion_dof(dof_vel, self.dof_indices_sim, self.dof_indices_motion, self.valid_dof_body_ids)
        return dof_pos, dof_vel


    