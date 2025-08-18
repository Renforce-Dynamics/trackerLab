import torch
try:
    from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
except ModuleNotFoundError:
    print("ImportError: Unable to import ManagerBase or ManagerTermBase. Please check the module structure.")
    ManagerBase = object
from trackerLab.motion_buffer import MotionBuffer
from trackerLab.motion_drawer import MotionDrawer
# from isaaclab.assets import Articulation, RigidObject

# from isaaclab.utils.math import quat_rotate_inverse, quat_rotate
# from trackerLab.utils.torch_utils import quat_rotate, quat_rotate_inverse, euler_from_quaternion
from trackerLab.utils.torch_utils import slerp 

from typing import List
from ..joint_id_caster import JointIdCaster

class MotionManager(ManagerBase):
    """
    This will manage the motion info and visuallize the debug info of it.
    
    We storage two level of data and mechanism:
    1. The motion buffer will store the motion data with future motion intension. 
        - The motion buffer is in gym order and interact with motion lib.
        - Activate by the cfg.obs_from_buffer.
    2. The motion manager will manage the motion data and provide the interface for the env.
        - The motion manager is in lab order and interact with the buffer.
        - The motion manager contains the basic data at current time step.
        - Activate by the cfg.loc_gen.
        
    static_motion: bool = self.cfg.static_motion will determine if the motion buffer will update the motion times.
    
    """
    def __init__(self, cfg, env, device):
        super().__init__(cfg, env)
        # Init basic params
        self.speed_scale: float = getattr(self.cfg, "speed_scale", 1.0)
        self.static_motion: bool = self.cfg.static_motion
        self.motion_dt = env.step_dt * self.speed_scale
        self.obs_from_buffer: bool = getattr(self.cfg, "obs_from_buffer", True)
        self.loc_gen: bool = getattr(self.cfg, "loc_gen", True)
        self.reset_to_pose = getattr(self.cfg, "reset_to_pose", False)
        
        self.init_id_cast()

        self._motion_buffer = MotionBuffer(cfg.motion_buffer_cfg, env.num_envs, self.motion_dt, device, id_caster=self.id_caster)
        self.motion_lib = self._motion_buffer._motion_lib
    
        
    def init_id_cast(self):
        self.id_caster = JointIdCaster(env = self._env, device = self.device, robot_type = self.cfg.robot_type)
        self.lab_joint_names = self.id_caster.lab_joint_names # self._env.scene.articulations["robot"]._data.joint_names
        self.gym_joint_names = self.id_caster.gym_joint_names
        
        self.shared_subset_gym = self.id_caster.shared_subset_gym
        self.shared_subset_lab = self.id_caster.shared_subset_lab

    @property
    def gym2lab_dof_ids(self):
        return self.id_caster.gym2lab_dof_ids
    
    # @property
    # def lab2gym_dof_ids(self):
    #     return self.id_caster.lab2gym_dof_ids

    def compute(self):
        if self.loc_gen:
            self.loc_gen_state()
        if not self.static_motion:
            self._motion_buffer.update_motion_times()


    def calc_current_pose(self, env_ids):
        robot = self._env.scene["robot"]
        
        root_pos = robot.data.root_pos_w[env_ids].clone()
        joint_pos = robot.data.joint_pos[env_ids].clone()
        joint_vel = robot.data.joint_vel[env_ids].clone()
        
        # Get current motion state
        root_rot, root_vel, root_ang_vel, demo_root_pos, dof_pos_motion, dof_vel = \
            self.loc_root_rot[env_ids], self.loc_root_vel[env_ids], self.loc_ang_vel[env_ids], \
            self.loc_root_pos[env_ids],  self.loc_dof_pos[env_ids], self.loc_dof_vel[env_ids]

        # Later we will reset to the target position and have another function
        # The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        # Shape: [num_envs, 7], is the pos (3) + rot (4)
        root_pose = torch.cat((root_pos, root_rot), dim=-1)
        root_velocity = torch.cat((root_vel, root_ang_vel), dim=-1)
        
        # On reset save init poses.
        self.loc_init_root_pos = root_pos.clone()
        self.loc_init_demo_root_pos = demo_root_pos.clone()
        
        joint_pos = self.id_caster.fill_2lab(joint_pos, dof_pos_motion)
        joint_vel = self.id_caster.fill_2lab(joint_vel, dof_pos_motion)
        
        state = {
            "articulation": {
                "robot": {
                    "root_pose": root_pose,
                    "root_velocity": root_velocity,
                    "joint_position": joint_pos,
                    "joint_velocity": joint_vel,
                },
            }
        }
        return state

    def reset(self, env_ids):
        self._motion_buffer.update_motion_ids(env_ids)
        self.loc_gen_state(env_ids)
        state = self.calc_current_pose(env_ids)
        if not self.reset_to_pose:
            state = None
        return state

    def get_current_time(self, env_ids):
        motion_ids   = self._motion_buffer._motion_ids[env_ids].reshape(-1)
        motion_times = self._motion_buffer._motion_times[env_ids].reshape(-1)
        f0l, f1l, blend = self.motion_lib.get_frame_idx(motion_ids, motion_times)
        return f0l, f1l, blend

    # Local based terms
        
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
    
    def loc_gen_state(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self.device)
        
        f0l, f1l, blend = self.get_current_time(env_ids)
        
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
        # loc_dof_vel = self.motion_lib._local_rotation_to_dof_vel(terms_0[1], terms_1[1])

        # if loc_dof_pos.shape[-1] != loc_dof_vel.shape[-1]:
        #     loc_dof_vel = torch.zeros_like(loc_dof_pos)
        #     if getattr(self, "__warned_dof_vel_unmatch", True):
        #         print("[WARNNING]: the dof vel and dof is unmatched, this is caused by unmatched retargeting cfg.")
        #         setattr(self, "__warned_dof_vel_unmatch", False)

        loc_dof_pos, loc_dof_vel = self._motion_buffer.reindex_dof_pos_vel(loc_dof_pos, loc_dof_vel)
        self.loc_dof_pos, self.loc_dof_vel = loc_dof_pos[:, self.gym2lab_dof_ids], loc_dof_vel[:, self.gym2lab_dof_ids]
        
    def get_subset_real(self, source:torch.Tensor):
        return source[:, self.shared_subset_lab]
        # return joint_pos[:, self.lab2gym_dof_ids][:, self.gym2lab_dof_ids]
        
    # Super methods
    def _prepare_terms(self):
        # self.loc_gen_state()
        pass
    
    @property
    def active_terms(self):
        return [
            k for k, v in vars(self).items()
            if v is not None and not callable(v) and k.startswith("loc_")
        ]