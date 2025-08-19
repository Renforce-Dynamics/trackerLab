"""
motion manager for mujoco cpu version.

property: using as command input for policy
- loc_dof_pos: command for motion dof pos
- loc_root_pos: command for motion root pos

This is a CPU version adapted from the Isaac Lab motion manager.
Some features require environment data that may not be available in Mujoco context.
"""
import mujoco
from mujoco import MjModel, MjData
import numpy as np
import os
from typing import Optional, List, Dict, Any

try:
    # TODO: These imports may need to be adapted for CPU version
    from trackerLab.motion_buffer import MotionBuffer
    from trackerLab.motion_buffer.motion_lib import MotionLib
    from trackerLab.managers.joint_id_caster import JointIdCaster
    from trackerLab.utils.torch_utils import slerp
    from poselib import POSELIB_DATA_DIR
except ImportError as e:
    print(f"Warning: Could not import trackerLab modules: {e}")
    print("Some functionality may be limited")

class MotionManager:
    """
    Mujoco CPU version of motion manager.
    
    This class manages motion data and provides interface for policy commands.
    It stores two levels of data:
    1. Motion buffer: stores motion data with future motion intention
    2. Motion manager: manages current timestep motion data
    
    Some parameters may be related to Isaac Lab env and might be unavailable,
    marked with #TODO env comments.
    """
    
    # Private attributes for property setters
    _loc_dof_pos: Optional[np.ndarray] = None
    _loc_root_vel: Optional[np.ndarray] = None
    _loc_root_pos: Optional[np.ndarray] = None
    
    def __init__(self, cfg, model: MjModel, data: MjData):
        self.cfg = cfg
        self.model = model
        self.data = data
        self.device = 'cpu'
        
        # Initialize basic parameters
        self.speed_scale: float = getattr(cfg, "speed_scale", 1.0)
        self.static_motion: bool = getattr(cfg, "static_motion", False)
        self.motion_dt = self.model.opt.timestep * self.speed_scale
        self.obs_from_buffer: bool = getattr(cfg, "obs_from_buffer", True)
        self.loc_gen: bool = getattr(cfg, "loc_gen", True)
        self.reset_to_pose = getattr(cfg, "reset_to_pose", False)
        
        # Single environment - no need for parallel environments in Mujoco CPU
        self.num_envs = 1
        
        # Initialize joint mapping and motion data
        self.init_joint_mapping()
        self.init_motion_system()
        
        # Initialize local state variables
        self.loc_trans_base: Optional[np.ndarray] = None
        self.loc_root_pos: Optional[np.ndarray] = None  # Demo given root position
        self.loc_dof_pos: Optional[np.ndarray] = None
        self.loc_dof_vel: Optional[np.ndarray] = None
        self.loc_root_rot: Optional[np.ndarray] = None
        self.loc_root_vel: Optional[np.ndarray] = None
        self.loc_ang_vel: Optional[np.ndarray] = None
        
        # Initialize state
        self.loc_init_root_pos: Optional[np.ndarray] = None
        self.loc_init_demo_root_pos: Optional[np.ndarray] = None
        
        # Motion time tracking - single environment
        self.motion_time = 0.0
        self.motion_id = np.int32(0)  # Single motion ID
        self.motion_time_current = np.float32(0.0)  # Current motion time
    
    def compute(self):
        if self.loc_gen:
            self.loc_gen_state()
        if not self.static_motion:
            self._motion_buffer.update_motion_times()
            
            
    def calc_current_pose(self, env_ids):
        root_pos = self.data.qpos[:3].copy()  # First 3 DOF typically root position
        joint_pos = self.data.qpos[7:].copy()  # Remaining DOF typically joint positions
        joint_vel = self.data.qvel[7:].copy()  # Joint velocities
        
        # Get current motion state
        root_rot, root_vel, root_ang_vel, demo_root_pos, dof_pos_motion, dof_vel = \
            self.loc_root_rot, self.loc_root_vel, self.loc_ang_vel, \
            self.loc_root_pos,  self.loc_dof_pos, self.loc_dof_vel

        # Later we will reset to the target position and have another function
        # The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).
        # Shape: [num_envs, 7], is the pos (3) + rot (4)
        root_pose = np.concatenate((root_pos, root_rot), axis=-1)
        root_velocity = np.concatenate((root_vel, root_ang_vel), axis=-1)

        # On reset save init poses.
        self.loc_init_root_pos = root_pos.copy()
        self.loc_init_demo_root_pos = demo_root_pos.copy()
        
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
    
    
    def init_joint_mapping(self):
        """Initialize joint mapping between Mujoco and motion data"""
        try:
            # TODO env: This requires environment data that might not be available
            # For now, create basic mapping based on model
            self.joint_names = []
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name:
                    self.joint_names.append(joint_name)
            
            # TODO env: gym_joint_names and lab_joint_names mapping
            # This would typically come from JointIdCaster
            self.gym_joint_names = self.joint_names  # Simplified mapping
            self.lab_joint_names = self.joint_names
            
            # Create simple identity mapping for now
            self.shared_subset_gym = np.arange(len(self.joint_names))
            self.shared_subset_lab = np.arange(len(self.joint_names))
            
        except Exception as e:
            print(f"Warning: Could not initialize full joint mapping: {e}")
            # Fallback to basic setup
            self.joint_names = []
            self.gym_joint_names = []
            self.lab_joint_names = []
            self.shared_subset_gym = np.array([])
            self.shared_subset_lab = np.array([])
    
    def init_motion_system(self):
        """Initialize motion buffer and motion library"""
        try:
            # TODO env: This typically requires full env setup
            # For CPU version, we may need simplified motion loading
            
            if hasattr(self.cfg, 'motion_buffer_cfg'):
                # TODO env: MotionBuffer typically requires device and full env
                print("Warning: Motion buffer initialization requires full environment setup")
                self._motion_buffer = None
                self.motion_lib = None
            else:
                print("No motion buffer config provided")
                self._motion_buffer = None
                self.motion_lib = None
                
        except Exception as e:
            print(f"Warning: Could not initialize motion system: {e}")
            self._motion_buffer = None
            self.motion_lib = None
    
    def compute(self):
        """Main computation step"""
        if self.loc_gen:
            self.loc_gen_state()
        if not self.static_motion:
            self.update_motion_times()
    
    def update_motion_times(self):
        """Update motion timing for single environment"""
        self.motion_time += self.motion_dt
        self.motion_time_current += self.motion_dt
        
        # TODO: Handle motion looping and boundaries
        if self.motion_lib is not None:
            # Wrap motion times based on motion library
            pass
    
    def get_current_time(self):
        """Get current motion timing information for single environment"""
        motion_id = self.motion_id
        motion_time = self.motion_time_current
        
        if self.motion_lib is not None:
            # TODO env: This requires motion_lib implementation
            # f0l, f1l, blend = self.motion_lib.get_frame_idx(motion_id, motion_time)
            # For now, return dummy values
            f0l = motion_id
            f1l = motion_id + 1
            blend = 0.0
            return f0l, f1l, blend
        else:
            # Fallback when motion_lib is not available
            return motion_id, motion_id + 1, 0.0
    
    def calc_loc_terms(self, frame):
        """Calculate local terms at certain frame for single environment"""
        if self.motion_lib is not None:
            # TODO env: These require motion_lib data structures
            # loc_trans_base = self.motion_lib.trans_base[frame]
            # loc_root_rot = self.motion_lib.grs[frame, 0]
            # loc_root_pos = self.motion_lib.gts[frame, 0]
            # loc_local_rot = self.motion_lib.lrs[frame]
            # loc_dof_vel = self.motion_lib.dvs[frame]
            # loc_dof_pos = self.motion_lib.dof_pos[frame]
            # loc_root_vel = self.motion_lib.vels_base[frame]
            # loc_ang_vel = self.motion_lib.ang_vels_base[frame]
            
            # Return dummy data for now - single environment
            n_joints = len(self.joint_names) if self.joint_names else 0
            loc_trans_base = np.zeros(3)  # Single environment, no batch dimension
            loc_root_rot = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
            loc_root_pos = np.zeros(3)
            loc_local_rot = np.zeros((n_joints, 4)) if n_joints > 0 else np.zeros((1, 4))
            loc_dof_vel = np.zeros(n_joints) if n_joints > 0 else np.zeros(1)
            loc_dof_pos = np.zeros(n_joints) if n_joints > 0 else np.zeros(1)
            loc_root_vel = np.zeros(3)
            loc_ang_vel = np.zeros(3)
            
            return loc_trans_base, loc_root_rot, loc_root_pos, \
                   loc_dof_pos, loc_dof_vel, loc_root_vel, loc_ang_vel, loc_local_rot
        else:
            # Return zero arrays when motion_lib is not available - single environment
            n_joints = len(self.joint_names) if self.joint_names else 0
            return (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3),
                    np.zeros(n_joints) if n_joints > 0 else np.zeros(1), 
                    np.zeros(n_joints) if n_joints > 0 else np.zeros(1), 
                    np.zeros(3), np.zeros(3), 
                    np.zeros((n_joints, 4)) if n_joints > 0 else np.zeros((1, 4)))
    
    def loc_gen_state(self):
        """Generate local motion state for single environment"""
        f0l, f1l, blend = self.get_current_time()
        
        terms_0 = self.calc_loc_terms(f0l)
        terms_1 = self.calc_loc_terms(f1l)
        
        # Interpolate between frames
        terms = []
        for term0, term1 in zip(terms_0, terms_1):
            if term0 is not None and term1 is not None:
                # Simple linear interpolation (could use slerp for rotations)
                if hasattr(term0, 'shape') and len(term0.shape) > 0:
                    terms.append((term0 + term1) / 2.0)
                else:
                    terms.append(term0)  # Scalar case
            else:
                terms.append(term0)
        
        self.loc_trans_base, _, self.loc_root_pos, \
            _, loc_dof_vel, self.loc_root_vel, self.loc_ang_vel, _ = terms
        
        # TODO: Implement proper quaternion slerp for CPU
        self.loc_root_rot = terms_0[1]  # Use first frame rotation for now
        loc_local_rot = terms_0[7]
        
        # TODO env: These require motion_lib methods
        if self.motion_lib is not None:
            # loc_dof_pos = self.motion_lib._local_rotation_to_dof(loc_local_rot)
            loc_dof_pos = np.zeros(len(self.joint_names)) if self.joint_names else np.array([])
        else:
            loc_dof_pos = np.zeros(len(self.joint_names)) if self.joint_names else np.array([])
        
        self.loc_dof_pos = loc_dof_pos
        self.loc_dof_vel = loc_dof_vel
    
    def reset(self):
        """Reset motion manager state for single environment"""        
        # Reset motion timing
        self.motion_time_current = 0.0
        
        # TODO env: Reset motion ID - this typically requires motion library
        # self.motion_id = new_motion_id
        
        self.loc_gen_state()
        
        # TODO env: Return state for robot reset
        # This would require knowledge of robot state structure
        return None
    
    def get_current_pose(self):
        """Get current robot pose from Mujoco data for single environment"""        
        # Get current state from Mujoco
        root_pos = self.data.qpos[:3].copy()  # First 3 DOF typically root position
        root_quat = self.data.qpos[3:7].copy()  # Next 4 DOF typically root quaternion
        
        joint_pos = self.data.qpos[7:].copy()  # Remaining DOF are joints
        joint_vel = self.data.qvel[6:].copy()  # Skip root linear/angular velocity
        
        # TODO env: This structure matches Isaac Lab format but may need adaptation
        state = {
            "root_pose": np.concatenate([root_pos, root_quat]),
            "root_velocity": np.concatenate([self.data.qvel[:3], self.data.qvel[3:6]]),
            "joint_position": joint_pos,
            "joint_velocity": joint_vel,
        }
        return state
    
    @property
    def loc_height(self) -> float:
        """Get current height from root position"""
        if self.loc_root_pos is not None:
            if hasattr(self.loc_root_pos, '__getitem__'):
                return float(self.loc_root_pos[2])
            else:
                return 0.0
        else:
            return 0.0
    
    @property
    def loc_dof_pos(self) -> np.ndarray:
        """Get target joint positions"""
        if self._loc_dof_pos is not None:
            return self._loc_dof_pos
        else:
            # Fallback to current joint positions
            return self.data.qpos[7:] if len(self.data.qpos) > 7 else np.array([])
    
    @property
    def loc_root_vel(self) -> np.ndarray:
        """Get target root velocity"""
        if self._loc_root_vel is not None:
            return self._loc_root_vel
        else:
            # Fallback to current root velocity
            return self.data.qvel[:3]
    
    @property
    def loc_root_pos(self) -> np.ndarray:
        """Get target root position"""
        if self._loc_root_pos is not None:
            return self._loc_root_pos
        else:
            # Fallback to current root position
            return self.data.qpos[:3]
    
