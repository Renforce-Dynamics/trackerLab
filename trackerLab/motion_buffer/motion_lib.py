import numpy as np
import os
import yaml

from typing import List

from poselib.skeleton.skeleton3d import SkeletonMotion
from poselib.core.rotation3d import *
from poselib import POSELIB_DATA_DIR

from trackerLab.utils import torch_utils
from trackerLab.utils.torch_utils.isaacgym import normalize_angle, quat_rotate_inverse
import torch
import pickle

PKL_BUFFER_DIR = os.path.join(POSELIB_DATA_DIR, "pkl_buffer")
RETARGETED_DATA_DIR = os.path.join(POSELIB_DATA_DIR, "retargeted")

def calc_frame_blend(time, len, num_frames, dt):
    """
    Give a time in the range [0, len], return the frame index and blend factor.
    blend factor means how much to blend between the two frames.
    And next frame
    """
    phase = time / len
    phase = torch.clip(phase, 0.0, 1.0)

    frame_idx0 = (phase * (num_frames - 1)).long()
    frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
    blend = (time - frame_idx0 * dt) / dt

    return frame_idx0, frame_idx1, blend


USE_CACHE = False
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy
    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy

class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except: continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)  
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
        
        print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib():
    def __init__(self, motion_file, dof_body_ids, dof_offsets,
                 device, regen_pkl=False):
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._device = device
        
        print("*"*20 + " Loading motion library " + "*"*20)
        rela_dir = motion_file.split("/")[:-1]
        yaml_name = motion_file.split("/")[-1].split(".")[0]
        ext = os.path.splitext(motion_file)[1]
        pkl_dir = os.path.join(PKL_BUFFER_DIR, *rela_dir)
        os.makedirs(pkl_dir, exist_ok=True)
        pkl_file = os.path.join(pkl_dir, yaml_name + ".pkl")
        
        if not regen_pkl and ext == ".yaml" :
            try:
                self.deserialize_motions(pkl_file)
            except:
                print("No pkl file found, loading from yaml")
                print("Setting motion device: cpu")
                self._load_motions(motion_file)
                self.serialize_motions(pkl_file)
        else:
            self._load_motions(motion_file)
            if regen_pkl:
                self.serialize_motions(pkl_file)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(len(self._motions), dtype=torch.long, device=self._device)

        self._calc_frame_blend = calc_frame_blend
        
        self.load_terms()
        self.load_normed_terms()

    def load_terms(self):
        motions: List[SkeletonMotion] = self._motions
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        # self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)

        def lrs2vel(lrs, idx):
            dof_pos = self._local_rotation_to_dof(lrs)
            return self._dof_pos_to_dof_vel(dof_pos, self._motion_dt[idx])
        
        self.dvs = torch.cat([lrs2vel(m.local_rotation, idx) for idx, m in enumerate(motions)], dim=0).float().to(self._device)

    def load_normed_terms(self):
        root_pos = self.gts[:, 0:1, :]
        root_rot = self.grs[:, 0:1, :]
        num_joints = self.gts.shape[1]
        rel_pos = self.gts - root_pos
        self.vels_base = quat_rotate_inverse(root_rot.reshape(-1, 4), self.grvs.view(-1, 3)).view(self.grvs.shape)
        self.trans_base = quat_rotate_inverse(root_rot.expand(-1, num_joints, -1).reshape(-1, 4), rel_pos.view(-1, 3)).view(self.gts.shape)
        self.ang_vels_base = quat_rotate_inverse(root_rot.reshape(-1, 4), self.gravs.view(-1, 3)).view(self.grvs.shape)

        self.dof_pos = self._local_rotation_to_dof(self.lrs)

    def num_motions(self):
        return len(self._motions)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion(self, motion_id):
        return self._motions[motion_id]

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert(truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    # Property Terms

    def get_motion_difficulty(self, motion_ids):
        return self._motion_difficulty[motion_ids]
    
    def get_motion_files(self, motion_ids):
        return [self._motion_files[motion_id] for motion_id in motion_ids]
    
    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_fps(self, motion_ids):
        return self._motion_fps[motion_ids]
    
    def get_motion_num_frames(self, motion_ids):
        return self._motion_num_frames[motion_ids]
    
    def get_motion_description(self, motion_id):
        return self.motion_description[motion_id]
    
    def get_frame_idx(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]
        return f0l, f1l, blend
    
    # Utility functions
    
    def _load_motions(self, motion_file, *args, **kwargs):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motions_local_key_body_pos = []
        # self._motion_features = []
        self._motion_difficulty = []

        total_len = 0.0

        motion_files, motion_weights, motion_difficulty, self.motion_description = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for f in range(num_motion_files):
            curr_file = motion_files[f]
            print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, curr_file))
            curr_motion = SkeletonMotion.from_file(curr_file)

            motion_fps = int(curr_motion.fps)
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
 
            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            # Moving motion tensors to the GPU
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)                
            else:
                curr_motion.tensor = curr_motion.tensor.to(self._device)
                curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self._device)
                curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(self._device)
                curr_motion._rotation = curr_motion._rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)
            # self._motions_local_key_body_pos.append(curr_key_body_pos)

            curr_weight = motion_weights[f]
            self._motion_weights.append(curr_weight)
            self._motion_files.append(curr_file)

            curr_difficulty = motion_difficulty[f]
            self._motion_difficulty.append(curr_difficulty)

        self._motion_difficulty = torch.tensor(self._motion_difficulty, device=self._device, dtype=torch.float32)
        
        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        # self._motion_features = torch.stack(self._motion_features).squeeze(1)
        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)


        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))


    def serialize_motions(self, pkl_file):
        objects = [self._motions, 
                   self._motion_lengths, 
                   self._motion_weights, 
                   self._motion_fps, 
                   self._motion_dt, 
                   self._motion_num_frames, 
                   self._motion_files, 
                   self._motions_local_key_body_pos, 
                   self._motion_difficulty, 
                   self.motion_description]
        with open(pkl_file, 'wb') as outp:
            pickle.dump(objects, outp, pickle.HIGHEST_PROTOCOL)
        print("Saved to: ", pkl_file)

    def deserialize_motions(self, pkl_file):
        with open(pkl_file, 'rb') as inp:
            objects = pickle.load(inp)
        self._motions = []
        for motion in objects[0]:
            motion.tensor = motion.tensor.to(self._device)
            motion._skeleton_tree._parent_indices = motion._skeleton_tree._parent_indices.to(self._device)
            motion._skeleton_tree._local_translation = motion._skeleton_tree._local_translation.to(self._device)
            motion._rotation = motion._rotation.to(self._device)
            self._motions.append(motion)
        self._motion_lengths = objects[1].to(self._device)
        self._motion_weights = objects[2].to(self._device)
        self._motion_fps = objects[3].to(self._device)
        self._motion_dt = objects[4].to(self._device)
        self._motion_num_frames = objects[5].to(self._device)
        self._motion_files = objects[6]
        self._motions_local_key_body_pos = objects[7]
        self._motion_difficulty = objects[8].to(self._device)
        self.motion_description = objects[9]

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))
    
    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if (ext == ".yaml"):
            motion_files = []
            motion_weights = []
            motion_difficulty = []
            motion_description = []
            with open(os.path.join(POSELIB_DATA_DIR, "configs", motion_file), 'r') as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            dir_name = os.path.join(RETARGETED_DATA_DIR, motion_config['motions']["root"])

            motion_list = motion_config['motions']
            for motion_entry in motion_list.keys():
                if motion_entry == "root":
                    continue
                curr_file = motion_entry
                curr_weight = motion_config["motions"][motion_entry]['weight']
                curr_difficulty = motion_config["motions"][motion_entry]['difficulty']
                curr_description = motion_config["motions"][motion_entry]['description']
                assert(curr_weight >= 0)

                curr_file = os.path.join(dir_name, curr_file + ".npy")
                motion_weights.append(curr_weight)
                motion_files.append(os.path.normpath(curr_file))
                motion_difficulty.append(curr_difficulty)
                motion_description.append(curr_description)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]
            motion_difficulty = [0]
            motion_description = ["None"]
        return motion_files, motion_weights, motion_difficulty, motion_description

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)
        
        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels
    
    
    # Using local rotation for calcing dof_pos, indicating that the joints are near.
    
    def _local_rotation_to_dof(self, local_rot):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_q = local_rot[:, body_id]
                joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map
            elif (joint_size == 2):
                joint_q = local_rot[:, body_id]
                joint_exp_map = torch_utils.quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset:(joint_offset + joint_size)] = joint_exp_map[:, :2]
            elif (joint_size == 1):
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q)
                joint_theta = joint_theta * joint_axis[..., 1] # assume joint is always along y axis

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta
            else:
                print("Unsupported joint type")
                assert(False)

        return dof_pos

    def _dof_pos_to_dof_vel(self, local_dof, motion_dt, pad=True):
        """
        Convert DOF positions to DOF velocities.

        Args:
            local_dof (torch.Tensor): Shape [N, dofs], DOF positions over time.
            motion_dt (float): Time step between frames.
            pad (bool): Whether to pad the first velocity to maintain the same length.

        Returns:
            torch.Tensor: Shape [N, dofs], DOF velocities.
        """
        # Compute velocity from finite difference
        vel = (local_dof[1:, :] - local_dof[:-1, :]) / motion_dt.item()

        if pad:
            # Pad first row with zeros (or repeat first velocity if preferred)
            first_row = torch.zeros_like(vel[0:1, :])
            vel = torch.cat([first_row, vel], dim=0)

        return vel

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        dof_vel = torch.zeros([self._num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if (joint_size == 3):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel
            elif (joint_size == 2):
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset:(joint_offset + joint_size)] = joint_vel[:2]
            elif (joint_size == 1):
                assert(joint_size == 1)
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1] # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert(False)

        return dof_vel