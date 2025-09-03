import numpy as np
import os
import yaml
import torch
import pickle
from typing import List

from poselib.skeleton.skeleton3d import SkeletonMotion
from trackerLab.utils.torch_utils.isaacgym import quat_rotate_inverse
from trackerLab.joint_id_caster import JointIdCaster
from .transforms.rot_2_dof import _local_rotation_to_dof

from trackerLab import TRACKERLAB_MOTION_CFG_DIR, TRACKERLAB_BUFFER_DIR, TRACKERLAB_RETARGETED_DATA_DIR

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
    def __init__(
            self, 
            motion_file, 
            id_caster: JointIdCaster,
            device, 
            regen_pkl=False,
            **kwargs
        ):
        self.id_caster:JointIdCaster = id_caster
        self.dof_body_ids = id_caster.dof_body_ids
        self.dof_offsets = id_caster.dof_offsets
        self.num_dof = id_caster.num_dof
        
        self._device = device
        self.regen_pkl = regen_pkl
        
        motion_file = os.path.join(TRACKERLAB_MOTION_CFG_DIR, motion_file)
        
        print("*"*20 + " Loading motion library " + "*"*20)
        rela_dir = motion_file.split("/")[-2]
        yaml_name = motion_file.split("/")[-1].split(".")[0]
        ext = os.path.splitext(motion_file)[1]
        pkl_dir = os.path.join(TRACKERLAB_BUFFER_DIR, rela_dir)
        os.makedirs(pkl_dir, exist_ok=True)
        pkl_file = os.path.join(pkl_dir, yaml_name + ".pkl")
        
        if not regen_pkl:
            try:
                self.deserialize_motions(pkl_file)
            except:
                print("No pkl buffer file found, loading from yaml")
                self._load_motions(motion_file)
                self.serialize_motions(pkl_file)
        else:
            self._load_motions(motion_file)
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
        
        # Here we have two ways of calcing dvs:
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        self.dps = torch.cat([m.dof_poses for m in motions], dim=0).float().to(self._device)

    # These values could be calced from the original data
    def load_normed_terms(self):
        root_pos = self.gts[:, 0:1, :]
        root_rot = self.grs[:, 0:1, :]
        num_joints = self.gts.shape[1]
        rel_pos = self.gts - root_pos
        self.vels_base = quat_rotate_inverse(root_rot.reshape(-1, 4), self.grvs.view(-1, 3)).view(self.grvs.shape)
        self.trans_base = quat_rotate_inverse(root_rot.expand(-1, num_joints, -1).reshape(-1, 4), rel_pos.view(-1, 3)).view(self.gts.shape)
        self.ang_vels_base = quat_rotate_inverse(root_rot.reshape(-1, 4), self.gravs.view(-1, 3)).view(self.grvs.shape)

        # self.dof_pos = self._local_rotation_to_dof(self.lrs)

    # Utility functions
    def _fill_motions(self, curr_motion, curr_dt):
        # Exbody mode for calc dof pos
        curr_dof_pos = _local_rotation_to_dof(self.id_caster, curr_motion.local_rotation, self._device)
        curr_motion.dof_poses = curr_dof_pos
        
        self.id_caster.post_cast(curr_dof_pos)
        
        # Two way of calc pos and vel
        curr_dof_vels = self._dof_pos_to_dof_vel(curr_dof_pos, curr_dt)
        # Calc dof vel with old api
        # curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
        curr_motion.dof_vels = curr_dof_vels

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
    
    def _load_motions(self, motion_file):
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
            curr_motion:SkeletonMotion = SkeletonMotion.from_file(curr_file)

            motion_fps = int(curr_motion.fps)
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            self._motion_fps.append(motion_fps)
            self._motion_dt.append(curr_dt)
            self._motion_num_frames.append(num_frames)
            
            self._fill_motions(curr_motion, curr_dt)
            
            # Moving motion tensors to the GPU
            if USE_CACHE:
                curr_motion = DeviceCache(curr_motion, self._device)                
            else:
                curr_motion.tensor = curr_motion.tensor.to(self._device)
                curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(self._device)
                curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(self._device)
                curr_motion._rotation = curr_motion._rotation.to(self._device)
                curr_motion._comp_local_rotation = curr_motion._comp_local_rotation.to(self._device)

            self._motions.append(curr_motion)
            self._motion_lengths.append(curr_len)

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
        self._motion_difficulty = objects[7].to(self._device)
        self.motion_description = objects[8]

        num_motions = self.num_motions()
        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))
    
    def _fetch_motion_files(self, motion_file):
        motion_files = []
        motion_weights = []
        motion_difficulty = []
        motion_description = []
        with open(os.path.join(motion_file), 'r') as f:
            motion_config = yaml.load(f, Loader=yaml.SafeLoader)

        dir_name = os.path.join(TRACKERLAB_RETARGETED_DATA_DIR, motion_config['motions']["root"])

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
        return motion_files, motion_weights, motion_difficulty, motion_description

    def _fetch_single_motion_file(motion_file):
        motion_files = [motion_file]
        motion_weights = [1.0]
        motion_difficulty = [0]
        motion_description = ["None"]
        return motion_files, motion_weights, motion_difficulty, motion_description

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _dof_pos_to_dof_vel(self, local_dof, motion_dt, pad=True):
        # Compute velocity from finite difference
        vel = (local_dof[1:, :] - local_dof[:-1, :]) / motion_dt

        if pad:
            # Pad first row with zeros (or repeat first velocity if preferred)
            first_row = torch.zeros_like(vel[0:1, :])
            vel = torch.cat([first_row, vel], dim=0)

        return vel
    
    
    
