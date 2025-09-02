import numpy as np
import os
import yaml
import torch
import pickle
from typing import List, Dict, Union, Any

from poselib.skeleton.skeleton3d import SkeletonMotion
from trackerLab.utils.torch_utils.isaacgym import quat_rotate_inverse
from trackerLab.joint_id_caster import JointIdCaster
from .transforms.rot_2_dof import _local_rotation_to_dof

from trackerLab import TRACKERLAB_MOTION_CFG_DIR, TRACKERLAB_BUFFER_DIR, TRACKERLAB_RETARGETED_DATA_DIR
from .motion_lib import MotionLib
from .transforms.hdf5_loader import load_trajectories_from_hdf5

from dataclasses import dataclass

from poselib.motion_data.dof_pos_motion import DofposMotion


class MotionLibDofPos(MotionLib):
    def __init__(self, motion_file, id_caster, device, regen_pkl=False, **kwargs):
        self.motion_type = kwargs.get("motion_type")
        print("MotionLibDofPos initialized with motion type:", self.motion_type)
        super().__init__(motion_file, id_caster, device, regen_pkl, **kwargs)

    def _load_motions(self, motion_file):
        self._motions = []
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_files = []
        self._motion_difficulty = []

        total_len = 0.0
        
        motion_files, motion_weights, motion_difficulty, self.motion_description, extras = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        for idx in range(num_motion_files):
            curr_file:str = motion_files[idx]
            print("Loading {:d}/{:d} motion files: {:s}".format(idx + 1, num_motion_files, curr_file))
            fps = extras["fps"][idx]
            motion_fps = int(fps)
            curr_dt = 1.0 / motion_fps
            curr_difficulty = motion_difficulty[idx]
            curr_weight = motion_weights[idx]
            
            is_load = False
            if self.motion_type == "replay":
                if curr_file.endswith(".hdf5"):
                    print("Loading with hdf5 file.")
                    self.load_from_replayed_hdf5(curr_file, locals())
                    is_load = True
            elif self.motion_type == "GMR":
                if curr_file.endswith(".pkl"):
                    print("Loading with hdf5 file.")
                    self.load_from_GMR_pkl(curr_file, locals())
                    is_load = True
            if not is_load:
                raise NotImplementedError("Unsupported.")
                    
                
        self._motion_difficulty = torch.tensor(self._motion_difficulty, device=self._device, dtype=torch.float32)
        
        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()
        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

    def load_from_replayed_hdf5(self, curr_file: str, locs):
        fps = locs["motion_fps"]
        dt = locs["curr_dt"]
        
        trajs: List[Dict[str, torch.Tensor]] = load_trajectories_from_hdf5(curr_file)

        for traj in trajs:
            motion = DofposMotion.from_replay(traj)
            frames = motion.frames
            motion.tensorlize(self._device)
            self._motions.append(motion)
            self._motion_fps.append(fps)
            self._motion_num_frames.append(frames)
            self._motion_lengths.append(dt * frames)
            
            self._motion_dt.append(dt)
            self._motion_weights.append(locs["curr_weight"])
            self._motion_files.append(curr_file)
            self._motion_difficulty.append(locs["curr_difficulty"])

    def load_from_GMR_pkl(self, curr_file: str, locs):
        fps = locs["motion_fps"]
        dt = locs["curr_dt"]
        motion = DofposMotion.from_GMR(curr_file)
        frames = motion.frames
        motion.tensorlize(self._device)
        self._motions.append(motion)
        self._motion_fps.append(fps)
        self._motion_num_frames.append(frames)
        self._motion_lengths.append(dt * frames)
        
        self._motion_dt.append(dt)
        self._motion_weights.append(locs["curr_weight"])
        self._motion_files.append(curr_file)
        self._motion_difficulty.append(locs["curr_difficulty"])
        raise NotImplementedError("This method is undone")
            
    def _fetch_motion_files(self, motion_file):
        motion_files = []
        motion_weights = []
        motion_difficulty = []
        motion_description = []
        with open(os.path.join(motion_file), 'r') as f:
            motion_config = yaml.load(f, Loader=yaml.SafeLoader)

        dir_name = os.path.join(TRACKERLAB_RETARGETED_DATA_DIR, motion_config['motions']["root"])

        extras = {
            "fps": []
        }

        motion_list = motion_config['motions']
        for motion_entry in motion_list.keys():
            if motion_entry == "root":
                continue
            curr_file = motion_entry
            curr_weight = motion_config["motions"][motion_entry]['weight']
            curr_difficulty = motion_config["motions"][motion_entry]['difficulty']
            curr_description = motion_config["motions"][motion_entry]['description']
            fps = motion_config["motions"][motion_entry].get("fps")
            assert(curr_weight >= 0)

            curr_file = os.path.join(dir_name, curr_file)
            motion_weights.append(curr_weight)
            motion_files.append(os.path.normpath(curr_file))
            motion_difficulty.append(curr_difficulty)
            motion_description.append(curr_description)
            extras["fps"].append(fps)
            
        return motion_files, motion_weights, motion_difficulty, motion_description, extras

    def deserialize_motions(self, pkl_file):
        with open(pkl_file, 'rb') as inp:
            objects = pickle.load(inp)
        self._motions = []
        for motion in objects[0]:
            # motion.tensor = motion.tensor.to(self._device)
            # motion._skeleton_tree._parent_indices = motion._skeleton_tree._parent_indices.to(self._device)
            # motion._skeleton_tree._local_translation = motion._skeleton_tree._local_translation.to(self._device)
            # motion._rotation = motion._rotation.to(self._device)
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
    