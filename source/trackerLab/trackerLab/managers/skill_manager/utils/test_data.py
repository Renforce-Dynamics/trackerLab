
import os
import argparse
import json
import torch

import yaml
from trackerLab.motion_buffer.utils.dataset import generate_sliding_trajs, get_edge_index_cmu
from trackerLab.motion_buffer.motion_lib import MotionLib
from trackerLab.utils.animation import animate_skeleton
from poselib import POSELIB_DATA_DIR
from typing import Tuple

from dataclasses import dataclass
from trackerLab.utils.torch_utils.isaacgym import quat_rotate, quat_apply_inverse

def skill_test_get_motion():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    _dof_offsets = [0, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 19, 20, 21]
    _key_body_ids = torch.tensor([3, 6, 9, 12], device=device)

    motion_dir = os.path.join(POSELIB_DATA_DIR, "configs")
    res_dir = "/workspace/isaaclab/logs/motion_gen"

        
    def mk_motion_lib(motion_file) -> Tuple[MotionLib, str]:
        motion_name = motion_file[:-5]
        res_path = os.path.join(res_dir, motion_name)
        os.makedirs(res_path, exist_ok=True)

        if motion_file.endswith(".npy"):
            tar_path = motion_file
        else:
            tar_path = os.path.join(motion_dir, motion_file)
        motion_lib = MotionLib(
            motion_file=tar_path,
            dof_body_ids=_dof_body_ids,
            dof_offsets=_dof_offsets,
            key_body_ids=_key_body_ids.cpu().numpy(),
            device=device,
            regen_pkl=True
        )
        return motion_lib, res_path

    edge_index = get_edge_index_cmu().to(device)

    file_name = "skill_graph2.yaml"
    motion_lib, _ = mk_motion_lib(file_name)
    return motion_lib, edge_index, res_dir

@dataclass
class NormedMotion:
    tar: torch.Tensor
    vels: torch.Tensor
    
    @classmethod
    def from_motion(cls, motion):
        gts = motion.global_translation
        grs = motion.global_rotation
        grvs = motion.global_root_velocity
        
        root_pos = gts[:, 0:1, :]
        root_rot = grs[:, 0:1, :]
        rel_pos = gts - root_pos
        tar = quat_apply_inverse(root_rot.expand(-1, 14, -1).reshape(-1, 4), rel_pos.view(-1, 3)).view(gts.shape)
        vels = quat_apply_inverse(root_rot.reshape(-1, 4), grvs.view(-1, 3)).view(grvs.shape)
        tar[..., -1:] += root_pos[..., -1:]
        return cls(tar=tar, vels=vels)