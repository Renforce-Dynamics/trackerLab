
import os
import argparse
import json
import torch
from trackerLab.motion_buffer.utils.dataset import generate_sliding_trajs, get_edge_index_cmu
from trackerLab.motion_buffer.motion_lib import MotionLib
from trackerLab.utils.animation import animate_skeleton
from poselib import POSELIB_DATA_DIR
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
_dof_offsets = [0, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 19, 20, 21]
_key_body_ids = torch.tensor([3, 6, 9, 12], device=device)

motion_dir = os.path.join(POSELIB_DATA_DIR, "configs")
res_dir = "/workspace/isaaclab/logs/motion_gen"

import yaml
    
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

file_name = "skill_graph.yaml"
motion_lib, _ = mk_motion_lib(file_name)

from trackerLab.utils.torch_utils.isaacgym import quat_rotate, quat_rotate_inverse

from dataclasses import dataclass

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
        tar = quat_rotate_inverse(root_rot.expand(-1, 14, -1).reshape(-1, 4), rel_pos.view(-1, 3)).view(gts.shape)
        vels = quat_rotate_inverse(root_rot.reshape(-1, 4), grvs.view(-1, 3)).view(grvs.shape)
        tar[..., -1:] += root_pos[..., -1:]
        return cls(tar=tar, vels=vels)

from collections import defaultdict
import random
MAX_WEIGHT = 1e9

@dataclass
class SkillData:
    name: str
    motion_id: int
    start_frame: int
    end_frame: int

    desc: str = None
    # root_pos: torch.Tensor
    # root_rot: torch.Tensor
    # rel_pos: torch.Tensor
    # grvs: torch.Tensor

    # num_patches: int
    @property
    def num_patches(self):
        return int((self.end_frame - self.start_frame) / 10)

    def get_tar(self, motions):
        return motions[self.motion_id].tar[self.start_frame:self.end_frame, :, :], \
            motions[self.motion_id].grvs[self.start_frame:self.end_frame, :, :]

@dataclass
class FSMEdge:
    from_node: int
    to_node: int
    blend: int = 0
    blend_type: str = "linear"

def build_fsm(skill_list: list[SkillData], edges: list[FSMEdge], num_blend_frame=10):
    N = len(skill_list)
    W = torch.full((N, N), MAX_WEIGHT, device=device)
    for edge in edges:
        W[edge.from_node, edge.to_node] = edge.blend
    return W

def generate_fsm_sequence(fsm_graph, start_node, num_skills=10):
    sequence = [start_node]
    current = start_node
    for _ in range(num_skills - 1):
        next_nodes = fsm_graph[current]
        if not next_nodes:
            break
        current = random.choice(next_nodes)
        sequence.append(current)
    return sequence

def test_fsm():
    motions = [
        NormedMotion.from_motion(motion) for motion in motion_lib._motions
    ]
    patch_len = 10
    max_patches_per_skill = 10
    buffer_size = 300
    
    obs_ts = torch.empty((buffer_size, 14, 3), device=device)
    obs_vels = torch.empty((buffer_size, 3), device=device)
    
    basic_skill_raw_datas = [
        SkillData("stand", 3, 10, 30),
        SkillData("walk_1_step_right", 0, 30, 70),
        SkillData("walk_1_step_left",  0, 70, 90),
        SkillData("walk_2_step_right", 1, 30, 60),
        SkillData("walk_2_step_left",  1, 60, 90),
        SkillData("speed_up_run",  1, 90, 150),
    ]
    
    num_skills = len(basic_skill_raw_datas)
    
    skill_lib = torch.zeros(
        (num_skills, max_patches_per_skill * patch_len, 14, 3), 
        device=device
    )
    
    for i, skill in enumerate(basic_skill_raw_datas):
        data: NormedMotion = motions[skill.motion_id]
        L = skill.end_frame - skill.start_frame
        skill_lib[i, :L, ...].copy_(
            data.tar[skill.start_frame:skill.end_frame, ...]
        )
        skill_lib[i, :L, ...].copy_(
            data.vels[skill.start_frame:skill.end_frame, ...]
        )
    
    skill_lib.reshape((num_skills, max_patches_per_skill, patch_len, 14, 3))
    
    fsm_edges = [
        FSMEdge(0, 1, blend=0),
        FSMEdge(0, 2, blend=0),
        FSMEdge(1, 3, blend=0),
        FSMEdge(2, 4, blend=0),
        FSMEdge(3, 6, blend=0),
        FSMEdge(4, 6, blend=0),
        FSMEdge(6, 0, blend=0),
    ]
    
    fsm_graph = defaultdict(list)
    for edge in fsm_edges:
        fsm_graph[edge.from_node].append(edge.to_node)
    
    fsm_sequence = generate_fsm_sequence(fsm_graph, start_node=0, num_skills=12)
    
    ptr = 0
    for node in fsm_sequence:
        skill = basic_skill_raw_datas[node]
        data: NormedMotion = motions[skill.motion_id]
        L = skill.end_frame - skill.start_frame
        cut = max(ptr + L - buffer_size, 0)
        obs_ts[ptr:ptr + L - cut, ...].copy_(
            data.tar[skill.start_frame:skill.end_frame - cut, ...]
        )
        obs_vels[ptr:ptr + L - cut, ...].copy_(
            data.vels[skill.start_frame:skill.end_frame - cut, ...]
        )
        ptr += L
        if ptr >= buffer_size:
            break
    
    animate_skeleton(
        obs_ts, edge_index, obs_vels, 
        desc="fsm_looped",
        interval=30,
        save_path="./logs/fsm_looped.mp4"
    )
    
test_fsm()