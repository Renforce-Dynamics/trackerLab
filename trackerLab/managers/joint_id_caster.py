import os
import json
import torch
from typing import Dict, List
from trackerLab.motion_buffer.utils.jit_func import reindex_motion_dof
from poselib.retarget.retargeting_processor import RetargetingProcessor
from poselib.skeleton.skeleton3d import SkeletonTree
from poselib import POSELIB_DATA_DIR

MOTION_ALIGN_DIR = os.path.join(POSELIB_DATA_DIR, "motion_align")

# ====================================================================================
# Following descrips the sim joint subset for control

# If not having yaw pitch roll in h1, make the g1 pictch to match it
# To sum, the ankle, elbow need to be filled in the g1

def get_indices(list1: List[str], list2: List[str], strict=True) -> List[int]:
    if strict:
        return [list1.index(item) for item in list2]
    return [list1.index(item) for item in list2 if item in list1]

class JointMapper(object):
    def __init__(self, source_joints, target_joints, device, strict=True):
        self.source_joints = source_joints
        self.target_joints = target_joints
        self.device = device
        self.strict = strict
        self.init_mapping()
    
    def init_mapping(self):
        self.source_subset = [item for item in self.source_joints if item in self.target_joints]
        self.source2target = torch.tensor(get_indices(self.source_joints, self.target_joints, self.strict), 
                                                dtype=torch.long, device=self.device)
        self.target2source = torch.tensor(get_indices(self.target_joints, self.source_joints, False), 
                                                dtype=torch.long, device=self.device)

class JointIdCaster(object):
    def __init__(self, env, device, 
                 robot_type="H1"):
        self._env = env
        self.device = device
        self.robot_type = robot_type.lower()
        
        self.init_id_names()
        # self.init_id_cast()
    
    def init_id_names(self):
        # self.robot_skeleton:SkeletonTree = RetargetingProcessor.load_tpose(self.robot_type).skeleton_tree
        
        self.lab_joint_names = self._env.scene.articulations["robot"]._data.joint_names
        # self.lab_joint_names = [item + "_link" for item in self.lab_joint_names]
        # self.motion_joint_names = self.robot_skeleton.node_names
        self.valid_joints = self._env.scene.articulations["robot"]._data.joint_names
        
        self.init_gym_motion_offset()
        
    def init_id_cast(self):
        """
            Note following are calced results:
            self.gym2lab_dof_ids = [0, 5, 10, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19]
            self.gym2lab_dof_ids = torch.tensor(self.gym2lab_dof_ids, dtype=torch.long, device=self.device)
        """
        # Only using the gym2lab where the contrl model is equal
        self.gym2lab_map = JointMapper(self.gym_joint_names, self.lab_joint_names, self.device, True)
        self.lab2gym_map = JointMapper(self.lab_joint_names, self.gym_joint_names, self.device, False)
        self.labsub_map  = JointMapper(self.lab_joint_names, self.lab2gym_map.source_subset, self.device, False)

    def init_gym_motion_offset(self):
        self.align_cfg_path = os.path.join(MOTION_ALIGN_DIR, f"{self.robot_type}.json")
        with open(self.align_cfg_path, 'r') as f:
            config:dict = json.load(f)
        self.align_cfg = config
        dof_body_ids = config["dof_body_ids"]
        
        gym_joint_names = config["gym_joint_names"]
        self.gym_joint_names = gym_joint_names
        
        dof_offsets = torch.Tensor(config["dof_offsets"]).long().to(self.device)
        dof_indices_sim = torch.Tensor(config["dof_indices_sim"]).long().to(self.device)
        dof_indices_motion = torch.Tensor(config["dof_indices_motion"]).long().to(self.device)
        
        invalid_dof_id = config["invalid_dof_id"]
        valid_dof_body_ids = torch.ones(dof_offsets[-1], device=self.device, dtype=torch.bool)
        for idx in invalid_dof_id: valid_dof_body_ids[idx] = 0
        
        return gym_joint_names, dof_body_ids, dof_offsets, valid_dof_body_ids, dof_indices_sim, dof_indices_motion