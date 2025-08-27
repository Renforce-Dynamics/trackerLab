import os
import json
import torch
from typing import Dict, List
from poselib import POSELIB_MOTION_ALIGN_DIR
from trackerLab import TRACKERLAB_LABJOINTS_DIR

# ====================================================================================
# Following descrips the sim joint subset for control

# If not having yaw pitch roll in h1, make the g1 pictch to match it
# To sum, the ankle, elbow need to be filled in the g1

def get_indices(list1: List[str], list2: List[str], strict=True) -> List[int]:
    if strict:
        return [list1.index(item) for item in list2]
    return [list1.index(item) for item in list2 if item in list1]

class JointIdCaster(object):
    def __init__(
            self, 
            device, lab_joint_names,
            robot_type="H1",
            motion_align_cfg=None,
        ):
        self.lab_joint_names = lab_joint_names
        self.device = device
        self.robot_type = robot_type.lower()
        
        if isinstance(motion_align_cfg, dict):
            config = motion_align_cfg
        elif isinstance(motion_align_cfg, str):
            motion_align_cfg_path = motion_align_cfg
            with open(motion_align_cfg_path, 'r') as f:
                config:dict = json.load(f)
        else:
            print("[Warning] Using default motion align config which will be clean in future.")
            motion_align_cfg_path = os.path.join(POSELIB_MOTION_ALIGN_DIR, f"{self.robot_type}.json")
            with open(motion_align_cfg_path, 'r') as f:
                config:dict = json.load(f)

        self.align_cfg = config
        
        self.init_gym_motion_offset()
        self.init_id_cast()
        
    def init_id_cast(self):
        """
            Note following are calced results:
            self.gym2lab_dof_ids = [0, 5, 10, 1, 6, 11, 16, 2, 7, 12, 17, 3, 8, 13, 18, 4, 9, 14, 19]
            self.gym2lab_dof_ids = torch.tensor(self.gym2lab_dof_ids, dtype=torch.long, device=self.device)
        """
        # Only using the gym2lab where the contrl model is equal
        self.shared_subset_lab = [idx for idx, item in enumerate(self.lab_joint_names)if item in self.gym_joint_names]
        self.shared_subset_gym = [idx for idx, item in enumerate(self.gym_joint_names)if item in self.lab_joint_names]
        
        self.shared_subset_gym_names = [item for idx, item in enumerate(self.gym_joint_names)if item in self.lab_joint_names]
        self.shared_subset_lab_names = [item for idx, item in enumerate(self.lab_joint_names)if item in self.gym_joint_names]

        self.shared_subset_lab = torch.tensor(self.shared_subset_lab, dtype=torch.long, device=self.device)
        self.shared_subset_gym = torch.tensor(self.shared_subset_gym, dtype=torch.long, device=self.device)
        
        self.gym2lab_dof_ids = torch.tensor(get_indices(self.gym_joint_names, self.lab_joint_names, False), 
                                            dtype=torch.long, device=self.device)
        self.lab2gym_dof_ids = torch.tensor(get_indices(self.lab_joint_names, self.gym_joint_names, False), 
                                                dtype=torch.long, device=self.device)
        
        self.gymsub2lab_dof_ids = torch.tensor(get_indices(self.shared_subset_gym_names, self.lab_joint_names, False), 
                                            dtype=torch.long, device=self.device)

    @property
    def num_dof(self) -> int:
        return self.dof_offsets[-1]
    dof_body_ids       :torch.Tensor = None
    gym_joint_names    :torch.Tensor = None
    dof_offsets        :torch.Tensor = None
    dof_indices_sim    :torch.Tensor = None
    dof_indices_motion :torch.Tensor = None
    invalid_dof_id     :torch.Tensor = None
    valid_dof_body_ids :torch.Tensor = None

    def init_gym_motion_offset(self):
        config = self.align_cfg
        dof_body_ids = config["dof_body_ids"]
        
        gym_joint_names = config["gym_joint_names"]
        
        dof_offsets = torch.Tensor(config["dof_offsets"]).long().to(self.device)
        dof_indices_sim = torch.Tensor(config["dof_indices_sim"]).long().to(self.device)
        dof_indices_motion = torch.Tensor(config["dof_indices_motion"]).long().to(self.device)
        
        invalid_dof_id = config["invalid_dof_id"]
        valid_dof_body_ids = torch.ones(dof_offsets[-1], device=self.device, dtype=torch.bool)
        for idx in invalid_dof_id: valid_dof_body_ids[idx] = 0
        
        self.dof_body_ids = dof_body_ids
        self.gym_joint_names = gym_joint_names
        self.dof_offsets = dof_offsets
        self.dof_indices_sim = dof_indices_sim
        self.dof_indices_motion = dof_indices_motion
        self.invalid_dof_id = invalid_dof_id
        self.valid_dof_body_ids = valid_dof_body_ids
    
    # Utils
    
    def post_cast(self, dof_pos: torch.Tensor):
        reverse_dof_idx = self.align_cfg.get("reverse_dof", [])
        if reverse_dof_idx:
            dof_pos[:, reverse_dof_idx] = -dof_pos[:, reverse_dof_idx]
        
        return dof_pos
    
    def fill_2lab(self, source:torch.Tensor, target: torch.Tensor):
        """
        Move subset of gym and feed into the lab tensor.
        """
        source = source.clone()
        assert self.shared_subset_lab.shape[0] == target.shape[-1], "Cannot fill to lab tensor."
        source[:, self.shared_subset_lab] = target[:, :]
        return source
    
    def save_joint_details(self):
        ret_dir = os.path.join(TRACKERLAB_LABJOINTS_DIR, f"{self.robot_type}.json")
        ret_obj = {
            "lab_joint_names": self.lab_joint_names,
            "gym_joint_names": self.gym_joint_names,
            # "gym2lab_dof_ids": self.gym2lab_dof_ids.tolist(),
            # "lab2gym_dof_ids": self.lab2gym_dof_ids.tolist(),
            # "shared_subset_gym": self.shared_subset_gym.tolist(),
            # "shared_subset_lab": self.shared_subset_lab.tolist(),
            "shared_subset_gym_names": self.shared_subset_gym_names
        }
        with open(ret_dir, 'w') as f:
            json.dump(ret_obj, f, indent=4)
            
        print(f"Joint details saved to {ret_dir}.")