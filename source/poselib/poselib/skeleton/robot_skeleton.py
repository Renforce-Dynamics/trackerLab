import torch
import json
from typing import Union
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.core.rotation3d import quat_angle_axis, dof_to_axis_angle, quat_from_angle_axis

class RobotSkeleton(object):
    """
    Stateless for key feature transfer.
    """
    # skeleton_motion: SkeletonMotion
    skeleton_tree: SkeletonTree
    dof_axis_cast: dict
    dof_axis_tensor: torch.Tensor # [J, 3]
    def __init__(self, skeleton_tree, dof_axis_cfg: Union[str, dict], device):
        self.device = device
        self.dof_axis_cast = self.get_dof_axis_cast(dof_axis_cfg)
        self.skeleton_tree = skeleton_tree
        self.dof_axis_tensor = self.make_dof_axis_tensor(self.skeleton_tree.node_names, self.dof_axis_cast, self.device)
    
    @classmethod
    def from_robot_type(cls, robo_type, device):
        return cls()
    
    subset_idx = None
    def set_subset_cast(self, lab_joint_names, device):
        subset_idx = torch.tensor([lab_joint_names.index(node) for node in self.skeleton_tree.node_names if node in lab_joint_names], dtype=torch.long, device=device)
        self.subset_idx = subset_idx
    
    
    @staticmethod
    def get_dof_axis_cast(dof_axis_cfg: Union[str, dict]):
        if isinstance(dof_axis_cfg, str):
            with open(dof_axis_cfg, "rt") as f:
                dof_axis_cast = json.load(f)
        else:
            dof_axis_cast = dof_axis_cfg
        return dof_axis_cast
    
    @staticmethod
    def make_dof_axis_tensor(node_names, dof_axis_cast, device):
        dof_axis_tensor = torch.tensor([dof_axis_cast[node] for node in node_names], dtype=torch.float32, device=device)  
        return dof_axis_tensor
    
    def to_pose_axis_angle(self, skeleton_motion: SkeletonMotion) -> torch.Tensor:
        """
        Convert local_rot (quaternion) of the skeleton motion to axis-angle representation.
        
        Returns:
            pose_aa (torch.Tensor): [N, J, 4] axis-angle representation,
                                    where last dim is (axis_x, axis_y, axis_z, angle)
        """
        quat = skeleton_motion.local_rot  # [N, J, 4]
        angle, axis = quat_angle_axis(quat)
        pose_aa = torch.cat([axis, angle], dim=-1)  # [N, J, 4]
        return pose_aa

    def to_dof_pose(self, skeleton_motion:SkeletonMotion) -> torch.Tensor:
        """
        Project axis-angle pose onto each joint's DOF axis to obtain scalar DOF values.
        
        Returns:
            dof_pose (torch.Tensor): [N, J] scalar DOF poses (radians)
        """
        pose_aa = self.to_pose_axis_angle(skeleton_motion)  # [N, J, 4]
        axis, angle = pose_aa[..., :3], pose_aa[..., 3:]
        dof_axis = self.dof_axis_tensor.unsqueeze(0)  # [1, J, 3]

        # dot product between rotation axis and dof axis
        proj = (axis * dof_axis).sum(dim=-1, keepdim=True)  # [N, J, 1]
        dof_pose = (proj * angle).squeeze(-1)  # [N, J]

        return dof_pose
    
    def from_hdf5(self, hdf5_path, device):
        import h5py
        dof_axis_tensor = self.dof_axis_tensor
        
        with h5py.File(hdf5_path, "r") as f:
            _joint_pos = torch.tensor(f["joint_pos"][:], dtype=torch.float32, device=device)  # [T, J]
                    
        joint_pos = _joint_pos[:, self.subset_idx]
            
        axis_angle:torch.Tensor = dof_to_axis_angle(joint_pos, dof_axis_tensor)
        axes = axis_angle[..., :3].reshape(-1, 3)
        angles = axis_angle[..., 3:].reshape(-1, 1)
        
        quat = quat_from_angle_axis(angles, axes)
        return
    
    def from_robot_state(self, _joint_pos, root_link_pose_w, device):
        
        root_trans = root_link_pose_w[:, :3]
        root_rot = root_link_pose_w[:, 3:]
        
        joint_pos = _joint_pos[:, self.subset_idx]
        
        axis_angle:torch.Tensor = dof_to_axis_angle(joint_pos, self.dof_axis_tensor)
        axes = axis_angle[..., :3].reshape(-1, 3)
        angles = axis_angle[..., 3:].reshape(-1, 1)
        quat = quat_from_angle_axis(angles, axes).reshape(-1, 4)
        
        ret_tensor = torch.cat([root_rot, quat, root_trans], dim=1)
        
        return ret_tensor