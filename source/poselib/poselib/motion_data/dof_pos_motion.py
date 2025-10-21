import torch
from dataclasses import dataclass
import pickle
from poselib.retarget import AMASSLoader
from trackerLab.utils.torch_utils import quat_to_angle_axis

@dataclass
class DofposMotion:
    global_translation: torch.Tensor
    global_rotation: torch.Tensor
    local_rotation: torch.Tensor
    global_root_velocity: torch.Tensor
    global_root_angular_velocity: torch.Tensor
    dof_vels: torch.Tensor
    dof_poses: torch.Tensor
    
    def tensorlize(self, device):
        self.global_translation             = torch.tensor(self.global_translation, device=device)
        self.global_rotation                = torch.tensor(self.global_rotation, device=device)
        self.local_rotation                 = torch.tensor(self.local_rotation, device=device)
        self.global_root_velocity           = torch.tensor(self.global_root_velocity, device=device)
        self.global_root_angular_velocity   = torch.tensor(self.global_root_angular_velocity, device=device)
        self.dof_vels                       = torch.tensor(self.dof_vels, device=device)
        self.dof_poses                      = torch.tensor(self.dof_poses, device=device)
        
    @property
    def frames(self):
        return self.dof_poses.shape[0]
        
    @classmethod
    def from_replay(cls, traj:dict):
        body_link_pose_w = traj["body_link_pose_w"]
        joint_pos = traj["joint_pos"]
        joint_vel = traj["joint_vel"]
        root_lin_vel_b = traj["root_lin_vel_b"]
        root_ang_vel_b = traj["root_ang_vel_b"]
        body_com_quat_b = traj["body_com_quat_b"]

        motion = cls(
            global_translation=body_link_pose_w[..., :3],
            global_rotation=body_link_pose_w[..., 3:7],
            local_rotation=body_com_quat_b,
            global_root_velocity=root_lin_vel_b,
            global_root_angular_velocity=root_ang_vel_b,
            dof_vels=joint_vel,
            dof_poses=joint_pos
        )
        return motion
    
    @staticmethod
    def calc_root_vel(positions, fps):
        vel = (positions[1:] - positions[:-1]) * fps
        vel = torch.cat([vel, vel[-1:].clone()], dim=0)  # shape [T, J, 3]
        return vel
    
    @classmethod
    def from_GMR(cls, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            
        fps = data["fps"]
        root_pos = torch.tensor(data["root_pos"])
        root_rot = torch.tensor(data["root_rot"])
        # TODO
        dof_pos = torch.tensor(data["dof_pos"])
        local_body_pos = torch.tensor(data["local_body_pos"])
        
        local_body_pos += root_pos.unsqueeze(1)
        
        F, J = dof_pos.shape[:2]
        
        root_lin_vel = cls.calc_root_vel(root_pos, fps)
        root_ang_vel = cls.calc_root_vel(quat_to_angle_axis(root_rot)[1], fps)
        dof_vels = cls.calc_root_vel(dof_pos, fps)
        
        motion = cls(
            global_translation=local_body_pos,
            global_rotation=root_rot.unsqueeze(1),
            local_rotation=torch.zeros(F, J, 4),
            global_root_velocity=root_lin_vel,
            global_root_angular_velocity=root_ang_vel,
            dof_vels=dof_vels,
            dof_poses=dof_pos
        )
        return motion