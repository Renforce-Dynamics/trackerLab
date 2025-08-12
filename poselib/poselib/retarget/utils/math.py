import os
import numpy as np
import torch
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState


def axis_angle_to_quaternion_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = axis_angle / angle
    half_angle = angle * 0.5
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)
    return np.array([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, cos_half])


def normalize_tensor(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    return tensor / (norm + eps)


def safe_sqrt(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.clamp(tensor, min=eps))


def validate_quaternion(quat: torch.Tensor) -> torch.Tensor:
    # Ensure w component is positive (standard form)
    w_negative = quat[..., 3] < 0
    quat = torch.where(w_negative.unsqueeze(-1), -quat, quat)
    
    # Normalize
    return normalize_tensor(quat, dim=-1)


def batch_axis_angle_to_quaternion(axis_angles: torch.Tensor) -> torch.Tensor:
    angles = torch.norm(axis_angles, dim=-1, keepdim=True)
    
    # Handle zero angle case
    small_angle = angles < 1e-8
    angles = torch.where(small_angle, torch.ones_like(angles), angles)
    
    axes = axis_angles / angles
    half_angles = angles * 0.5
    sin_half = torch.sin(half_angles)
    cos_half = torch.cos(half_angles)
    
    quats = torch.cat([
        axes * sin_half,  # x, y, z
        cos_half         # w
    ], dim=-1)
    
    # Return identity quaternion for zero angle case
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=quats.device, dtype=quats.dtype)
    quats = torch.where(small_angle.expand_as(quats), identity_quat, quats)
    
    return validate_quaternion(quats)


def compute_bone_length(translation: torch.Tensor) -> torch.Tensor:
    return torch.norm(translation, dim=-1)


def compute_symmetry_error(left_pos: torch.Tensor, right_pos: torch.Tensor) -> float:
    mirrored_left = left_pos.clone()
    mirrored_left[..., 1] = -mirrored_left[..., 1] 
    
    error = torch.norm(mirrored_left - right_pos, dim=-1)
    return float(torch.mean(error))