import os
import json
import numpy as np
import torch
from collections import OrderedDict
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion
from poselib.core.rotation3d import quat_mul, quat_rotate


def save_skeleton_state_to_file(skeleton_state, file_path):
    
    def to_numpy(tensor):
        if hasattr(tensor, 'numpy'):
            return tensor.numpy()
        elif hasattr(tensor, 'detach'):
            return tensor.detach().numpy()
        else:
            return np.array(tensor)
    
    skeleton_dict = OrderedDict([
        ('rotation', {
            'arr': to_numpy(skeleton_state.global_rotation).astype(np.float32),
            'context': {'dtype': 'float32'}
        }),
        ('root_translation', {
            'arr': to_numpy(skeleton_state.root_translation).astype(np.float32),
            'context': {'dtype': 'float32'}
        }),
        ('skeleton_tree', OrderedDict([
            ('node_names', skeleton_state.skeleton_tree.node_names),
            ('parent_indices', {
                'arr': to_numpy(skeleton_state.skeleton_tree.parent_indices).astype(np.int64),
                'context': {'dtype': 'int64'}
            }),
            ('local_translation', {
                'arr': to_numpy(skeleton_state.skeleton_tree.local_translation).astype(np.float32),
                'context': {'dtype': 'float32'}
            })
        ])),
        ('is_local', True),
        ('__name__', 'SkeletonState')
    ])

    np.save(file_path, skeleton_dict, allow_pickle=True)
    print(f"💾 SkeletonState saved in compatible format: {file_path}")


def apply_height_adjustment(motion: SkeletonMotion, target_height: float = 0.0, method: str = "ground_align") -> SkeletonMotion:
    local_rotation = motion.local_rotation
    root_translation = motion.root_translation.clone()
    
    if method == "ground_align":
        # Align to ground
        global_pos = motion.global_translation
        min_z = torch.min(global_pos[..., 2])
        root_translation[:, 2] += -min_z + target_height
    
    elif method == "fixed_height":
        # Fixed height
        root_translation[:, 2] = target_height
    
    elif method == "relative":
        # Relative adjustment
        root_translation[:, 2] += target_height
    
    # Rebuild motion
    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, local_rotation, root_translation, is_local=True
    )
    
    return SkeletonMotion.from_skeleton_state(new_state, fps=motion.fps)


def apply_scale_transform(motion: SkeletonMotion, scale_factor: float) -> SkeletonMotion:

    # Scale root translation
    root_translation = motion.root_translation * scale_factor
    
    # Rebuild motion (local rotation remains unchanged)
    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, motion.local_rotation, root_translation, is_local=True
    )
    
    return SkeletonMotion.from_skeleton_state(new_state, fps=motion.fps)


def apply_rotation_transform(motion: SkeletonMotion, rotation_quat: torch.Tensor) -> SkeletonMotion:

    # Rotate root translation
    root_translation = motion.root_translation.clone()
    for i in range(root_translation.shape[0]):
        root_translation[i] = quat_rotate(rotation_quat.unsqueeze(0), root_translation[i].unsqueeze(0)).squeeze(0)
    
    # Rotate root rotation
    local_rotation = motion.local_rotation.clone()
    root_rotation = local_rotation[:, 0]  # Assume root joint is the first
    for i in range(root_rotation.shape[0]):
        local_rotation[i, 0] = quat_mul(rotation_quat.unsqueeze(0), root_rotation[i].unsqueeze(0)).squeeze(0)
    
    # Rebuild motion
    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, local_rotation, root_translation, is_local=True
    )
    
    return SkeletonMotion.from_skeleton_state(new_state, fps=motion.fps)


def smooth_motion(motion: SkeletonMotion, window_size: int = 5) -> SkeletonMotion:

    if window_size <= 1:
        return motion
    
    # Smooth root translation
    root_translation = motion.root_translation.clone()
    num_frames = root_translation.shape[0]
    
    for i in range(num_frames):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(num_frames, i + window_size // 2 + 1)
        root_translation[i] = torch.mean(root_translation[start_idx:end_idx], dim=0)
    
    # Rebuild motion (local rotation remains unchanged to avoid interpolation complexity)
    new_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree, motion.local_rotation, root_translation, is_local=True
    )
    
    return SkeletonMotion.from_skeleton_state(new_state, fps=motion.fps)


def extract_motion_statistics(motion: SkeletonMotion) -> dict:

    global_pos = motion.global_translation
    root_translation = motion.root_translation
    
    stats = {
        'num_frames': int(global_pos.shape[0]),
        'num_joints': int(global_pos.shape[1]),
        'fps': float(motion.fps),
        'duration': float(global_pos.shape[0] / motion.fps),
        'height_range': {
            'min': float(torch.min(global_pos[..., 2]).item()),
            'max': float(torch.max(global_pos[..., 2]).item()),
            'mean': float(torch.mean(global_pos[..., 2]).item())
        },
        'root_displacement': {
            'total': float(torch.norm(root_translation[-1] - root_translation[0]).item()),
            'max_velocity': float(torch.max(torch.norm(root_translation[1:] - root_translation[:-1], dim=-1)).item())
        },
        'joint_names': list(motion.skeleton_tree.node_names)
    }
    
    return stats


def validate_motion_quality(motion: SkeletonMotion, checks: list = None) -> dict:

    if checks is None:
        checks = ['nan_check', 'range_check', 'smoothness_check']
    
    results = {}
    
    if 'nan_check' in checks:
        # Check for NaN values
        global_pos = motion.global_translation
        local_rot = motion.local_rotation
        root_trans = motion.root_translation
        
        has_nan_pos = bool(torch.isnan(global_pos).any().item())
        has_nan_rot = bool(torch.isnan(local_rot).any().item())
        has_nan_trans = bool(torch.isnan(root_trans).any().item())
        
        results['nan_check'] = {
            'passed': not (has_nan_pos or has_nan_rot or has_nan_trans),
            'details': {
                'position_nan': has_nan_pos,
                'rotation_nan': has_nan_rot,
                'translation_nan': has_nan_trans
            }
        }
    
    if 'range_check' in checks:
 
        global_pos = motion.global_translation
        heights = global_pos[..., 2]
        
        reasonable_height = bool(torch.all(heights >= -0.5).item() and torch.all(heights <= 3.0).item())
        
        results['range_check'] = {
            'passed': reasonable_height,
            'details': {
                'height_min': float(torch.min(heights).item()),
                'height_max': float(torch.max(heights).item()),
                'reasonable_height': reasonable_height
            }
        }
    
    if 'smoothness_check' in checks:
        # Check smoothness
        root_trans = motion.root_translation
        velocities = torch.norm(root_trans[1:] - root_trans[:-1], dim=-1)
        max_velocity = torch.max(velocities)
        
        smooth_motion = bool(max_velocity < 5.0)  # Threshold adjustable
        
        results['smoothness_check'] = {
            'passed': smooth_motion,
            'details': {
                'max_velocity': float(max_velocity.item()),
                'mean_velocity': float(torch.mean(velocities).item()),
                'smooth_motion': smooth_motion
            }
        }
    
    # Overall results
    all_passed = all(result['passed'] for result in results.values())
    results['overall'] = {
        'passed': all_passed,
        'score': float(sum(result['passed'] for result in results.values()) / len(results))
    }
    
    return results
