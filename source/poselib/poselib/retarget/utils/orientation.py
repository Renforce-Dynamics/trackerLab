import torch
import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as R
from poselib.core.rotation3d import quat_mul, quat_conjugate, quat_rotate, quat_from_angle_axis
from poselib.skeleton.skeleton3d import SkeletonState


def check_orientation_quality(skeleton_state, robot_type="h1"):
    """
    Check orientation quality of skeleton state
    
    Args:
        skeleton_state: SkeletonState object
        robot_type: Robot type ("h1", "r2", "smpl")
    
    Returns:
        dict: Dictionary containing orientation quality information
    """
    pos = skeleton_state.global_translation
    
    # Define shoulder joint names for different robot types
    shoulder_joint_map = {
        "h1": ("left_shoulder_pitch_link", "right_shoulder_pitch_link"),
        "r2": ("left_shoulder_pitch_link", "right_shoulder_pitch_link"),
        "smpl": ("left_shoulder", "right_shoulder")  # SMPL uses different naming
    }
    
    if robot_type not in shoulder_joint_map:
        return {
            'orientation_quality': 0.0,
            'orientation_correct': False,
            'shoulder_vector': [0, 0, 0],
            'needs_fix': True
        }
    
    left_shoulder_name, right_shoulder_name = shoulder_joint_map[robot_type]
    joint_names = skeleton_state.skeleton_tree.node_names
    
    if left_shoulder_name in joint_names and right_shoulder_name in joint_names:
        left_shoulder_idx = joint_names.index(left_shoulder_name)
        right_shoulder_idx = joint_names.index(right_shoulder_name)
        shoulder_vector = pos[right_shoulder_idx] - pos[left_shoulder_idx]
        y_component = abs(shoulder_vector[1])
        total_length = torch.norm(shoulder_vector)
        
        if total_length > 0:
            y_ratio = y_component / total_length
            orientation_correct = y_ratio > 0.8
            
            return {
                'orientation_quality': float(y_ratio),
                'orientation_correct': orientation_correct,
                'shoulder_vector': shoulder_vector.numpy().tolist(),
                'needs_fix': not orientation_correct
            }
    
    return {
        'orientation_quality': 0.0,
        'orientation_correct': False,
        'shoulder_vector': [0, 0, 0],
        'needs_fix': True
    }


def apply_orientation_fix(skeleton_state, robot_type="h1"):

    quality_before = check_orientation_quality(skeleton_state, robot_type)
    
    if not quality_before['needs_fix']:
        print(f"✅ Orientation correct, quality: {quality_before['orientation_quality']:.3f}")
        return skeleton_state, quality_before
    
    print(f"🔧 Applying orientation correction (current quality: {quality_before['orientation_quality']:.3f})")

    # 90-degree rotation around X-axis for correction
    fix_rotation = torch.tensor([0.7071067811865475, 0.0, 0.0, -0.7071067811865476])

    global_pos = skeleton_state.global_translation.clone()
    global_rot = skeleton_state.global_rotation.clone()

    for i in range(len(global_pos)):
        pos = global_pos[i]
        rotated_pos = quat_rotate(fix_rotation.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)
        global_pos[i] = rotated_pos
        rot = global_rot[i]
        rotated_rot = quat_mul(fix_rotation.unsqueeze(0), rot.unsqueeze(0)).squeeze(0)
        global_rot[i] = rotated_rot

    fixed_state = SkeletonState.from_translation_and_rotation(
        skeleton_state.skeleton_tree, global_pos, global_rot, is_local=False
    )

    quality_after = check_orientation_quality(fixed_state, robot_type)
    print(f"✅ Orientation correction completed, quality: {quality_before['orientation_quality']:.3f} -> {quality_after['orientation_quality']:.3f}")
    return fixed_state, quality_after


def validate_phc_compatibility(tpose, skeleton_type):

    print(f"🔍 Validating PHC compatibility - {skeleton_type}...")
    pos = tpose.global_translation

    orientation_check = check_orientation_quality(tpose, skeleton_type)
    print(f"📐 Orientation check:")
    print(f"  Orientation quality: {orientation_check['orientation_quality']:.3f}")
    print(f"  Orientation correct: {'✅' if orientation_check['orientation_correct'] else '❌'}")
    
    if skeleton_type == "smpl":
        pelvis_pos = pos[0]  # pelvis
        head_pos = pos[15]   # head
        left_shoulder_pos = pos[16]   # left_shoulder  
        left_hand_pos = pos[22]       # left_hand
        left_hip_pos = pos[1]         # left_hip
        left_foot_pos = pos[10]       # left_foot
        total_height = torch.norm(head_pos - pelvis_pos)
        arm_length = torch.norm(left_hand_pos - left_shoulder_pos)
        leg_length = torch.norm(left_foot_pos - left_hip_pos)
        
        print(f"📏 SMPL measurements:")
        print(f"  Total height: {total_height:.3f}m")
        print(f"  Arm length: {arm_length:.3f}m") 
        print(f"  Leg length: {leg_length:.3f}m")
    
        structure_valid = (1.4 <= total_height <= 2.0 and 
                          0.4 <= arm_length <= 0.8 and 
                          0.7 <= leg_length <= 1.2)
        
        if structure_valid and orientation_check['orientation_correct']:
            print("✅ SMPL structure meets PHC standards")
            return True
        else:
            print("⚠️ SMPL structure may not meet PHC standards")
            return False
            
    elif skeleton_type == "h1":
        pelvis_pos = pos[0]  # pelvis
        torso_pos = pos[7]   # torso_link
        left_hand_pos = pos[10]   # left_hand_link
        left_ankle_pos = pos[3]   # left_ankle_link
        torso_height = torch.norm(torso_pos - pelvis_pos)
        arm_reach = torch.norm(left_hand_pos - torso_pos)
        leg_length = torch.norm(left_ankle_pos - pelvis_pos)
        
        print(f"📏 H1 measurements:")
        print(f"  Torso height: {torso_height:.3f}m")
        print(f"  Arm reach: {arm_reach:.3f}m") 
        print(f"  Leg length: {leg_length:.3f}m")

        structure_valid = (0.2 <= torso_height <= 0.3 and 
                          0.4 <= arm_reach <= 0.7 and 
                          0.6 <= leg_length <= 0.8)
        
        if structure_valid and orientation_check['orientation_correct']:
            print("✅ H1 structure meets PHC standards")
            return True
        else:
            print("⚠️ H1 structure may not meet PHC standards")
            return False
    
    elif skeleton_type == "r2":
        base_pos = pos[0]    # base_link
        waist_pos = pos[7]   # waist_pitch_link
        left_hand_pos = pos[10]   # left_hand_link
        left_ankle_pos = pos[3]   # left_ankle_roll_link
        torso_height = torch.norm(waist_pos - base_pos)
        arm_reach = torch.norm(left_hand_pos - waist_pos)
        leg_length = torch.norm(left_ankle_pos - base_pos)
        
        print(f"📏 R2 measurements:")
        print(f"  Torso height: {torso_height:.3f}m")
        print(f"  Arm reach: {arm_reach:.3f}m") 
        print(f"  Leg length: {leg_length:.3f}m")

        structure_valid = (0.15 <= torso_height <= 0.25 and 
                          0.3 <= arm_reach <= 0.6 and 
                          0.4 <= leg_length <= 0.6)
        
        if structure_valid and orientation_check['orientation_correct']:
            print("✅ R2 structure meets PHC standards")
            return True
        else:
            print("⚠️ R2 structure may not meet PHC standards")
            return False
    
    return False


def save_orientation_report(tpose, skeleton_type, output_dir):

    orientation_check = check_orientation_quality(tpose, skeleton_type)
    report = {
        'skeleton_type': skeleton_type,
        'orientation_quality': float(orientation_check['orientation_quality']),
        'orientation_correct': bool(orientation_check['orientation_correct']),
        'shoulder_vector': [float(x) for x in orientation_check['shoulder_vector']],
        'needs_fix': bool(orientation_check['needs_fix']),
        'validation_status': 'PASS' if orientation_check['orientation_correct'] else 'FAIL'
    }
    
    report_file = os.path.join(output_dir, f"{skeleton_type}_orientation_report.json")
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📊 Orientation report saved: {report_file}")
    return report