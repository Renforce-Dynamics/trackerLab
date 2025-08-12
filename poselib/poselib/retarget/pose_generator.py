import json
import torch
import os
from poselib.core.rotation3d import quat_from_angle_axis
from .utils.orientation import apply_orientation_fix, validate_phc_compatibility, save_orientation_report
from .utils.motion_modi import save_skeleton_state_to_file
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree, SkeletonMotion
from poselib import POSELIB_DATA_DIR

from poselib.visualization.common import plot_skeleton_state

class PoseGenerator:

    def __init__(self, config_dir: str = None):
        self.config_dir = config_dir
        self.skeleton = None
        self.local_rotation = None
        self.root_translation = None
        self.tpose = None
        
        if config_dir is None:
            self.config_dir = os.path.join(POSELIB_DATA_DIR, "retarget")

    def _load_config(self, robot_type: str):

        config_file = os.path.join(self.config_dir, f"skeleton_{robot_type}.json")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Loaded {robot_type.upper()} skeleton from config file: {config_file}")
        
        skeleton_config = config["skeleton"]
        node_names = skeleton_config["node_names"]
        parent_indices = torch.tensor(skeleton_config["parent_indices"], dtype=torch.long)
        local_translation = torch.tensor(skeleton_config["local_translation"], dtype=torch.float32)
        
        skeleton = SkeletonTree(node_names, parent_indices, local_translation)
        
        root_translation = torch.tensor(config["tpose"]["root_translation"], dtype=torch.float32)
        
        n_joints = len(skeleton.node_names)
        local_rotation = torch.zeros((n_joints, 4), dtype=torch.float32)
        local_rotation[:, 3] = 1.0  
        
        for joint_name, rot in config["tpose"].get("local_rotations", {}).items():
            if joint_name not in skeleton.node_names:
                raise ValueError(f"Joint '{joint_name}' not found in skeleton node_names")
            idx = skeleton.node_names.index(joint_name)
            angle = torch.tensor(rot["angle"], dtype=torch.float32)
            axis = torch.tensor(rot["axis"], dtype=torch.float32)
            degree = rot.get("degree", False)
            quat = quat_from_angle_axis(angle, axis, degree)
            local_rotation[idx] = quat
        
        return skeleton, local_rotation, root_translation

    def generate_tpose(self, robot_type: str):
 
        skeleton, local_rotation, root_translation = self._load_config(robot_type)
        
        self.tpose = SkeletonState.from_rotation_and_root_translation(
            skeleton, local_rotation, root_translation, is_local=True
        )
        return self.tpose

    def create_phc_smpl_skeleton(self):
        skeleton, _, _ = self._load_config("smpl")
        return skeleton

    def create_phc_h1_skeleton(self):
        skeleton, _, _ = self._load_config("h1")
        return skeleton

    def create_r2_skeleton(self):
        skeleton, _, _ = self._load_config("r2")
        return skeleton

    def generate_phc_smpl_tpose(self):
        print("🔧 Generating PHC-style SMPL T-pose from config file...")
        return self.generate_tpose("smpl")

    def generate_phc_h1_tpose(self):
        print("🔧 Generating corrected PHC-style H1 robot T-pose with orientation correction...")
        skeleton, local_rotation, root_translation = self._load_config("h1")
        
        tpose = SkeletonState.from_rotation_and_root_translation(
            skeleton, local_rotation, root_translation, is_local=True
        )
        
        print("🔍 Validating H1 pose symmetry...")
        pos = tpose.global_translation

        left_shoulder = pos[8]   # left_shoulder_pitch_link
        right_shoulder = pos[11] # right_shoulder_pitch_link
        left_elbow = pos[9]      # left_elbow_link  
        right_elbow = pos[12]    # right_elbow_link
        left_hand = pos[10]      # left_hand_link
        right_hand = pos[13]     # right_hand_link
        
        print(f"  Left shoulder: [{left_shoulder[0]:6.3f}, {left_shoulder[1]:6.3f}, {left_shoulder[2]:6.3f}]")
        print(f"  Right shoulder: [{right_shoulder[0]:6.3f}, {right_shoulder[1]:6.3f}, {right_shoulder[2]:6.3f}]")
        print(f"  Left elbow: [{left_elbow[0]:6.3f}, {left_elbow[1]:6.3f}, {left_elbow[2]:6.3f}]")
        print(f"  Right elbow: [{right_elbow[0]:6.3f}, {right_elbow[1]:6.3f}, {right_elbow[2]:6.3f}]")
        print(f"  Left hand: [{left_hand[0]:6.3f}, {left_hand[1]:6.3f}, {left_hand[2]:6.3f}]")
        print(f"  Right hand: [{right_hand[0]:6.3f}, {right_hand[1]:6.3f}, {right_hand[2]:6.3f}]")
        
        y_symmetry_shoulder = abs(left_shoulder[1] + right_shoulder[1]) < 0.01
        y_symmetry_elbow = abs(left_elbow[1] + right_elbow[1]) < 0.01
        y_symmetry_hand = abs(left_hand[1] + right_hand[1]) < 0.01
        
        if y_symmetry_shoulder and y_symmetry_elbow and y_symmetry_hand:
            print("✅ H1 upper body is perfectly symmetric!")
        else:
            print("⚠️ H1 upper body is asymmetric, needs fixing")
        
        print("\n🔍 Checking robot orientation...")
        fixed_tpose, orientation_quality = apply_orientation_fix(tpose, "h1")
        
        print(f"📊 Orientation quality report:")
        print(f"  Orientation quality: {orientation_quality['orientation_quality']:.3f}")
        print(f"  Orientation correct: {'✅' if orientation_quality['orientation_correct'] else '❌'}")
        print(f"  Shoulder vector: {orientation_quality['shoulder_vector']}")
        
        return fixed_tpose

    def generate_phc_r2_tpose(self):

        print("🔧 Generating corrected PHC-style R2 robot T-pose with orientation correction...")
        
        skeleton, local_rotation, root_translation = self._load_config("r2")
        
        tpose = SkeletonState.from_rotation_and_root_translation(
            skeleton, local_rotation, root_translation, is_local=True
        )
        
        print("🔍 Validating R2 pose symmetry...")
        pos = tpose.global_translation
        
        left_shoulder = pos[8]   # left_shoulder_pitch_link
        right_shoulder = pos[11] # right_shoulder_pitch_link
        left_elbow = pos[9]      # left_arm_pitch_link  
        right_elbow = pos[12]    # right_arm_pitch_link
        left_hand = pos[10]      # left_hand_link
        right_hand = pos[13]     # right_hand_link
        
        print(f"  Left shoulder: [{left_shoulder[0]:6.3f}, {left_shoulder[1]:6.3f}, {left_shoulder[2]:6.3f}]")
        print(f"  Right shoulder: [{right_shoulder[0]:6.3f}, {right_shoulder[1]:6.3f}, {right_shoulder[2]:6.3f}]")
        print(f"  Left elbow: [{left_elbow[0]:6.3f}, {left_elbow[1]:6.3f}, {left_elbow[2]:6.3f}]")
        print(f"  Right elbow: [{right_elbow[0]:6.3f}, {right_elbow[1]:6.3f}, {right_elbow[2]:6.3f}]")
        print(f"  Left hand: [{left_hand[0]:6.3f}, {left_hand[1]:6.3f}, {left_hand[2]:6.3f}]")
        print(f"  Right hand: [{right_hand[0]:6.3f}, {right_hand[1]:6.3f}, {right_hand[2]:6.3f}]")
        
        y_symmetry_shoulder = abs(left_shoulder[1] + right_shoulder[1]) < 0.01
        y_symmetry_elbow = abs(left_elbow[1] + right_elbow[1]) < 0.01
        y_symmetry_hand = abs(left_hand[1] + right_hand[1]) < 0.01
        
        if y_symmetry_shoulder and y_symmetry_elbow and y_symmetry_hand:
            print("✅ R2 upper body is perfectly symmetric!")
        else:
            print("⚠️ R2 upper body is asymmetric, needs fixing")
        
        print("\n🔍 Checking R2 robot orientation...")
        fixed_tpose, orientation_quality = apply_orientation_fix(tpose, "r2")
        
        print(f"📊 R2 orientation quality report:")
        print(f"  Orientation quality: {orientation_quality['orientation_quality']:.3f}")
        print(f"  Orientation correct: {'✅' if orientation_quality['orientation_correct'] else '❌'}")
        print(f"  Shoulder vector: {orientation_quality['shoulder_vector']}")
        
        return fixed_tpose

    def generate_and_save_all_tposes(self, output_dir: str = None):

        if output_dir is None:
            output_dir = POSELIB_DATA_DIR

        print("🎯 PHC-style T-pose Generator (Config-driven version)")
        print("=" * 60)
        print("📚 Reference: Perpetual Humanoid Control (PHC)")
        print("🦴 Using JSON config files to define all skeletons")
        print("🔧 Integrated orientation detection and auto-correction")

        os.makedirs(output_dir, exist_ok=True)
        
        print("\n1️⃣ Generating PHC-style SMPL T-pose...")
        smpl_tpose = self.generate_phc_smpl_tpose()
        smpl_valid = validate_phc_compatibility(smpl_tpose, "smpl")
        smpl_tpose_path = os.path.join(output_dir, "smpl_tpose.npy")
        save_skeleton_state_to_file(smpl_tpose, smpl_tpose_path)
        print(f"💾 SMPL T-pose saved: {smpl_tpose_path}")

        save_orientation_report(smpl_tpose, "smpl", output_dir)

        print("\n2️⃣ Generating PHC-style H1 robot T-pose...")
        h1_tpose = self.generate_phc_h1_tpose()
        h1_valid = validate_phc_compatibility(h1_tpose, "h1")
        h1_tpose_path = os.path.join(output_dir, "h1_tpose.npy")
        save_skeleton_state_to_file(h1_tpose, h1_tpose_path)
        print(f"💾 H1 T-pose saved: {h1_tpose_path}")

        plot_skeleton_state(smpl_tpose)

        h1_report = save_orientation_report(h1_tpose, "h1", output_dir)
        
        print("\n3️⃣ Generating PHC-style R2 robot T-pose...")
        r2_tpose = self.generate_phc_r2_tpose()
        r2_valid = validate_phc_compatibility(r2_tpose, "r2")
        r2_tpose_path = os.path.join(output_dir, "r2_tpose.npy")
        save_skeleton_state_to_file(r2_tpose, r2_tpose_path)
        print(f"💾 R2 T-pose saved: {r2_tpose_path}")

        r2_report = save_orientation_report(r2_tpose, "r2", output_dir)
        
        print(f"\n✅ PHC-style T-pose generation completed!")
        print(f"📁 T-pose directory: {os.path.abspath(output_dir)}")
        print(f"📁 Config file directory: {os.path.abspath(self.config_dir)}")
        
        if smpl_valid and h1_valid and r2_valid:
            print("🎉 All T-poses meet PHC standards!")
        else:
            print("⚠️ Some T-poses may need further adjustment")
        
        print(f"\n📐 Final orientation quality:")
        print(f"   H1 robot: {h1_report['orientation_quality']:.3f} ({'✅ Excellent' if h1_report['orientation_quality'] > 0.9 else '⚠️ Needs improvement'})")
        print(f"   R2 robot: {r2_report['orientation_quality']:.3f} ({'✅ Excellent' if r2_report['orientation_quality'] > 0.9 else '⚠️ Needs improvement'})")
        
        return {
            'smpl_tpose': smpl_tpose,
            'h1_tpose': h1_tpose,
            'r2_tpose': r2_tpose,
            'reports': {
                'h1': h1_report,
                'r2': r2_report
            }
        }