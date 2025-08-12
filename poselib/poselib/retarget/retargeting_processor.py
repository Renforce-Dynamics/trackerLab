import os
import json
import numpy as np
import torch
from pathlib import Path
from poselib.skeleton.skeleton3d import SkeletonState, SkeletonMotion, SkeletonTree
from poselib.visualization.common import plot_skeleton_motion_mp4
from poselib.core.rotation3d import quat_mul, quat_inverse, quat_from_angle_axis
from .utils.motion_modi import apply_height_adjustment, validate_motion_quality, extract_motion_statistics
from poselib import POSELIB_DATA_DIR
from typing import Literal, Tuple, Union

RobotType = Literal["smpl", "h1", "g1", "r2"]
RETARGET_CFG_DIR = os.path.join(POSELIB_DATA_DIR, "retarget_cfg")
TPOSE_DIR = os.path.join(POSELIB_DATA_DIR, "tpose")
OUTPUT_DIR = os.path.join(POSELIB_DATA_DIR, "..", "results")

def log(*args, **kwargs):
    pass

class RetargetingProcessor:
    
    def __init__(self,
                 source_robot: RobotType,
                 target_robot: RobotType,
                 retarget_cfg=None,
                #  output_dir: str = None
                 ):
        
        self.target_robot = target_robot
        self.source_robot = source_robot
        self.retarget_cfg_file = retarget_cfg if retarget_cfg is not None else os.path.join(RETARGET_CFG_DIR, f"retar_{source_robot}_2_{target_robot}.json")
        with open(self.retarget_cfg_file, 'r') as f:
            config = json.load(f)
        self.retarget_config = config
        self.load_tposes()
        
        self.output_dir = Path(OUTPUT_DIR) / self.target_robot
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_tposes(self):
        self.source_tpose_path = os.path.join(TPOSE_DIR, f"{self.source_robot}_tpose.npy")
        self.target_tpose_path = os.path.join(TPOSE_DIR, f"{self.target_robot}_tpose.npy")
        
        self.source_tpose = SkeletonState.from_file(self.source_tpose_path)
        self.target_tpose = SkeletonState.from_file(self.target_tpose_path)
        
        self.rotation_to_target = torch.tensor(self.retarget_config["rotation"], dtype=torch.float32)
        self.joint_mapping = self.retarget_config["joint_mapping"]
        self.scale = self.retarget_config["scale"]

    @staticmethod
    def load_tpose(source_robot) -> SkeletonState:
        tpose_path = os.path.join(TPOSE_DIR, f"{source_robot}_tpose.npy")
        tpose = SkeletonState.from_file(tpose_path)
        return tpose

    def retarget_base(self, source_motion: SkeletonMotion) -> SkeletonMotion:
        target_motion = source_motion.retarget_to_by_tpose(
            joint_mapping=self.joint_mapping,
            source_tpose=self.source_tpose,
            target_tpose=self.target_tpose,
            rotation_to_target_skeleton=self.rotation_to_target,
            scale_to_target_skeleton=self.scale
        )
 
        target_motion = apply_height_adjustment(
            target_motion, 
            target_height=self.retarget_config["root_height_offset"],
            method="ground_align"
        )
        
        log("*" * 10 + f"Retargeting completed" + "*" * 10)
        log(f"  Target motion shape: {target_motion.local_rotation.shape}")
        log(f"  Final height range: {torch.min(target_motion.root_translation[:, 2]):.3f} to {torch.max(target_motion.root_translation[:, 2]):.3f} m")
        
        return target_motion

    def save_outputs(
            self, motion: SkeletonMotion, method: str = "simple",
            video = False, report = False
        ) -> dict:

        output_npy = self.output_dir / f"amass_{self.target_robot}_retarget.npy"
        output_mp4 = self.output_dir / f"amass_{self.target_robot}_retarget.mp4"
        output_json = self.output_dir / f"{self.target_robot}_retarget_report.json"

        motion.to_file(str(output_npy))
        log(f"Motion data saved to {output_npy}")

        if video:
            plot_skeleton_motion_mp4(motion, output_filename=str(output_mp4))
        
        if report:
            stats = extract_motion_statistics(motion)
            quality = validate_motion_quality(motion)
            report = {
                "method": f"Retarget",
                "source_robot": self.source_robot,
                "target_robot": self.target_robot,
                "rotation_applied": self.retarget_config["rotation"],
                "scale_applied": self.retarget_config["scale"],
                "height_offset": self.retarget_config["root_height_offset"],
                "additional_operations": method,
                "statistics": stats,
                "quality_check": quality,
                "output_files": {
                    "motion_data": str(output_npy),
                    "video": str(output_mp4)
                }
            }

            with open(output_json, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        else:
            report = {}
        
        return report
    
    @staticmethod
    def save_tpose_json(tpose: SkeletonState):
        tree: SkeletonTree = tpose.skeleton_tree
        report = {
            "skeleton": {
                "node_names": tree.node_names,
                "parent_indices": tree.parent_indices.numpy().tolist(),
                "local_translation": tpose.local_translation.numpy().tolist(),
            },
            "tpose": {
                "root_translation": tpose.root_translation.numpy().tolist(),
                "local_rotation": {}# tpose.local_rotation
            }
        }
        return report
    
    @staticmethod
    def adjust_motion(source_motion: SkeletonMotion, root_idx=0, angle=90, 
                      axis_rot: Union[Tuple[float, float, float], int] = (0.0, 1.0, 0.0)) -> SkeletonMotion:
        """
        Make the motion face to the right direction
        """
        rotations: torch.Tensor = source_motion.rotation  # [T, J, 4]
        frames = rotations.shape[0]
        if isinstance(axis_rot, int):
            axis = torch.tensor([0.0, 0.0, 0.0], dtype=rotations.dtype, device=rotations.device)
            axis[axis_rot] += 1
        else:
            axis = torch.tensor(axis_rot, dtype=rotations.dtype, device=rotations.device)
            
        angle = torch.tensor([angle], dtype=rotations.dtype, device=rotations.device)
        rot_fix = quat_from_angle_axis(angle, axis, degree=True).reshape((1,4)).expand(rotations.shape[0], -1)
        fixed_root = quat_mul(rot_fix, rotations[:, root_idx])
        rotations = rotations.clone()
        rotations[:, root_idx] = fixed_root
        tensor: torch.Tensor = source_motion.tensor.clone()
        replace = rotations.reshape((frames, -1))
        tensor[:, :replace.shape[-1]] = replace
        return SkeletonMotion(tensor,
                            skeleton_tree=source_motion.skeleton_tree,
                            is_local=True,
                            fps=source_motion.fps)
        
    @staticmethod
    def reorder_translation_axes(
        source_motion: SkeletonMotion, 
        order: Tuple[int, int, int] = (2, 1, 0)
    ) -> SkeletonMotion:
        """
        Reorder the root translation axes, e.g., XYZ → ZYX if order=(2,1,0)
        """
        tensor: torch.Tensor = source_motion.tensor.clone()
        num_joints = source_motion.skeleton_tree.num_joints
        frames = tensor.shape[0]

        trans = tensor[:, num_joints * 4 : num_joints * 4 + 3]  # [T, 3]
        trans_reordered = trans[:, list(order)]  # reorder axes
        tensor[:, num_joints * 4 : num_joints * 4 + 3] = trans_reordered

        return SkeletonMotion(
            tensor,
            skeleton_tree=source_motion.skeleton_tree,
            is_local=True,
            fps=source_motion.fps
        )
