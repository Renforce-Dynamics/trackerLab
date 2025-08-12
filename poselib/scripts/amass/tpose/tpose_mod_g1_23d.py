import os
import torch
from poselib.retarget.amass_loader import AMASSLoader
from poselib.retarget.pose_generator import PoseGenerator
from poselib.retarget.retargeting_processor import RetargetingProcessor
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.visualization.common import plot_skeleton_motion_mp4
from poselib import POSELIB_DATA_DIR
from trackerLab.utils.animation import animate_skeleton
from poselib.core.rotation3d import quat_mul, quat_from_angle_axis

AMASS_DATA_DIR = os.path.join(POSELIB_DATA_DIR, "amass")
TPOSE_DATA_DIR = os.path.join(POSELIB_DATA_DIR, "tpose")

def visualize(source_motion, name):
    edges = AMASSLoader.get_edge_map(source_motion.skeleton_tree)
    animate_skeleton(
        source_motion.global_translation, edges, source_motion.global_root_velocity, 
        interval= 30,
        save_path=f"./results/{name}.mp4")

def tpose_del_nodes():
    g1_drop_names = [
        # "pelvis",
        "left_hip_pitch_link",
        # "left_hip_roll_link",
        # "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        # "left_ankle_roll_link",
        "right_hip_pitch_link",
        # "right_hip_roll_link",
        # "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        # "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        # "left_shoulder_roll_link",
        # "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_rubber_hand",
        "right_shoulder_pitch_link",
        # "right_shoulder_roll_link",
        # "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_rubber_hand",
    ]
    g1_drop_names = [
        # "pelvis",
        # "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        # "left_knee_link",
        # "left_ankle_pitch_link",
        "left_ankle_roll_link",
        # "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        # "right_knee_link",
        # "right_ankle_pitch_link",
        "right_ankle_roll_link",
        # "torso_link",
        # "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        # "left_elbow_link",
        # "left_wrist_roll_rubber_hand",
        # "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        # "right_elbow_link",
        # "right_wrist_roll_rubber_hand",
    ]
    g1_tpose:SkeletonState = SkeletonState.from_file(os.path.join(TPOSE_DATA_DIR, "g1_23dof_tpose.npy"))
    # plot_skeleton_state(g1_tpose)
    g1_tree: SkeletonTree = g1_tpose.skeleton_tree
    g1_tree = g1_tree.drop_nodes_by_names(g1_drop_names)
    print(g1_tree.node_names)
    # plot_skeleton_state(SkeletonState.zero_pose(g1_tree))
    g1_tpose = SkeletonState.zero_pose(g1_tree)

    return g1_tpose



def tpose_rot_joints(g1_tpose: SkeletonState):
    skeleton: SkeletonTree = g1_tpose.skeleton_tree
    local_rotation: torch.Tensor = g1_tpose.local_rotation

    def rot_at_joint(joint_name, angle, axis):
        local_rotation[skeleton.index(joint_name)] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([angle]), axis=torch.tensor(axis), degree=True), 
            local_rotation[skeleton.index(joint_name)]
        )

    rot_at_joint("left_shoulder_pitch_link", 80.0, [1.0, 0.0, 0.0])
    rot_at_joint("right_shoulder_pitch_link", -80.0, [1.0, 0.0, 0.0])
    rot_at_joint("left_elbow_link", 90.0, [0.0, 1.0, 0.0])
    rot_at_joint("right_elbow_link", 90.0, [0.0, 1.0, 0.0])
    # rot_at_joint("left_hip_yaw_link", -15.0, [0.0, 1.0, 0.0])
    # rot_at_joint("right_hip_yaw_link", -15.0, [0.0, 1.0, 0.0])
    # rot_at_joint("left_knee_link", 15.0, [0.0, 1.0, 0.0])
    # rot_at_joint("right_knee_link", 15.0, [0.0, 1.0, 0.0])
    # rot_at_joint("left_shoulder_pitch_link", -15.0, [0.0, 1.0, 0.0])
    
    # Create T-pose
    t_pose = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=skeleton,
        r=local_rotation,
        t=g1_tpose.root_translation,
        is_local=True
    )

    return t_pose

if __name__ == "__main__":
    g1_tpose = tpose_del_nodes()
    g1_tpose = tpose_rot_joints(g1_tpose)
    plot_skeleton_state(g1_tpose)
    g1_tpose.to_file("./data/tpose/g1_23d_tpose.npy")
    pass