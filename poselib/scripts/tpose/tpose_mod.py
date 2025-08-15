import os
import torch
import json
from poselib.retarget.amass_loader import AMASSLoader
from poselib.retarget.pose_generator import PoseGenerator
from poselib.retarget.retargeting_processor import RetargetingProcessor
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.visualization.common import plot_skeleton_motion_mp4
from poselib import POSELIB_DATA_DIR
from trackerLab.utils.animation import animate_skeleton
from poselib.core.rotation3d import quat_mul, quat_from_angle_axis

import argparse

AMASS_DATA_DIR = os.path.join(POSELIB_DATA_DIR, "amass")
TPOSE_DATA_DIR = os.path.join(POSELIB_DATA_DIR, "tpose")
TPOSE_MODI_DIR = os.path.join(TPOSE_DATA_DIR, "modify")

def visualize(source_motion, name):
    edges = AMASSLoader.get_edge_map(source_motion.skeleton_tree)
    animate_skeleton(
        source_motion.global_translation, edges, source_motion.global_root_velocity, 
        interval= 30,
        save_path=f"./results/{name}.mp4")

def verbose_list(tar: list):
    for idx, item in enumerate(tar):
        print(f"{idx:02d}:\t{item}")

def load_cfg(robot_type):
    tar_file = os.path.join(TPOSE_MODI_DIR, f"tpose_{robot_type}_mod.json")
    with open(tar_file, "rb") as f:
        cfg = json.load(f)
        
    return cfg["node_drop"], cfg["mod_seq"]

def tpose_del_nodes(robot, drop_names):
    tpose:SkeletonState = SkeletonState.from_file(os.path.join(TPOSE_DATA_DIR, f"{robot}_tpose.npy"))
    tree: SkeletonTree = tpose.skeleton_tree
    tree = tree.drop_nodes_by_names(drop_names)
    print(tree.node_names)
    # plot_skeleton_state(SkeletonState.zero_pose(g1_tree))
    ret_tpose = SkeletonState.zero_pose(tree)

    return ret_tpose

def tpose_rot_joints(tpose: SkeletonState, rot_seq:list):
    skeleton: SkeletonTree = tpose.skeleton_tree
    local_rotation: torch.Tensor = tpose.local_rotation

    def rot_at_joint(joint_name, angle, axis):
        local_rotation[skeleton.index(joint_name)] = quat_mul(
            quat_from_angle_axis(angle=torch.tensor([angle]), axis=torch.tensor(axis), degree=True), 
            local_rotation[skeleton.index(joint_name)]
        )

    for rot in rot_seq:
        rot_at_joint(*rot)
    
    # Create T-pose
    t_pose = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree=skeleton,
        r=local_rotation,
        t=tpose.root_translation,
        is_local=True
    )

    return t_pose

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    
    source:str = args.source
    target:str = args.target

    node_drop, rot_seq = load_cfg(target)
    
    tpose = tpose_del_nodes(source, node_drop)
    tpose = tpose_rot_joints(tpose, rot_seq)
    
    verbose_list(tpose.skeleton_tree.node_names)
    
    plot_skeleton_state(tpose)
    
    ret_path = os.path.join(TPOSE_DATA_DIR, f"{target}_tpose.npy")
    tpose.to_file(ret_path)
