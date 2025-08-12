import os
import sys
import argparse
from pathlib import Path

from poselib.retarget.amass_loader import AMASSLoader
from poselib.retarget.pose_generator import PoseGenerator
from poselib.retarget.retargeting_processor import RetargetingProcessor
from poselib import POSELIB_DATA_DIR
from poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from poselib.visualization.common import plot_skeleton_motion_mp4
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion

from trackerLab.utils.animation import animate_skeleton

AMASS_DATA_DIR = os.path.join(POSELIB_DATA_DIR, "amass")
TPOSE_DATA_DIR = os.path.join(POSELIB_DATA_DIR, "tpose")

amass_file = os.path.join(AMASS_DATA_DIR, "CMU", "02", "02_02_poses.npz")

def visualize(source_motion, name):
    edges = AMASSLoader.get_edge_map(source_motion.skeleton_tree)
    animate_skeleton(
        source_motion.global_translation, edges, source_motion.global_root_velocity, 
        interval= 30,
        save_path=f"./results/{name}.mp4")

def verbose_list(tar: list):
    for idx, item in enumerate(tar):
        print(f"{idx:02d}:\t{item}")

def check_tpose():
    
    def save_json(report, ret_file):
        import json
        with open(ret_file, "wt") as f:
            json.dump(report, f, indent=4)
    
    smpl_tpose = SkeletonState.from_file(os.path.join(TPOSE_DATA_DIR, "smpl_tpose.npy"))
    verbose_list(smpl_tpose.skeleton_tree.node_names)
    plot_skeleton_state(smpl_tpose)
    smplh_tpose = SkeletonState.from_file(os.path.join(TPOSE_DATA_DIR, "smplh_tpose.npy"))
    # plot_skeleton_state(smplh_tpose)
    h1_tpose = SkeletonState.from_file(os.path.join(TPOSE_DATA_DIR, "h1_tpose.npy"))
    # plot_skeleton_state(h1_tpose)
    g1_tpose = SkeletonState.from_file(os.path.join(TPOSE_DATA_DIR, "g1_tpose.npy"))
    # verbose_list(g1_tpose.skeleton_tree.node_names)
    # plot_skeleton_state(g1_tpose)
    g1_23dof_tpose = SkeletonState.from_file(os.path.join(TPOSE_DATA_DIR, "g1_23dof_tpose.npy"))
    verbose_list(g1_23dof_tpose.skeleton_tree.node_names)
    plot_skeleton_state(g1_23dof_tpose)
    g1_23d_tpose = SkeletonState.from_file(os.path.join(TPOSE_DATA_DIR, "g1_23d_tpose.npy"))
    verbose_list(g1_23d_tpose.skeleton_tree.node_names)
    plot_skeleton_state(g1_23d_tpose)
    r2_tpose = SkeletonState.from_file(os.path.join(TPOSE_DATA_DIR, "r2_tpose.npy"))
    r2_json = RetargetingProcessor.save_tpose_json(r2_tpose)
    # plot_skeleton_state(r2_tpose)
    r2y_tpose = SkeletonState.from_file(os.path.join(TPOSE_DATA_DIR, "r2y_tpose.npy"))
    # plot_skeleton_state(r2y_tpose)
    # save_json(r2_json, "./data/tpose/r2.json")
    return

def mod_tpose():
    r2_drop_names = [
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
    ]
    r2_tpose:SkeletonState = SkeletonState.from_file(os.path.join(TPOSE_DATA_DIR, "r2_tpose.npy"))
    # plot_skeleton_state(r2_tpose)
    r2_tree: SkeletonTree = r2_tpose.skeleton_tree
    r2_tree = r2_tree.drop_nodes_by_names(r2_drop_names)
    # plot_skeleton_state(SkeletonState.zero_pose(r2_tree))
    r2_tpose = SkeletonState.zero_pose(r2_tree)
    
    r2_tpose.to_file("./data/tpose/r2y_tpose.npy")
    return

def check_smpl_skeleton():
    loader = AMASSLoader(max_frames=1000)
    sk_1 = loader.smpl_skeleton
    sk_2 = loader.smpl_skeleton_amass
    return

def check_amass_data():
    import matplotlib
    matplotlib.use('Agg')
    loader = AMASSLoader(max_frames=1000)
    source_motion = loader.load_and_process(amass_file)
    # plot_skeleton_motion_mp4(source_motion, output_filename="./results/motion.mp4")
    visualize(source_motion, "check_amass_0")
    
def check_retarget():
    import matplotlib
    matplotlib.use('Agg')
    loader = AMASSLoader(max_frames=1000)
    source_motion = loader.load_and_process(amass_file)
    retargetor = RetargetingProcessor("smpl", "g1_23d")
    target_motion = retargetor.retarget_base(source_motion)
    target_motion = RetargetingProcessor.adjust_motion(target_motion, 0, angle=-90, axis_rot=1)
    target_motion = RetargetingProcessor.adjust_motion(target_motion, 0, angle=-90, axis_rot=2)
    target_motion = RetargetingProcessor.reorder_translation_axes(target_motion, (1, 2, 0))
    target_motion = AMASSLoader.fill_motion_vel(target_motion)
    # plot_skeleton_motion_mp4(target_motion, output_filename="./results/retarget.mp4")
    visualize(target_motion, "retarget_0_g1_23d")
    return

def check_motion():
    import matplotlib
    matplotlib.use('Agg')
    ret_dir = os.path.join(POSELIB_DATA_DIR, "amass_results", "r2y", "CMU/02/02_02_poses.npy")
    source_motion = SkeletonMotion.from_file(ret_dir)
    visualize(source_motion, "test1")

if __name__ == "__main__":
    # check_smpl_skeleton()
    check_tpose()
    # mod_tpose()
    # check_amass_data()
    # check_retarget()
    # check_motion()
    pass