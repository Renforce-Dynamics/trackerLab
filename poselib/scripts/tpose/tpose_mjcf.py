from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state
import os
import argparse
from poselib import POSELIB_DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--xml", required=True)
parser.add_argument("--robot", required=True)
args = parser.parse_args()

# load in XML mjcf file and save zero rotation pose in npy format
xml_path = args.xml
skeleton = SkeletonTree.from_mjcf(xml_path)
zero_pose = SkeletonState.zero_pose(skeleton)

ret_file = os.path.join(POSELIB_DATA_DIR, f"tpose/{args.robot}_tpose.npy")
zero_pose.to_file(ret_file)

def verbose_list(tar: list):
    for idx, item in enumerate(tar):
        print(f"{idx:02d}:\t{item}")
        
verbose_list(skeleton.node_names)

# visualize zero rotation pose
plot_skeleton_state(zero_pose)