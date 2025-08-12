from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
from poselib.visualization.common import plot_skeleton_state

# load in XML mjcf file and save zero rotation pose in npy format
xml_path = "data/assets/g1_description/g1_23dof.xml"
skeleton = SkeletonTree.from_mjcf(xml_path)
zero_pose = SkeletonState.zero_pose(skeleton)
zero_pose.to_file("data/tpose/g1_23dof.npy")

def verbose_list(tar: list):
    for idx, item in enumerate(tar):
        print(f"{idx:02d}:\t{item}")
        
verbose_list(skeleton.node_names)

# visualize zero rotation pose
plot_skeleton_state(zero_pose)