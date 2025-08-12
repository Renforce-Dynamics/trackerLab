

import torch
import os
from trackerLab.motion_buffer.motion_lib import MotionLib
from poselib.visualization.common import plot_skeleton_motion, plot_skeleton_motion_mp4
from poselib import POSELIB_DATA_DIR
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
_dof_offsets = [0, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 19, 20, 21]
_key_body_ids = torch.tensor([3, 6, 9, 12], device=device)

motion_dir = os.path.join(POSELIB_DATA_DIR, "configs")
res_dir = "./logs/motion_gen"

device = "cpu"

_dof_body_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
_dof_offsets = [0, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 19, 20, 21]
_key_body_ids = torch.tensor([3, 6, 9, 12], device=device)


motion_files = [
    "motions_autogen_debug_walk.yaml",
]

l
def visualize_motions(motion_file):
    motion_name = motion_file[:-5]
    res_path = os.path.join(res_dir, motion_name)

    tar_path = os.path.join(motion_dir, motion_file)
    motion_lib = MotionLib(
            motion_file=tar_path,
            dof_body_ids=_dof_body_ids,
            dof_offsets=_dof_offsets,
            key_body_ids=_key_body_ids.cpu().numpy(), 
            device=device, 
            regen_pkl=True
        )
        
    os.makedirs(res_path, exist_ok=True)
    for idx, motion in enumerate(motion_lib._motions):
        plot_skeleton_motion_mp4(motion, output_filename=os.path.join(res_path, f"motion_{idx}.mp4"))

for m in motion_files:
    visualize_motions(m)
