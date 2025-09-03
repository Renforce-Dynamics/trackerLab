import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import SceneEntityCfg
from trackerLab.motion_buffer import MotionBuffer

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..manager_based_tracker_env import ManagerBasedTrackerEnv


def truncate_motion_out(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    buffer = env.motion_manager._motion_buffer
    return buffer._motion_times >= buffer._motion_lengths
