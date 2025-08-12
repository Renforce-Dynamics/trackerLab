from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# import isaaclab.utils.math as math_utils
# from isaaclab.assets import Articulation, RigidObject
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.managers.manager_base import ManagerTermBase
# from isaaclab.managers.manager_term_cfg import ObservationTermCfg
# from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..manager_based_tracker_env import ManagerBasedTrackerEnv
    
# Demo terms, demo terms are directly get from the env motion manager
def motion_dof_pos_whb(env: ManagerBasedTrackerEnv) -> torch.Tensor:
    return env.motion_manager.loc_dof_pos

def demo_height(env: ManagerBasedTrackerEnv) -> torch.Tensor:
    return env.motion_manager.loc_height.reshape(-1, 1)

def demo_root_vel(env: ManagerBasedTrackerEnv) -> torch.Tensor:
    return env.motion_manager.loc_root_vel

def demo_ang_vel(env: ManagerBasedTrackerEnv) -> torch.Tensor:
    return env.motion_manager.loc_ang_vel
