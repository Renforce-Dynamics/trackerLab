from __future__ import annotations

import torch
from typing import TYPE_CHECKING

# import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
# from isaaclab.managers.manager_base import ManagerTermBase
# from isaaclab.managers.manager_term_cfg import ObservationTermCfg
# from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..manager_based_tracker_env import ManagerBasedTrackerEnv
    
def track_whole_body_trans(env: ManagerBasedTrackerEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_link_pos_w