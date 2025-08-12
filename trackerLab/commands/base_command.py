import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat
from isaaclab.utils.math import quat_rotate_inverse, quat_rotate

from trackerLab.motion_drawer.single_motion_drawer import SingleMotionDrawer

class BaseCommand(CommandTerm):
    """
    Treating the pos info as command. The motion calced is done in motion manager.
    Here we only transform it into different observation.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def __str__(self) -> str:
        msg = "BaseCommand for motion:\n"
        msg += "\tCommand dimension: N/A\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    def _debug_vis_callback(self, event):
        return
    
    def _set_debug_vis_impl(self, debug_vis):
        return

    def _update_metrics(self):
        return

    def _update_command(self):
        return

    def _resample_command(self, env_ids):
        return
    
    def _resample(self, env_ids):
        if self.cfg.resampling_time_range is not None:
            return super()._resample(env_ids)
        else:
            return None

    @property
    def command(self):
        raise NotImplementedError("No Command for Base pos cmd.")


class SelfTransCommand(BaseCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
    
    def _debug_vis_callback(self, event):    
        super()._debug_vis_callback(event)    
        if self.self_motion_drawer is not None:
            # Draw Local infos.
            asset = self._env.scene["robot"]
            self_points = asset.data.body_link_pos_w
            self.self_motion_drawer.visualize(self_points.reshape(-1, 3))

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            self.self_motion_drawer = SingleMotionDrawer(self.device, (0.5, 0.5, 0.0), "SelfMotion")
        else:
            self.self_motion_drawer = None

from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg
@configclass
class SelfTransCommandCfg(CommandTermCfg):
    class_type: type = SelfTransCommand
    
    def __post_init__(self):
        self.resampling_time_range = None