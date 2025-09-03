import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat

from trackerLab.commands.base_command import BaseCommand

class EndposCommand(BaseCommand):
    """
    Sample end pos from motion manager.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def __str__(self) -> str:
        msg = "Endpos for motion:\n"
        msg += "\tCommand dimension: N/A\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    def _set_debug_vis_impl(self, debug_vis):
        super()._set_debug_vis_impl(debug_vis)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)

    @property
    def command(self):
        return super().command