import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply_inverse, wrap_to_pi, yaw_quat
from isaaclab.utils.math import quat_apply_inverse, quat_rotate, quat_apply

from trackerLab.motion_drawer.single_motion_drawer import SingleMotionDrawer
from trackerLab.motion_drawer.single_vel_drawer import SingleVelDrawer

from trackerLab.commands.base_command import BaseCommand

class DofposCommand(BaseCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.verbose_detail: bool = getattr(cfg, "verbose_detail", False)
        self.lab_joint_names = self._env.scene.articulations["robot"]._data.joint_names

    def __str__(self) -> str:
        msg = "Exbody dof pos."
        return msg

    def _set_debug_vis_impl(self, debug_vis):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            self.ref_motion_drawer = SingleMotionDrawer(self.device, color=(1.0, 0.0, 0.0), prim_name="RefJoints")
        else:
            self.ref_motion_drawer = None

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if self.ref_motion_drawer is not None:
            # Draw Local infos.
            asset = self._env.scene["robot"]
            root_pos = asset.data.root_pos_w
            current_rot = asset.data.root_quat_w.unsqueeze(1)
            local_points = self._env.motion_manager.loc_trans_base # shape: [N, 14, 3]
            n_joints = local_points.shape[1]
            local_points = quat_apply(current_rot.expand(-1, n_joints, -1).reshape(-1, 4), 
                                    local_points.view(-1,3)).reshape(local_points.shape)  
            local_points = local_points  + root_pos.unsqueeze(1)
            self.ref_motion_drawer.visualize(local_points.reshape(-1, 3))
            

    def _update_metrics(self):
        env = self._env
        diff_angle: torch.Tensor = env.joint_subset - env.motion_manager.loc_dof_pos
        diff_total = torch.sum(torch.abs(diff_angle), dim=1)
        self.metrics["Total dof pos diff"] = diff_total
        if self.verbose_detail:
            diff_angle = torch.abs(diff_angle)
            for idx, ptr in enumerate(env.motion_manager.shared_subset_lab) :
                name = self.lab_joint_names[ptr]
                self.metrics[f"dof diff: {name}"] = diff_angle[:, idx]

    @property
    def command(self):
        return self._env.motion_manager.loc_dof_pos
    
class HeightCommand(BaseCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

    def __str__(self) -> str:
        msg = "Height Command."
        return msg

    def _update_metrics(self):
        env = self._env
        asset: Articulation = env.scene["robot"]
        demo_height = env.motion_manager.loc_height
        cur_height = asset.data.root_pos_w[:, 2]
        diff_height = torch.abs(cur_height - demo_height)
        self.metrics["diff_height"] =diff_height

    @property
    def command(self):
        return self._env.motion_manager.loc_height.reshape(-1, 1)
    
class RootVelCommand(BaseCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        env = self._env
        asset: Articulation = env.scene["robot"]
        self.robot = asset


    def __str__(self) -> str:
        msg = "Root vel command from motion."
        return msg

    def _update_metrics(self):
        diff = self.robot.data.root_lin_vel_b - self._env.motion_manager.loc_root_vel * self._env.motion_manager.speed_scale
        self.metrics["diff_root_vel"] = torch.norm(diff, dim=1)

    def _set_debug_vis_impl(self, debug_vis):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            self.vel_drawer = SingleVelDrawer(self.device)
        else:
            self.vel_drawer = None

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if self.vel_drawer is not None:
            self.vel_drawer.visualize_self_vel(self.robot)
            self.vel_drawer.visualize_ref_vel(self.robot, self._env.motion_manager.loc_root_vel * self._env.motion_manager.speed_scale)

    @property
    def command(self):
        return self._env.motion_manager.loc_root_vel * self._env.motion_manager.speed_scale
    
class RootAngVelCommand(BaseCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        
        env = self._env
        asset: Articulation = env.scene["robot"]
        self.robot = asset

    def __str__(self) -> str:
        msg = "Root angular vel command from motion."
        return msg

    def _update_metrics(self):

        diff = self.robot.data.root_ang_vel_b - self._env.motion_manager.loc_ang_vel * self._env.motion_manager.speed_scale
        self.metrics["diff_root_ang_vel"] = torch.abs(diff)

    def _set_debug_vis_impl(self, debug_vis):
        super()._set_debug_vis_impl(debug_vis)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)

    @property
    def command(self):
        return self._env.motion_manager.loc_ang_vel
    
from trackerLab.utils.torch_utils import euler_from_quaternion
    
class RootPoseCommand(BaseCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        
        env = self._env
        asset: Articulation = env.scene["robot"]
        self.robot = asset
        # demo_roll, demo_pitch, demo_yaw = euler_from_quaternion(env.motion_manager.loc_root_rot)

    def __str__(self) -> str:
        msg = "Root angular vel command from motion."
        return msg

    def _update_metrics(self):

        diff = self.robot.data.root_ang_vel_b - self._env.motion_manager.loc_ang_vel
        self.metrics["diff_root_ang_vel"] = torch.abs(diff)

    def _set_debug_vis_impl(self, debug_vis):
        super()._set_debug_vis_impl(debug_vis)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)

    @property
    def command(self):
        return self._env.motion_manager.loc_ang_vel
    
