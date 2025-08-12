import torch
try:
    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG
    from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
except ModuleNotFoundError:
    print("ImportError: Unable to import isaaclab.sim or isaaclab.utils.math. Please check the module structure.")


class SingleVelDrawer(object):
    def __init__(self, device):
        self.device = device
        self._initialized = False
        super().__init__()
        
    def _initialize_vel_markers(self):
        self._initialized = True
        print("Initialize markers for Vel.")
        
        goal_vel_visualizer_cfg = GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/velocity_goal"
        )
        current_vel_visualizer_cfg = BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/velocity_current"
        )
        goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
        self.goal_vel_visualizer = VisualizationMarkers(goal_vel_visualizer_cfg)
        self.current_vel_visualizer = VisualizationMarkers(current_vel_visualizer_cfg)
        
    def visualize_ref_vel(self, robot, command):
        if not self._initialized: self._initialize_vel_markers()
        base_pos_w = robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(command[:, :2], robot)
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        
    def visualize_self_vel(self, robot):
        if not self._initialized: self._initialize_vel_markers()
        base_pos_w = robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(robot.data.root_lin_vel_b[:, :2], robot)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        
    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor, robot) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat