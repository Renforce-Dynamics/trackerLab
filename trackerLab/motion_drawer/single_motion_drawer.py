import torch
try:
    import isaaclab.sim as sim_utils
    import isaaclab.utils.math as math_utils
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import DEFORMABLE_TARGET_MARKER_CFG
    from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
except ModuleNotFoundError:
    print("ImportError: Unable to import isaaclab.sim or isaaclab.utils.math. Please check the module structure.")


class SingleMotionDrawer(object):
    def __init__(self, device, color=(0.0, 0.0, 1.0), prim_name="ref_motion"):
        self.device = device
        self.color = color
        self.prim_name = prim_name
        self._initialized = False
        self._active_ref_motion_markers = None
        self._self_ref_markers = None
        super().__init__()
        
    def _initialize_ref_motion_markers(self):
        self._initialized = True
        print("Initialize markers for motion joints.")
        
        # Visualizer for the active reference body positions.
        active_marker_cfg = DEFORMABLE_TARGET_MARKER_CFG.copy()
        active_marker_cfg.markers["target"].radius = 0.05
        active_marker_cfg.markers["target"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=self.color
        )  # blue
        active_marker_cfg.prim_path = f"/Visuals/Command/Motion/{self.prim_name}"
        self._active_ref_motion_markers = VisualizationMarkers(active_marker_cfg)
        self._active_ref_motion_markers.set_visibility(True)

    def visualize(self, ref_motion):
        if not self._initialized:
            self._initialize_ref_motion_markers()
        self._active_ref_motion_markers.set_visibility(True)
        self._active_ref_motion_markers.visualize(ref_motion)