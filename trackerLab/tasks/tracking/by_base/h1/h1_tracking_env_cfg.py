import torch
from isaaclab.utils import configclass
from trackerLab.tracker_env.manager_based_tracker_env import ManagerBasedTrackerEnvCfg
from isaaclab_assets.robots.unitree import H1_MINIMAL_CFG

@configclass
class H1TrackingEnvCfg(ManagerBasedTrackerEnvCfg):
    def __post_init__(self):
        self.set_no_scanner()
        self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.robot_type = "h1"

        self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.adjust_contact([".*torso_link"])
        self.adjust_external_events([".*torso_link"])
        


@configclass
class H1TrackingWalk(H1TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/h1/simple_walk.yaml"