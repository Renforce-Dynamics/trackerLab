import torch
from isaaclab.utils import configclass
from trackerLab.tracker_env.manager_based_tracker_env_cfg import ManagerBasedTrackerEnvCfg
from trackerLab.assets.humanoids.r2 import R2_CFG

@configclass
class R2TrackingEnvCfg(ManagerBasedTrackerEnvCfg):
    def __post_init__(self):
        self.set_no_scanner()
        self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.robot_type = "r2y"

        self.scene.robot = R2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.adjust_contact(["base_link", ".*_hip_.*", ".*_knee_.*", "waist_.*", ".*_shoulder_.*", ".*_arm_.*"])
        self.adjust_external_events(["base_link"])
        


@configclass
class R2TrackingWalk(R2TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/r2y/simple_walk.yaml"
