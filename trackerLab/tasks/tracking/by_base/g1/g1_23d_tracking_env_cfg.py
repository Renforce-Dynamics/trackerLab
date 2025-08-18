import torch
from isaaclab.utils import configclass
from trackerLab.tracker_env.manager_based_tracker_env import ManagerBasedTrackerEnvCfg
from trackerLab.assets.humanoids.g1 import G1_23D_CFG

@configclass
class G1TrackingEnvCfg(ManagerBasedTrackerEnvCfg):
    def __post_init__(self):
        self.set_no_scanner()
        self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.robot_type = "g1_23d"

        self.scene.robot = G1_23D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.adjust_contact([
                "pelvis.*", ".*shoulder.*", "torso_link", ".*elbow.*", ".*wrist.*", ".*head.*"
            ])
        self.adjust_external_events(["torso_link"])
        


@configclass
class G1TrackingWalk(G1TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/g1_23d/simple_walk.yaml"
