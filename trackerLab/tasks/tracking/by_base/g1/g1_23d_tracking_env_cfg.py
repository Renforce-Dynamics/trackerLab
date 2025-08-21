import torch
from isaaclab.utils import configclass
from trackerLab.tracker_env.manager_based_tracker_env import ManagerBasedTrackerEnvCfg
from trackerLab.assets.humanoids.g1 import G1_23D_CFG
from trackerLab.tracker_env.manager_based_amp_tracker_env_cfg import ManagerBasedAMPTrackerEnvCfg

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
class G1AMPTrackingEnvCfg(ManagerBasedAMPTrackerEnvCfg):
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
        
        self.reference_body = "torso_link"
        self.key_body_names = ["l_ankle_pitch_link", "r_ankle_pitch_link", "l_claw_link", "r_claw_link"]
        self.num_amp_observations = 2
        self.num_amp_observation_space = 59
        


@configclass
class G1TrackingWalk(G1TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/g1_23d/simple_walk.yaml"

@configclass
class G1AMPTrackingWalk(G1AMPTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/g1_23d/simple_walk.yaml"
