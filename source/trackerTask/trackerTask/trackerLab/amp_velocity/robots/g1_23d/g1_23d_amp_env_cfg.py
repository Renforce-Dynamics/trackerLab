import torch
from isaaclab.utils import configclass
from trackerLab.tracker_env.manager_based_tracker_env import ManagerBasedTrackerEnvCfg
from robotlib.trackerLab.assets.unitree import UNITREE_G1_23DOF_CFG

from .g1_env_cfg import G1FlatEnvCfg

@configclass
class G1AMPEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        # self.set_no_scanner()
        # self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.robot_type = "g1_23d"

        self.scene.robot = UNITREE_G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.adjust_contact([
                "pelvis.*", ".*shoulder.*", "torso_link", ".*elbow.*", ".*wrist.*", ".*head.*"
            ])
        # self.adjust_external_events(["torso_link"])
        


@configclass
class G1AMPWalk(G1AMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion_name = "amass/g1_23d/simple_walk.yaml"
