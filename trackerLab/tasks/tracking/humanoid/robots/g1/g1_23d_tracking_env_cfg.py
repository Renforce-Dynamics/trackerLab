import torch
from isaaclab.utils import configclass
from trackerLab.tasks.tracking.humanoid import TrackingHumanoidEnvCfg
from trackerLab.assets.humanoids.g1 import G1_23D_CFG
from .motion_align_cfg import G1_23D_MOTION_ALIGN_CFG

@configclass
class G1TrackingEnvCfg(TrackingHumanoidEnvCfg):
    def __post_init__(self):
        self.set_no_scanner()
        self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.set_motion_align_cfg(G1_23D_MOTION_ALIGN_CFG)
        self.motion.robot_type = "g1_23d"
        
        self.observations.policy.base_lin_vel.scale = 1.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.actions.scale = 1.0
        
        self.observations.policy.history_length = 5
        

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
