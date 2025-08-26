import torch
from isaaclab.utils import configclass
from trackerLab.tasks.tracking.humanoid import TrackingHumanoidEnvCfg
from trackerLab.assets.humanoids.r2 import R2_CFG
from .motion_align_cfg import R2B_MOTION_ALIGN_CFG

@configclass
class R2TrackingEnvCfg(TrackingHumanoidEnvCfg):
    def __post_init__(self):
        self.set_no_scanner()
        self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.robot_type = "r2b"
        self.motion.set_motion_align_cfg(R2B_MOTION_ALIGN_CFG)

        self.scene.robot = R2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.adjust_contact(["base_link", ".*_hip_.*", ".*_knee_.*", "waist_.*", ".*_shoulder_.*", ".*_arm_.*"])
        self.adjust_external_events(["base_link"])
        


@configclass
class R2TrackingWalk(R2TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/r2b/simple_walk.yaml"
        self.observations.policy.set_no_noise()
        self.events.set_event_determine()
        # self.commands.dofpos_command.verbose_detail = True
        
        
@configclass
class R2TrackingRun(R2TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/r2b/simple_run.yaml"
        self.observations.policy.set_no_noise()
        self.events.set_event_determine()
        # self.commands.dofpos_command.verbose_detail = True
        
        self.rewards.motion_base_lin_vel.weight = 0.5
        self.rewards.motion_base_lin_vel_x.weight = 0.5
        
        self.rewards.reward_alive.weight = 5
        # self.set_test_motion_mode()
