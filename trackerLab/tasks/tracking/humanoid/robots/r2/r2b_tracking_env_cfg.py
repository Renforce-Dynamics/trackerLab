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
        
        self.rewards.set_no_deviation()
        
        self.observations.policy.base_lin_vel.scale = 1.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.actions.scale = 1.0
        
        self.observations.policy.set_history(5)
        
        self.actions.joint_pos.scale = 0.25
        

        

@configclass
class R2TrackingWalk(R2TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/r2b/simple_walk.yaml"
        
        
@configclass
class R2TrackingRun(R2TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/r2b/simple_run.yaml"
        

@configclass
class R2TrackingWalk_Play(R2TrackingWalk):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1
        
@configclass
class R2TrackingRun_Play(R2TrackingRun):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1

