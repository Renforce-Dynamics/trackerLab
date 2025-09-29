import torch
import math
from isaaclab.utils import configclass
from trackerLab.tasks.tracking.humanoid import TrackingHumanoidEnvCfg, HumanoidRewardsCfgV2
from robotlib.trackerLab.assets.humanoids.r2 import R2_CFG
from .motion_align_cfg import R2B_MOTION_ALIGN_CFG_SUB, R2B_MOTION_ALIGN_CFG
import trackerLab.tracker_env.mdp.tracker.reward as tracker_reward
import trackerLab.tracker_env.mdp as mdp

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

@configclass
class R2TrackingEnvCfg(TrackingHumanoidEnvCfg):
    def __post_init__(self):
        self.set_no_scanner()
        # self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.robot_type = "r2b"
        self.motion.set_motion_align_cfg(R2B_MOTION_ALIGN_CFG_SUB)
        self.scene.robot = R2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Modify Observation
        
        self.actions.joint_pos.scale = 0.5

        self.episode_length_s = 20.0
        
        self.adjust_contact(["base_link", ".*_hip_.*", ".*_knee_.*", "waist_.*", ".*_shoulder_.*", ".*_arm_.*"])
        self.adjust_external_events(["base_link"])
        
        self.rewards.set_feet([".*ankle_roll.*"])
        self.rewards.body_orientation_l2.params["asset_cfg"].body_names = ["base_link"]
        
        self.rewards.dof_acc_l2.params = {
            "asset_cfg": SceneEntityCfg(
                name = "robot", joint_names=[ ".*_hip_.*", ".*_knee_joint" ]
            )
        }
        self.rewards.arms_deviation.params["asset_cfg"].joint_names = [".*_arm_pitch_.*", ".*_shoulder_yaw_.*"]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            ".*shoulder.*", ".*arm.*", "base_link", "waist.*", ".*hip.*", ".*knee.*"
        ]
        
        
        self.actions.joint_pos.clip = {
            ".*arm.*": (-0.5, 0.5),
            ".*shoulder.*": (-0.5, 0.5)
        }

@configclass
class R2TrackingWalk(R2TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion_name = "amass/r2b/simple_run.yaml"
        
        self.rewards.feet_slide.weight = -0.5
        self.rewards.feet_too_near.weight = -0.05
        self.rewards.feet_stumble.weight = -0.01
        self.rewards.motion_whb_dof_pos.weight = 3.0
        self.rewards.motion_base_ang_vel.weight = 0.5
        self.rewards.motion_base_lin_vel.weight = 3.0
        self.rewards.alive.weight = 3.0
        
        self.align_friction()
        self.domain_randomization()
        # self.events.set_event_determine()
        
@configclass
class R2TrackingRun(R2TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion_name = "amass/r2b/simple_run.yaml"
        self.rewards.feet_slide.weight = -0.5
        self.rewards.feet_too_near.weight = -0.05
        self.rewards.feet_stumble.weight = -0.01
        self.rewards.motion_whb_dof_pos.weight = 3.0
        self.rewards.motion_base_ang_vel.weight = 0.5
        self.rewards.motion_base_lin_vel.weight = 3.0
        self.rewards.alive.weight = 3.0
        self.align_friction()
        self.domain_randomization()
        

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

