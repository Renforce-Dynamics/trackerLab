import torch
import math
from isaaclab.utils import configclass
from trackerLab.tasks.tracking.humanoid import TrackingHumanoidEnvCfg, HumanoidRewardsCfgV2
from trackerLab.assets.humanoids.r2 import R2_CFG
from .motion_align_cfg import R2B_MOTION_ALIGN_CFG_SUB, R2B_MOTION_ALIGN_CFG
import trackerLab.tracker_env.mdp.tracker.reward as tracker_reward
import trackerLab.tracker_env.mdp as mdp

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

@configclass
class R2Rewards(HumanoidRewardsCfgV2):
    arm_deviation = RewTerm(func=mdp.joint_deviation_l1,  weight=-2,
                                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_arm_pitch_.*", ".*_shoulder_yaw_.*"])})

    punish_base_ang_vel = RewTerm(func=tracker_reward.punish_base_ang_vel,  weight=-1)

@configclass
class R2TrackingEnvCfg(TrackingHumanoidEnvCfg):
    rewards:R2Rewards = R2Rewards()
    def __post_init__(self):
        self.set_no_scanner()
        # self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.robot_type = "r2b"
        self.motion.set_motion_align_cfg(R2B_MOTION_ALIGN_CFG_SUB)

        self.scene.robot = R2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.observations.policy.base_lin_vel.scale = 1.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.actions.scale = 1.0
        
        # self.observations.policy.set_history(5)
        
        self.actions.joint_pos.scale = 0.25
        
        
        self.adjust_contact(["base_link", ".*_hip_.*", ".*_knee_.*", "waist_.*", ".*_shoulder_.*", ".*_arm_.*"])
        self.adjust_external_events(["base_link"])
        
        # self.terminations.base_contact
        self.terminations.base_height = None
        self.terminations.bad_orientation = None
        
        self.episode_length_s = 4
        
        self.rewards.set_feet([".*ankle_roll.*"])
        self.rewards.body_orientation_l2.params["asset_cfg"].body_names = ["base_link"]
        
        self.rewards.motion_whb_dof_pos.weight = 5
        self.rewards.motion_whb_dof_pos.params["std"] = 2
        
        self.rewards.motion_whb_dof_pos_punish.weight = -3
        self.rewards.motion_base_ang_vel_punish.weight = -2
        self.rewards.motion_base_lin_vel_punish.weight = -2
        
        
        self.rewards.motion_base_ang_vel.weight = 3
        self.rewards.motion_base_ang_vel.params["std"] = 2
        
        self.rewards.motion_base_lin_vel.weight = 3
        self.rewards.motion_base_lin_vel.params["std"] = 1.5
        
        self.rewards.feet_slide.weight = -1.5
        self.rewards.feet_stumble.weight = -1.5
        
        self.rewards.waists_deviation = None
        self.rewards.arm_deviation = None
        
        self.rewards.alive.weight = 5
        
        self.rewards.dof_acc_l2.params = {
            "asset_cfg": SceneEntityCfg(
                name = "robot",
                joint_names=[
                    ".*_hip_.*",
                    ".*_knee_joint"
                ]
            )
        }
        self.rewards.dof_vel_l2 = None
        
        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, -1.0)

@configclass
class R2TrackingWalk(R2TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/r2b/simple_walk.yaml"
        
        # self.align_friction()
        self.events.set_event_determine()
        
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

