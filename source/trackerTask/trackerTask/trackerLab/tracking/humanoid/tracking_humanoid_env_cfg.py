import math
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm

from trackerLab.tracker_env.manager_based_tracker_env import ManagerBasedTrackerEnvCfg
from trackerLab.tracker_env.manager_based_tracker_env.manager_based_tracker_env_cfg.configs import (
    RewardsCfg,
    TerminationsCfg,
    ObservationsCfg
)
from isaaclab.managers import SceneEntityCfg

import trackerLab.tracker_env.mdp as mdp

from trackerTask.trackerLab.playground import COBBLESTONE_ROAD_CFG

@configclass
class HumanoidTerminationCfg(TerminationsCfg):
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class HumanoidRewardsCfgV2:
    # task rewards
    motion_whb_dof_pos  = RewTerm(func=mdp.motion_whb_dof_pos_subset_exp, 
                                  params={"std": math.sqrt(2)},
                                  weight=5.0)
    
    motion_whb_dof_pos_punish = RewTerm(
        func=mdp.punish_motion_l1_whb_dof_pos_subset, 
        weight = -1.0
    )
    
    motion_base_lin_vel_punish = RewTerm(func=mdp.punish_motion_l1_lin_vel, weight=-0.6)
    motion_base_ang_vel_punish = RewTerm(func=mdp.punish_motion_l1_ang_vel, weight=-0.6)
    
    motion_base_lin_vel = RewTerm(
        func=mdp.motion_lin_vel_xy_yaw_frame_exp,
        params={"std": 0.5, "vel_scale": 1.0}, weight=1.0
    )
    
    motion_base_ang_vel = RewTerm(
        func=mdp.motion_ang_vel_z_world_exp,
        params={"std": 0.5}, weight=0.5
    )
    motion_roll_pitch_world_exp = RewTerm(
        func=mdp.motion_roll_pitch_world_exp,
        params={"std": 1}, weight=3
    )
    motion_base_lin_vel = RewTerm(func=mdp.motion_lin_vel_xy_yaw_frame_exp,
                                  params={"std": 0.5},
                                  weight=1.0)
    
    # base rewards
    lin_vel_z_l2        = RewTerm(func=mdp.lin_vel_z_l2,        weight=-1.0)
    ang_vel_xy_l2       = RewTerm(func=mdp.ang_vel_xy_l2,       weight=-0.05)
    dof_vel_l2          = RewTerm(func=mdp.joint_vel_l2,        weight=-0.001)
    dof_acc_l2          = RewTerm(func=mdp.joint_acc_l2,        weight=-2.5e-7)
    energy              = RewTerm(func=mdp.energy,              weight=-2e-5)
    action_rate_l2      = RewTerm(func=mdp.action_rate_l2,      weight=-0.15)
    dof_pos_limits      = RewTerm(func=mdp.joint_pos_limits,    weight=-2.0)
    alive               = RewTerm(func=mdp.is_alive,            weight=0.05)

    # contact rewards
    undesired_contacts  = RewTerm(
        func=mdp.undesired_contacts,  
        weight=-1e-2,
        params={"sensor_cfg": 
            SceneEntityCfg(
                "contact_forces", 
                body_names=[
                    ".*shoulder.*", 
                    ".*elbow.*", 
                    ".*wrist.*",
                    "torso_link",
                    "pelvis.*",
                    ".*hip.*",
                    ".*knee.*"]
                ),
                "threshold": 1.0}
        )
    
    # gravity rewards
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1e-3)
    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, weight=-1e-3,
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})

    # termination rewards
    termination_penalty = RewTerm(func=mdp.is_terminated,       weight=-200.0)

    # humanoid specific rewards
    feet_slide          = RewTerm(func=mdp.feet_slide,          weight=-1.50,
                                  params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                                          "asset_cfg":  SceneEntityCfg("robot", body_names=".*ankle_roll.*"),},)
    feet_force          = RewTerm(func=mdp.body_force,          weight=-1e-2,
                                  params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                                          "threshold": 500, "max_reward": 400})
    feet_too_near       = RewTerm(func=mdp.feet_too_near,       weight=-0.5,
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"), 
                                          "threshold": 0.2})
    feet_stumble        = RewTerm(func=mdp.feet_stumble,        weight=-0.5,
                                  params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")})
    feet_async_stable   = RewTerm(func=mdp.feet_async_stable,   weight=-2.0,
                                  params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                                          "n_dt": 2.0})
    
    # joint deviation rewards
    waists_deviation    = RewTerm(func=mdp.joint_deviation_l1,  weight=-0.1,
                                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*"])})
    arms_deviation      = RewTerm(func=mdp.joint_deviation_l1,  weight=-0.01,
                                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*shoulder.*", ".*elbow.*", ".*wrist.*"])})
    legs_deviation      = RewTerm(func=mdp.joint_deviation_l1,  weight=-0.01,
                                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip.*", ".*knee.*", ".*ankle.*"])})

    def set_feet(self, names):
        self.feet_slide.params["sensor_cfg"].body_names = names
        self.feet_slide.params["asset_cfg"].body_names = names
        self.feet_force.params["sensor_cfg"].body_names = names
        self.feet_too_near.params["asset_cfg"].body_names = names
        self.feet_stumble.params["sensor_cfg"].body_names = names

@configclass
class TrackingHumanoidEnvCfg(ManagerBasedTrackerEnvCfg):
    rewards: HumanoidRewardsCfgV2 = HumanoidRewardsCfgV2()
    terminations: HumanoidTerminationCfg = HumanoidTerminationCfg()
    
    def __post_init__(self):
        super().__post_init__()
        
    def align_friction(self):
        self.scene.terrain.physics_material.dynamic_friction = 0.45
        self.scene.terrain.physics_material.static_friction = 0.5
        
    def domain_randomization(self):
        # Reset Terms
        self.events.reset_base.params = {
                "pose_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-1.0, 1.0),
                    "y": (-1.0, 1.0),
                    "z": (-1.0, 1.0),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            }
        self.events.reset_robot_joints.params = {
                "position_range": (0.5, 0.5),
                "velocity_range": (-0.5, 0.5),
            }

        # Push terms
        self.events.push_robot.params = {
                "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "z": (-0.8, 1.5)}
            }
        

    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)