import torch
from isaaclab.utils import configclass

from trackerLab.managers.motion_manager import MotionManagerCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import RecorderTerm, RecorderManagerBaseCfg

from isaaclab.managers import SceneEntityCfg

from trackerLab.motion_buffer.motion_buffer_cfg import MotionBufferCfg
import trackerLab.tracker_env.mdp.tracker.observation as mdp_obs
import trackerLab.tracker_env.mdp.tracker.reward as mdp_rew

from factoryIsaac.tasks.humanoid.run.hurdle_r2 import R2Rewards, R2Hurdle

from factoryIsaac.assets.config.r2 import R2_SUB_CFG, R2_RD_CFG, R2_CFG

from trackerLab.commands.base_command import SelfTransCommandCfg
import trackerLab.commands.manager.commands_cfg as cmd
@configclass
class TrackCommands:
    self_trans_command = SelfTransCommandCfg(
        debug_vis=True
    )
    dofpos_command = cmd.DofposCommandCfg(
        debug_vis=True
    )
    height_command = cmd.HeightCommandCfg()

def alive_func(env):
    return torch.ones((env.num_envs, ), device=env.device)

@configclass
class R2TrackRewards(R2Rewards):
    
    demo_height = RewTerm(
        func=mdp_rew.reward_tracking_demo_height, 
        weight=0.0
    )
    
    motion_l1_whb_dof_pos = RewTerm(
        func=mdp_rew.reward_motion_l1_whb_dof_pos, 
        weight = -1.0
    )
    
    motion_exp_whb_dof_pos = RewTerm(
        func=mdp_rew.reward_motion_exp_whb_dof_pos, 
        weight = 20.0,
        params={
            "factor": 0.3
        }
    )
    
    motion_base_lin_vel = RewTerm(
        func=mdp_rew.reward_motion_base_lin_vel, 
        params = {
            "vel_scale": 0.6
        },
        weight=2.0
    )
    
    motion_base_ang_vel = RewTerm(
        func=mdp_rew.reward_motion_base_ang_vel, 
        weight=0.1
    )
    
    punish_base_ang_vel = RewTerm(
        func=mdp_rew.punish_base_ang_vel, 
        weight=0.0
    )

    reward_alive = RewTerm(
        func = alive_func,
        weight = 0.1
    )

@configclass
class R2TrackObs(ObservationsCfg):
    @configclass
    class Policy(ObservationsCfg.PolicyCfg):
        demo_root_vel = ObsTerm(
            func=mdp_obs.demo_root_vel,
        )
        demo_ang_vel = ObsTerm(
            func=mdp_obs.demo_ang_vel,
        )
        motion_dof_pos_whb = ObsTerm(
            func=mdp_obs.motion_dof_pos_whb,
        )
        
    policy: Policy = Policy()


@configclass
class R2TrackEnvCfg(R2Hurdle):
    rewards: R2TrackRewards = R2TrackRewards()
    observations: R2TrackObs = R2TrackObs()
    # Motion settings
    motion: MotionManagerCfg = MotionManagerCfg(
        motion_buffer_cfg=MotionBufferCfg(
                motion=MotionBufferCfg.MotionCfg(
                    motion_name=None
                    ),
                regen_pkl=False,
            ),
        static_motion=False,
        robot_type="r2y",
        obs_from_buffer=False,
        reset_to_pose=False
    )
    
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = R2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        self.episode_length_s = 4.0
        # self.scene.robot.spawn.articulation_props.fix_root_link = True
        # self.scene.robot.spawn.rigid_props.disable_gravity = True
        
        # Loco
        self.rewards.joint_deviation_hip = None
        self.rewards.joint_deviation_arms = None
        
        # Commands
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time = None
        self.rewards.joint_deviation_arms = None
        self.rewards.termination_penalty.weight = -300
        self.rewards.lin_vel_z_l2 = None
        self.rewards.action_rate_l2.weight = -0.01
        
        self.rewards.dof_torques_l2 = None
        self.rewards.dof_pos_limits = None
        # self.rewards.feet_slide = None
        # self.rewards.motion_l1_whb_dof_pos = None
        self.rewards.flat_orientation_l2 = None
        
        self.commands.base_velocity = None
        self.observations.policy.velocity_commands = None
        
        self.commands = TrackCommands()

    def zero_cmd(self):
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

@configclass
class R2TrackWalk(R2TrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/r2y/simple_walk.yaml"

@configclass
class R2TrackRun(R2TrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/r2y/simple_run.yaml"
  
@configclass
class R2TrackJumpOver(R2TrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/r2y/simple_jump_over.yaml"
        