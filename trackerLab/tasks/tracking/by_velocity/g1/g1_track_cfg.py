import torch
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1Rewards
from trackerLab.managers.motion_manager import MotionManagerCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm

import trackerLab.tracker_env.mdp.tracker.observation as mdp_obs
import trackerLab.tracker_env.mdp.tracker.reward as mdp_rew

from trackerLab.commands.base_command import SelfTransCommandCfg
import trackerLab.commands.manager.commands_cfg as cmd
@configclass
class TrackCommands:
    self_trans_command = SelfTransCommandCfg(
        debug_vis=True
    )
    dofpos_command = cmd.DofposCommandCfg(
        debug_vis=True,
        verbose_detail=False
    )
    # height_command = cmd.HeightCommandCfg()

def alive_func(env):
    return torch.ones((env.num_envs, ), device=env.device)

@configclass
class G1CmdTrackRewards(G1Rewards):
    
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
        weight = 5.0,
        params={
            "factor": 0.3
        }
    )
    
    motion_base_lin_vel = RewTerm(
        func=mdp_rew.reward_motion_base_lin_vel, 
        params = {
            "vel_scale": 1.0
        },
        weight=2.0
    )
    
    motion_base_ang_vel = RewTerm(
        func=mdp_rew.reward_motion_base_ang_vel, 
        weight=0.0
    )
    
    punish_base_ang_vel = RewTerm(
        func=mdp_rew.punish_base_ang_vel, 
        weight=-1.0
    )
    
    reward_alive = RewTerm(
        func = alive_func,
        weight = 1.0
    )

@configclass
class G1TrackObs(ObservationsCfg):
    @configclass
    class G1Policy(ObservationsCfg.PolicyCfg):
        demo_root_vel = ObsTerm(
            func=mdp_obs.demo_root_vel,
        )
        demo_root_vel = ObsTerm(
            func=mdp_obs.demo_root_vel,
        )
        demo_ang_vel = ObsTerm(
            func=mdp_obs.demo_ang_vel,
        )
        motion_dof_pos_whb = ObsTerm(
            func=mdp_obs.motion_dof_pos_whb,
        )
        
    policy: G1Policy = G1Policy()

from trackerLab.assets.humanoids.g1 import G1_29D_LOCO_CFG, G1_23D_CFG

@configclass
class G1TrackEnvCfg(G1FlatEnvCfg):
    rewards: G1CmdTrackRewards = G1CmdTrackRewards()
    observations: G1TrackObs = G1TrackObs()
    # Motion settings
    motion: MotionManagerCfg = MotionManagerCfg(
        # robot_type="g1_23d",
        robot_type="g1_29d_loco",
    )
    
    def __post_init__(self):
        super().__post_init__()
        # self.scene.robot = G1_23D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot = G1_29D_LOCO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.episode_length_s = 5.0
        # self.scene.robot.spawn.articulation_props.fix_root_link = True
        # self.scene.robot.spawn.rigid_props.disable_gravity = True
        
        self.motion.static_motion = False
        self.motion.obs_from_buffer = False
        self.motion.speed_scale = 0.8
        self.motion.motion_buffer_cfg.regen_pkl = True
        
        # Commands
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time = None
        self.rewards.joint_deviation_arms = None
        self.rewards.termination_penalty.weight = -300
        self.rewards.lin_vel_z_l2 = None
        self.rewards.action_rate_l2.weight = 1e-4
        
        self.rewards.joint_deviation_fingers = None
        self.rewards.joint_deviation_torso = None
        
        self.commands.base_velocity = None
        self.observations.policy.velocity_commands = None

        self.commands = TrackCommands()
        
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "pelvis.*", ".*shoulder.*", "torso_link", ".*elbow.*", ".*wrist.*", ".*head.*"
            ]
        
        

@configclass
class G1TrackWalkRun(G1TrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # self.motion.motion_buffer_cfg.motion.motion_name = "amass/g1_23d/simple_walk_run.yaml"
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/g1_29d_loco/simple_walk_run.yaml"

        