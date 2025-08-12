from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.flat_env_cfg import H1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1Rewards
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm

import trackerLab.tracker_env.mdp.tracker.observation as mdp_obs
import trackerLab.tracker_env.mdp.tracker.reward as mdp_rew

from trackerLab.managers.motion_manager import MotionManagerCfg

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


@configclass
class H1TrackRewards(H1Rewards):
    demo_height = RewTerm(
        func=mdp_rew.reward_tracking_demo_height, 
        weight=1.0
    )
    
    motion_l1_whb_dof_pos = RewTerm(
        func=mdp_rew.reward_motion_l1_whb_dof_pos, 
        weight = -1.0
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
        weight=0.2
    )
    
    punish_base_ang_vel = RewTerm(
        func=mdp_rew.punish_base_ang_vel, 
        weight=1.0
    )
    

@configclass
class H1TrackObs(ObservationsCfg):
    @configclass
    class H1Policy(ObservationsCfg.PolicyCfg):
        demo_root_vel = ObsTerm(
            func=mdp_obs.demo_root_vel,
        )
        demo_height = ObsTerm(
            func=mdp_obs.demo_height,
        )
        demo_ang_vel = ObsTerm(
            func=mdp_obs.demo_ang_vel,
        )
        motion_dof_pos_whb = ObsTerm(
            func=mdp_obs.motion_dof_pos_whb,
        )
        
    policy: H1Policy = H1Policy()


@configclass
class H1TrackEnvCfg(H1FlatEnvCfg):
    rewards: H1TrackRewards = H1TrackRewards()
    observations: H1TrackObs = H1TrackObs()
    # Motion settings
    motion: MotionManagerCfg = MotionManagerCfg(
        static_motion = False,
        robot_type="H1"
    )
    
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 5.0
        # self.scene.robot.spawn.articulation_props.fix_root_link = True
        # self.scene.robot.spawn.rigid_props.disable_gravity = True
        
        self.motion.static_motion = False
        self.motion.obs_from_buffer = True
        self.motion.speed_scale = 1.0
        
        # Commands
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time = None
        self.rewards.joint_deviation_arms = None
        self.rewards.termination_penalty.weight = -300
        
        self.commands.base_velocity = None
        self.observations.policy.velocity_commands = None

    def zero_cmd(self):
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

@configclass
class H1TrackAll(H1TrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "motions_autogen_all.yaml"

@configclass
class H1TrackDebugPunch(H1TrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "motions_autogen_debug_punch.yaml"


@configclass
class H1TrackDebugWalk(H1TrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "skill_graph/simple_walk.yaml"
        
@configclass
class H1TrackRun(H1TrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.motion.motion_buffer_cfg.motion.motion_name = "motions_autogen_run.yaml"
        self.motion.speed_scale = 1
        
@configclass
class H1TrackJumpOver(H1TrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.motion.motion_buffer_cfg.motion.motion_name = "re_jump_over.yaml"
        self.motion.speed_scale = 1
        
@configclass
class H1TrackJump(H1TrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_base_lin_vel.weight = -0.3
        self.rewards.motion_base_ang_vel.weight = -0.2
        self.rewards.motion_l1_whb_dof_pos.weight = -0.1
        self.rewards.motion_base_lin_vel.params["vel_scale"] = 1.1
        self.rewards.demo_height.weight = 2
        
        self.motion.motion_buffer_cfg.motion.motion_name = "motions_autogen_jump.yaml"
        self.motion.speed_scale = 1
        
@configclass
class H1TrackWalkRunNavi(H1TrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.motion_base_lin_vel.weight = -0.3
        self.rewards.motion_base_ang_vel.weight = -0.2
        self.rewards.motion_l1_whb_dof_pos.weight = -0.1
        self.rewards.motion_base_lin_vel.params["vel_scale"] = 1.1
        
        self.motion.motion_buffer_cfg.motion.motion_name = "motions_autogen_walk_run_navigate.yaml"
        self.motion.speed_scale = 1