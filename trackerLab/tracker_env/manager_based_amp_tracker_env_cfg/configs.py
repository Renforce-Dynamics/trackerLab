from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from trackerLab.tracker_env.manager_based_tracker_env_cfg import ObservationsCfg, MotionCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
# TODO: amp_mdp放的位置好像不对，需要调一下
# TODO: key_body_positions的key_body是哪些需要定义一下，可能会写到Cfg里
from .amp_mdp import observations as amp_mdp

@configclass
class AMPObservationsCfg(ObservationsCfg):
    @configclass
    class AmpCfg(ObsGroup):
        """Observations for the policy."""
        dof_positions = ObsTerm(func=mdp.joint_pos)
        dof_velocities = ObsTerm(func=mdp.joint_vel)
        root_positions = ObsTerm(func=amp_mdp.body_pos_w)
        root_rotations =ObsTerm(func=amp_mdp.body_quat_w)
        root_linear_velocities =ObsTerm(func=amp_mdp.body_lin_vel_w)
        root_angular_velocities =ObsTerm(func=amp_mdp.body_ang_vel_w)
        key_body_positions =ObsTerm(func=amp_mdp.key_body_pos_w)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    amp_obs: AmpCfg = AmpCfg()
    
@configclass
class AMPMotionManagerCfg(MotionCfg):
    amp = True
    pass