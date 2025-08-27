from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from trackerLab.tracker_env.manager_based_tracker_env import ManagerBasedTrackerEnvCfg
from trackerLab.tracker_env.manager_based_tracker_env.manager_based_tracker_env_cfg.configs import (
    RewardsCfg,
    TerminationsCfg,
    ObservationsCfg
)
from isaaclab.managers import SceneEntityCfg

from trackerLab.tasks.playground import COBBLESTONE_ROAD_CFG

@configclass
class HumanoidTerminationCfg(TerminationsCfg):
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    def __post_init__(self):
        pass

@configclass
class HuamnoidRewardsCfg(RewardsCfg):
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    
    def set_no_deviation(self):
        self.joint_deviation_arms = None
        self.joint_deviation_waists = None


@configclass
class TrackingHumanoidEnvCfg(ManagerBasedTrackerEnvCfg):
    rewards: HuamnoidRewardsCfg = HuamnoidRewardsCfg()
    terminations: HumanoidTerminationCfg = HumanoidTerminationCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # self.decimation = 20
        # self.sim.dt = 0.001
        
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
            },
        self.events.reset_robot_joints.params = {
                "position_range": (1.0, 1.0),
                "velocity_range": (-1.0, 1.0),
            },

        # Push terms
        self.events.push_robot.params = {
                "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "z": (-0.8, 1.5)}
            }
        

