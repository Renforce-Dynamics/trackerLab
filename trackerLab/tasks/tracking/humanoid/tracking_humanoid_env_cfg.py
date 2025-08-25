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
    # joint_deviation_waists = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-1,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "waist.*",
    #             ],
    #         )
    #     },
    # )
    pass


@configclass
class TrackingHumanoidEnvCfg(ManagerBasedTrackerEnvCfg):
    rewards = HuamnoidRewardsCfg()
    terminations = HumanoidTerminationCfg()
    
    def __post_init__(self):
        super().__post_init__()
        # self.decimation = 20
        # self.sim.dt = 0.001