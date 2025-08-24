from isaaclab.utils import configclass
from trackerLab.tracker_env.manager_based_tracker_env import ManagerBasedTrackerEnvCfg

from trackerLab.tracker_env.manager_based_tracker_env.manager_based_tracker_env_cfg.configs import (
    RewardsCfg,
    TerminationsCfg,
    ObservationsCfg
)

@configclass
class HumanoidTerminationCfg(TerminationsCfg):
    def __post_init__(self):
        return


@configclass
class TrackingHumanoidEnvCfg(ManagerBasedTrackerEnvCfg):
    terminations = HumanoidTerminationCfg()