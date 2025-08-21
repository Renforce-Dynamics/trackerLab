from isaaclab.utils import configclass
from trackerLab.tracker_env.manager_based_tracker_env.manager_based_tracker_env_cfg import ManagerBasedTrackerEnvCfg
from .configs import AMPObservationsCfg, AMPMotionManagerCfg

@configclass
class ManagerBasedAMPTrackerEnvCfg(ManagerBasedTrackerEnvCfg):
    observations:   AMPObservationsCfg = AMPObservationsCfg()
    motion:         AMPMotionManagerCfg = AMPMotionManagerCfg()