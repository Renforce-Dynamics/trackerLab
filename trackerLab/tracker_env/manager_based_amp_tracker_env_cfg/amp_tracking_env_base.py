from isaaclab.utils import configclass
from trackerLab.tracker_env.manager_based_tracker_env_cfg import ManagerBasedTrackerEnvCfg
from .configs import AMPObservationsCfg, AMPMotionManagerCfg


from trackerLab.tracker_env.manager_based_tracker_env_cfg import MySceneCfg

@configclass
class ManagerBasedAMPTrackerEnvCfg(ManagerBasedTrackerEnvCfg):
    # TODO: Debug时省点显存，正式可以删掉
    scene:          MySceneCfg = MySceneCfg(num_envs=8, env_spacing=2.5)
    observations:   AMPObservationsCfg = AMPObservationsCfg()
    motion:         AMPMotionManagerCfg = AMPMotionManagerCfg()