from isaaclab.utils import configclass
from trackerLab.tracker_env.manager_based_tracker_env import ManagerBasedTrackerEnvCfg

@configclass
class TrackingHumanoidEnvCfg(ManagerBasedTrackerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.observations.policy.base_lin_vel.scale = 1.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.actions.scale = 1.0