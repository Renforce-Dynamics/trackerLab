from isaaclab.utils import configclass

from robotlib.trackerLab.assets.humanoids.r2 import R2_CFG, R2_ACTION_SCALE
from trackerTask.beyondMimic.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from trackerTask.beyondMimic.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class R2FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = R2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = R2_ACTION_SCALE
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "left_shoulder_roll_link",
            "left_arm_pitch_link",
            "right_shoulder_roll_link",
            "right_arm_pitch_link",
        ]
        self.commands.motion.debug_vis = False
        
        self.commands.motion.motion_file = "./motion.npz"
        
        self.events.base_com.params["asset_cfg"].body_names = "base_link"
        
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class R2FlatWoStateEstimationEnvCfg(R2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class R2FlatLowFreqEnvCfg(R2FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
