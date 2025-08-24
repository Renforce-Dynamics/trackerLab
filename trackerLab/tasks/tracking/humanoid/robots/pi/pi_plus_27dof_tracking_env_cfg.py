import torch
from isaaclab.utils import configclass
from trackerLab.tasks.tracking.humanoid import TrackingHumanoidEnvCfg
from trackerLab.assets.humanoids.pi import PI_PLUS_27DOF_CFG

@configclass
class PiTrackingEnvCfg(TrackingHumanoidEnvCfg):
    def __post_init__(self):
        self.set_no_scanner()
        self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.robot_type = "pi_plus_27dof"
        
        self.observations.policy.base_lin_vel.scale = 1.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.actions.scale = 1.0
        
        self.actions.joint_pos.scale = 0.25

        self.scene.robot = PI_PLUS_27DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.adjust_contact([
            "base_link",
            ".*_hip_.*", ".*_calf_.*", 
            "waist_.*", ".*_shoulder_.*", ".*_claw_.*",
            ".*_elbow_.*", ".*_wrist_.*"
            ])
        self.adjust_external_events(["base_link"])
        


@configclass
class PiTrackingWalk(PiTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/pi_plus_27dof/simple_walk.yaml"
        self.observations.policy.set_no_noise()
        self.events.set_event_determine()
        # self.commands.dofpos_command.verbose_detail = True
        # self.set_test_motion_mode()
        
@configclass
class PiTrackingRun(PiTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/pi_plus_27dof/simple_run.yaml"
        self.observations.policy.set_no_noise()
        self.events.set_event_determine()
        self.commands.dofpos_command.verbose_detail = True
        # self.set_test_motion_mode()
        
        self.rewards.reward_alive.weight = 5
        
@configclass
class PiTrackingJump(PiTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/pi_plus_27dof/simple_jump.yaml"
        self.observations.policy.set_no_noise()
        self.events.set_event_determine()
        # self.commands.dofpos_command.verbose_detail = True
        # self.set_test_motion_mode()
        
        self.rewards.reward_alive.weight = 5