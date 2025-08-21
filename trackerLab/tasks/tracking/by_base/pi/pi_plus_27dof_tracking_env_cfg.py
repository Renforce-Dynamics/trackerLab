import torch
from isaaclab.utils import configclass
from trackerLab.tracker_env.manager_based_tracker_env.manager_based_tracker_env_cfg import ManagerBasedTrackerEnvCfg
from trackerLab.tracker_env.manager_based_amp_tracker_env_cfg import ManagerBasedAMPTrackerEnvCfg
from trackerLab.assets.humanoids.pi import PI_PLUS_27DOF_CFG

@configclass
class PiTrackingEnvCfg(ManagerBasedTrackerEnvCfg):
    def __post_init__(self):
        self.set_no_scanner()
        # self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.robot_type = "pi_plus_27dof"

        self.scene.robot = PI_PLUS_27DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.adjust_contact([
            "base_link",
            ".*_hip_.*", ".*_calf_.*", 
            "waist_.*", ".*_shoulder_.*", ".*_claw_.*",
            ".*_elbow_.*", ".*_wrist_.*"
            ])
        self.adjust_external_events(["base_link"])
        
@configclass
class PiAMPTrackingEnvCfg(ManagerBasedAMPTrackerEnvCfg):
    def __post_init__(self):
        self.set_no_scanner()
        # self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.robot_type = "pi_plus_27dof"

        self.scene.robot = PI_PLUS_27DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.adjust_contact([
            "base_link",
            ".*_hip_.*", ".*_calf_.*", 
            "waist_.*", ".*_shoulder_.*", ".*_claw_.*",
            ".*_elbow_.*", ".*_wrist_.*"
            ])
        self.adjust_external_events(["base_link"])
        
        self.reference_body = "waist_link"
        self.key_body_names = ["l_ankle_pitch_link", "r_ankle_pitch_link", "l_claw_link", "r_claw_link"]
        self.num_amp_observations = 2
        self.num_amp_observation_space = 65

@configclass
class PiTrackingWalk(PiTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/pi_plus_27dof/simple_walk.yaml"
        self.observations.policy.set_no_noise()
        self.events.set_event_determine()
        # self.commands.dofpos_command.verbose_detail = True
        # self.set_test_motion_mode()
        
        self.rewards.undesired_contacts = None
        


@configclass
class PiAMPTrackingWalk(PiAMPTrackingEnvCfg):
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