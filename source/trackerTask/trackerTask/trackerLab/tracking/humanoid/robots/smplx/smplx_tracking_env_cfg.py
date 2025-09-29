import torch
from isaaclab.utils import configclass
from trackerLab.tasks.tracking.humanoid import TrackingHumanoidEnvCfg
# from trackerLab.assets.humanoids.pi import PI_PLUS_25DOF_CFG
from trackerLab.assets.humanoids.smplx_humanoid import SMPLX_CFG
from .motion_align_cfg import (
    SMPLX_MOTION_ALIGN_CFG
)

@configclass
class SMPLXTrackingEnvCfg(TrackingHumanoidEnvCfg):
    def __post_init__(self):
        self.set_no_scanner()
        # self.set_plane()
        # self.adjust_scanner("Pelvis")
        super().__post_init__()
        self.motion.robot_type = "smplx"
        self.motion.set_motion_align_cfg(SMPLX_MOTION_ALIGN_CFG)
        
        self.scene.robot = SMPLX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/bodies/.*"
        self.events.add_base_mass.params["asset_cfg"].body_names = ["Torso"]
        
        self.observations.policy.base_lin_vel.scale = 1.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.actions.scale = 1.0
        self.observations.policy.set_history(5)
        
        self.actions.joint_pos.scale = 0.25
        self.episode_length_s = 12

        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "Pelvis",
            ".*_Hip", ".*_calf_.*", 
            ".*_shoulder_.*",
            ".*_elbow_.*", ".*_wrist_.*"
            ]
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            "Pelvis",
            ".*_Hip", 
            ".*_Shoulder",
            ".*_Elbow", ".*_Wrist"
            ]
        self.rewards.waists_deviation.params["asset_cfg"].joint_names = [".*_Wrist_.*"]
        # self.adjust_contact([
        #     "Pelvis",
        #     ".*_Hip", ".*_calf_.*", 
        #     ".*_shoulder_.*",
        #     ".*_elbow_.*", ".*_wrist_.*"
        #     ])
        self.terminations.base_contact = None        

        self.adjust_external_events(["Pelvis"])
        

        self.align_friction()
        self.domain_randomization()
        
        self.rewards.set_feet([".*_Ankle"])
        self.rewards.body_orientation_l2.params["asset_cfg"].body_names = ["Pelvis"]


@configclass
class SMPLXTrackingWalk(SMPLXTrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion_name = "amass/smplx/GMR_simple_walk.yaml"
        
        
        self.rewards.feet_slide.weight = -0.5
        self.rewards.feet_too_near.weight = -0.05
        self.rewards.feet_stumble.weight = -0.01
        self.rewards.motion_whb_dof_pos.weight = 3.0
        self.rewards.motion_base_ang_vel.weight = 0.5
        self.rewards.motion_base_lin_vel.weight = 3.0
        self.rewards.alive.weight = 3.0
        
# @configclass
# class PiTrackingRun(PiTrackingEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.motion.motion_buffer_cfg.motion_name = "amass/pi_plus_25dof/simple_run.yaml"
        
#         self.rewards.feet_slide.weight = -1.0
#         self.rewards.motion_whb_dof_pos.weight = 3.0
#         self.rewards.motion_base_ang_vel.weight = 0.5
#         self.rewards.motion_base_lin_vel.weight = 3.0
#         self.rewards.alive.weight = 3.0
        
# @configclass
# class PiTrackingRun_Replay(PiTrackingRun):
#     def __post_init__(self):
#         super().__post_init__()
#         self.motion.motion_buffer_cfg.motion_name = "amass/pi_plus_25dof/simple_run_replay.yaml"
#         self.motion.set_motion_align_cfg(PI_25D_MOTION_ALIGN_CFG_REPLAY)
#         self.motion.motion_buffer_cfg.motion_lib_type = "MotionLibDofPos"
#         self.motion.motion_buffer_cfg.motion_type = "replay"
        
        
# @configclass
# class PiTrackingJump(PiTrackingEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.motion.motion_buffer_cfg.motion_name = "amass/pi_plus_25dof/simple_jump.yaml"
        
        
# @configclass
# class PiTrackingWalk_Play(PiTrackingWalk):
#     def __post_init__(self):
#         super().__post_init__()
#         self.scene.num_envs = 32
#         self.scene.terrain.terrain_generator.num_rows = 2
#         self.scene.terrain.terrain_generator.num_cols = 1

# @configclass
# class PiTrackingRun_Play(PiTrackingRun):
#     def __post_init__(self):
#         super().__post_init__()
#         self.scene.num_envs = 32
#         self.scene.terrain.terrain_generator.num_rows = 2
#         self.scene.terrain.terrain_generator.num_cols = 1
        
# @configclass
# class PiTrackingJump_Play(PiTrackingJump):
#     def __post_init__(self):
#         super().__post_init__()
#         self.scene.num_envs = 32
#         self.scene.terrain.terrain_generator.num_rows = 2
#         self.scene.terrain.terrain_generator.num_cols = 1