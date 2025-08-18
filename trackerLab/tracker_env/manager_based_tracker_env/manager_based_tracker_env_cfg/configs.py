import inspect
import math
import sys
import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from trackerLab.commands.base_command import SelfTransCommandCfg
import trackerLab.commands.manager.commands_cfg as cmd

from trackerLab.tasks.playground import ROUGH_TERRAINS_CFG, FLAT_TERRAINS_CFG

import trackerLab.tracker_env.mdp.tracker.reward as tracker_reward
import trackerLab.tracker_env.mdp.tracker.observation as tracker_obs
import trackerLab.tracker_env.mdp.records as tracker_record

from trackerLab.motion_buffer.motion_buffer_cfg import MotionBufferCfg
from trackerLab.managers.motion_manager import MotionManagerCfg

from isaaclab.managers import RecorderTermCfg, RecorderManagerBaseCfg

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=FLAT_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    
    height_scanner: RayCasterCfg = MISSING
    height_scanner_base: RayCasterCfg = MISSING
    
    def __post_init__(self):
        self.height_scanner = None
        self.height_scanner_base = None
    
    def setup_scanner(self):
        try:
            # sensors
            self.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
            )
            self.height_scanner_base = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                ray_alignment="yaw",
                pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
                debug_vis=False,
                mesh_prim_paths=["/World/ground"],
            )
        except Exception as e:
            self.height_scanner = None
            self.height_scanner_base = None
            print("[Warning] scanner set fail, check isaaclab version")

@configclass
class CommandsCfg:
    self_trans_command = SelfTransCommandCfg(
        debug_vis=True
    )
    dofpos_command = cmd.DofposCommandCfg(
        debug_vis=True
    )
    height_command = cmd.HeightCommandCfg()
    root_vel_command = cmd.RootVelCommandCfg(
        debug_vis=True
    )
    root_ang_vel_command = cmd.RootAngVelCommandCfg()


@configclass
class MotionCfg(MotionManagerCfg):
    motion_buffer_cfg = MotionBufferCfg(
        motion = MotionBufferCfg.MotionCfg(
            motion_name = None
        ),
        regen_pkl=True
    )
    static_motion: bool = False
    obs_from_buffer: bool = False
    loc_gen: bool = True
    speed_scale: float = 1.0
    robot_type: str = None
    reset_to_pose: bool = False

@configclass
class RecordsCfg(RecorderManagerBaseCfg):
    dataset_export_dir_path = MISSING
    
    applied_torque = RecorderTermCfg(class_type=tracker_record.RecordAppliedTorque) 
    computed_torque = RecorderTermCfg(class_type=tracker_record.RecordComputedTorque)
    joint_effort_target = RecorderTermCfg(class_type=tracker_record.RecordJointEffortTarget)
    joint_acc = RecorderTermCfg(class_type=tracker_record.RecordJointAcc)
    joint_pos = RecorderTermCfg(class_type=tracker_record.RecordJointPos)
    joint_vel = RecorderTermCfg(class_type=tracker_record.RecordJointVel)    

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Command observation
        dofpos_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "dofpos_command"})
        # height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "height_command"})
        root_vel_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "root_vel_command"})
        # root_ang_vel_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "root_ang_vel_command"})

        # motion_dof_pos_whb = ObsTerm(tracker_obs.motion_dof_pos_whb)

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        # Recommend for No height scan
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            
        def set_history(self, history_len):
            # self.dofpos_command.history_length = history_len
            # self.root_vel_command.history_length = history_len
            
            self.base_ang_vel.history_length = history_len
            self.base_lin_vel.history_length = history_len
            self.projected_gravity.history_length = history_len
            self.joint_pos.history_length = history_len
            self.joint_vel.history_length = history_len
            self.actions.history_length = history_len
            
        def set_no_noise(self):
            def make_zero(tar):
                tar.noise.n_min = 0
                tar.noise.n_max = 0
            make_zero(self.base_lin_vel)
            make_zero(self.base_ang_vel)
            make_zero(self.projected_gravity)
            make_zero(self.joint_pos)
            make_zero(self.joint_vel)
            

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Task

    demo_height = RewTerm(
        func=tracker_reward.reward_tracking_demo_height, 
        weight=0.0
    )
    motion_l1_whb_dof_pos = RewTerm(
        func=tracker_reward.reward_motion_l1_whb_dof_pos_subset, 
        weight = -1.0
    )
    motion_exp_whb_dof_pos = RewTerm(
        func=tracker_reward.reward_motion_exp_whb_dof_pos_subset, 
        weight = 20.0,
        params={
            "factor": 0.3
        }
    )
    motion_base_lin_vel = RewTerm(
        func=tracker_reward.reward_motion_base_lin_vel, 
        params = {
            "vel_scale": 1.0
        },
        weight=2.0
    )
    motion_base_lin_vel_x = RewTerm(
        func=tracker_reward.reward_motion_base_lin_vel_x, 
        params = {
            "vel_scale": 1.0
        },
        weight=2.0
    )
    
    motion_base_ang_vel = RewTerm(
        func=tracker_reward.reward_motion_base_ang_vel, 
        weight=0.0
    )
    # punish_base_ang_vel = RewTerm(
    #     func=tracker_reward.punish_base_ang_vel, 
    #     weight= 1.0
    # )
    reward_alive = RewTerm(
        func = tracker_reward.reward_alive,
        weight = 1.0
    )
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle.*"),
        },
    )

    # -- penalties
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    
    def set_event_determine(self):
        self.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        
        self.reset_robot_joints.params = {
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        }
        
        self.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
        
        self.push_robot.params=={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}
        

@configclass
class CurriculumCfg:
    pass