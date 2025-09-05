import math
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
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

import trackerLab.tracker_env.mdp as mdp
from trackerLab.tasks.tracking.humanoid import TrackingHumanoidEnvCfg
from trackerLab.assets.humanoids.k1 import BOOSTER_K1SERIAL_22DOF_CFG
from .motion_align_cfg import K1_MOTION_ALIGN_CFG

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
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
    robot: ArticulationCfg = BOOSTER_K1SERIAL_22DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
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
            "static_friction_range": (0.3, 2.0),
            "dynamic_friction_range": (0.3, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    scale_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!Trunk$).*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-2.5, 2.5), "y": (-2.5, 2.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-1.0, 1.0),
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
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-0.5, 1.0)}},
    )
    
@configclass
class CurriculumCfg:
    event_push_robot_levels = CurrTerm(
        func=mdp.event_push_robot_levels,
        params={"velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0), "z": (-0.5, 1.0)}}
    )


@configclass
class K1_HumanoidRewardsCfg:
    # task rewards
    motion_whb_dof_pos  = RewTerm(func=mdp.motion_whb_dof_pos_subset_exp, 
                                  params={"std": 2},
                                  weight=5.0)
    
    motion_base_lin_vel = RewTerm(func=mdp.motion_lin_vel_xy_yaw_frame_exp,
                                  params={"std": 0.5},
                                  weight=1.0)
    
    motion_base_ang_vel = RewTerm(func=mdp.motion_ang_vel_z_world_exp,
                                  params={"std": 1.0},
                                  weight=0.5)
    # base rewards
    lin_vel_z_l2        = RewTerm(func=mdp.lin_vel_z_l2,        weight=-1.0)
    ang_vel_xy_l2       = RewTerm(func=mdp.ang_vel_xy_l2,       weight=-0.05)
    dof_vel_l2          = RewTerm(func=mdp.joint_vel_l2,        weight=-0.001)
    dof_acc_l2          = RewTerm(func=mdp.joint_acc_l2,        weight=-2.5e-7)
    energy              = RewTerm(func=mdp.energy,              weight=-2e-5)
    action_rate_l2      = RewTerm(func=mdp.action_rate_l2,      weight=-0.10)
    dof_pos_limits      = RewTerm(func=mdp.joint_pos_limits,    weight=-2.0)
    alive               = RewTerm(func=mdp.is_alive,            weight=0.05)

    # contact rewards
    undesired_contacts  = RewTerm(func=mdp.undesired_contacts,  weight=-1.0,
                                  params={"sensor_cfg": SceneEntityCfg("contact_forces", 
                                            body_names=[".*shoulder.*", 
                                                        ".*elbow.*", 
                                                        ".*wrist.*",
                                                        "torso_link",
                                                        "pelvis.*",
                                                        ".*hip.*",
                                                        ".*knee.*"]),
                                          "threshold": 1.0})
    
    # gravity rewards
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    body_orientation_l2 = RewTerm(func=mdp.body_orientation_l2, weight=-5.0,
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")})

    # termination rewards
    termination_penalty = RewTerm(func=mdp.is_terminated,       weight=-200.0)

    # humanoid specific rewards
    feet_slide          = RewTerm(func=mdp.feet_slide,          weight=-1.00,
                                  params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                                          "asset_cfg":  SceneEntityCfg("robot", body_names=".*ankle_roll.*"),},)
    feet_force          = RewTerm(func=mdp.body_force,          weight=-1e-3,
                                  params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                                          "threshold": 400, "max_reward": 500})
    feet_too_near       = RewTerm(func=mdp.feet_too_near,       weight=-2.0,
                                  params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"), 
                                          "threshold": 0.30})
    feet_stumble        = RewTerm(func=mdp.feet_stumble,        weight=-2.0,
                                  params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")})
    feet_async_stable   = RewTerm(func=mdp.feet_async_stable,   weight=-1.0,
                                  params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                                          "n_dt": 3.0})
    feet_flat_ankle     = RewTerm(func=mdp.feet_flat_ankle,     weight=-1.0,
                                  params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
                                          "asset_cfg":  SceneEntityCfg("robot", body_names=".*ankle_roll.*"),},)

    # joint deviation rewards
    waists_deviation    = RewTerm(func=mdp.joint_deviation_l1,  weight=-0.01,
                                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*waist.*"])})
    arms_deviation      = RewTerm(func=mdp.joint_deviation_l1,  weight=-0.01,
                                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*shoulder.*", ".*elbow.*", ".*wrist.*"])})
    legs_deviation      = RewTerm(func=mdp.joint_deviation_l1,  weight=-0.1,
                                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip.*", ".*knee.*", ".*ankle.*"])})
    head_deviation      = RewTerm(func=mdp.joint_deviation_l1,  weight=-0.1,
                                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*head.*"])})

    def set_feet(self, names):
        self.feet_slide.params["sensor_cfg"].body_names = names
        self.feet_slide.params["asset_cfg"].body_names = names
        self.feet_force.params["sensor_cfg"].body_names = names
        self.feet_too_near.params["asset_cfg"].body_names = names
        self.feet_stumble.params["sensor_cfg"].body_names = names
        self.feet_async_stable.params["sensor_cfg"].body_names = names
        self.feet_flat_ankle.params["sensor_cfg"].body_names = names
        self.feet_flat_ankle.params["asset_cfg"].body_names = names

@configclass
class Booster_K1_TrackingEnvCfg(TrackingHumanoidEnvCfg):
    
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=5.0)
    events: EventCfg = EventCfg()
    rewards: K1_HumanoidRewardsCfg = K1_HumanoidRewardsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    def __post_init__(self):
        self.set_no_scanner()
        # self.set_plane()
        # self.adjust_scanner("base_link")
        super().__post_init__()
        self.motion.set_motion_align_cfg(K1_MOTION_ALIGN_CFG)
        self.motion.robot_type = "k1"
        
        self.observations.policy.base_lin_vel.scale = 1.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.projected_gravity.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.actions.scale = 1.0
        self.actions.joint_pos.scale = 0.5
        self.observations.policy.history_length = 5
        self.terminations.base_contact = None
        self.episode_length_s = 20.0
        
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = "^(?!.*Ankle).*$"
        self.rewards.body_orientation_l2.params["asset_cfg"].body_names = "Trunk"
        self.rewards.set_feet(".*Ankle_Roll")
        
        self.rewards.waists_deviation.weight = 0
        self.rewards.arms_deviation.weight = 0
        self.rewards.feet_async_stable.weight = 0
        self.rewards.feet_flat_ankle.weight = 0

        self.rewards.legs_deviation.params["asset_cfg"].joint_names = [".*Ankle.*", ".*Hip_Roll.*", ".*Hip_Yaw.*"]
        self.rewards.legs_deviation.weight = -1.0
        self.rewards.head_deviation.params["asset_cfg"].joint_names = [".*Head.*"]
        
        self.motion.speed_scale = 1.0
        
        self.disable_zero_weight_rewards()
        
    def disable_zero_weight_rewards(self):
        """If the weight of rewards is 0, set rewards to None"""
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and reward_attr.weight == 0:
                    setattr(self.rewards, attr, None)

@configclass
class Booster_K1_TrackingWalk(Booster_K1_TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/booster_k1/simple_walk.yaml"
        self.motion.motion_buffer_cfg.motion_type = "GMR"
        self.motion.motion_buffer_cfg.motion_lib_type = "MotionLibDofPos"

@configclass
class Booster_K1_TrackingWalk_Play(Booster_K1_TrackingWalk):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1



@configclass
class Booster_K1_TrackingWalk_Full(Booster_K1_TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/booster_k1/cmu_walk_full.yaml"
        self.motion.motion_buffer_cfg.motion_type = "GMR"
        self.motion.motion_buffer_cfg.motion_lib_type = "MotionLibDofPos"

@configclass
class Booster_K1_TrackingWalk_Full_Play(Booster_K1_TrackingWalk_Full):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1



@configclass
class Booster_K1_TrackingRun(Booster_K1_TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.motion.speed_scale = 0.5
        self.rewards.motion_base_lin_vel.params["std"] = 1.0
        self.motion.motion_buffer_cfg.motion.motion_name = "amass/booster_k1/simple_run.yaml"
        self.motion.motion_buffer_cfg.motion_type = "GMR"
        self.motion.motion_buffer_cfg.motion_lib_type = "MotionLibDofPos"


@configclass
class Booster_K1_TrackingRun_Play(Booster_K1_TrackingRun):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 1



        
