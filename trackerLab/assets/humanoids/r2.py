import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from trackerLab import TRACKERLAB_USD_DIR, TRACKERLAB_ASSETS_DIR

JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_arm_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_arm_pitch_joint", 
]

INIT_POS={
    # left leg
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    # right leg
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    # arms
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_arm_pitch_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_arm_pitch_joint": 0.0
}

STIFFNESS_REAL={
    # left leg
    "left_hip_pitch_joint": 100.0,
    "left_hip_roll_joint": 100.0,
    "left_hip_yaw_joint": 100.0,
    "left_knee_joint": 150.0,
    "left_ankle_pitch_joint": 30.0,
    "left_ankle_roll_joint": 30.0,
    # right leg
    "right_hip_pitch_joint": 100.0,
    "right_hip_roll_joint": 100.0,
    "right_hip_yaw_joint": 100.0,
    "right_knee_joint": 150.0,
    "right_ankle_pitch_joint": 30.0,
    "right_ankle_roll_joint": 30.0,
    # arms
    "left_shoulder_pitch_joint": 300.0,
    "left_shoulder_roll_joint": 300.0,
    "left_arm_pitch_joint": 300.0,
    "right_shoulder_pitch_joint": 300.0,
    "right_shoulder_roll_joint": 300.0,
    "right_arm_pitch_joint": 300.0
}

DAMPING_REAL={
    # left leg
    "left_hip_pitch_joint": 5.0,
    "left_hip_roll_joint": 5.0,
    "left_hip_yaw_joint": 5.0,
    "left_knee_joint": 7.0,
    "left_ankle_pitch_joint": 3.0,
    "left_ankle_roll_joint": 3.0,
    # right leg
    "right_hip_pitch_joint": 5.0,
    "right_hip_roll_joint": 5.0,
    "right_hip_yaw_joint": 5.0,
    "right_knee_joint": 7.0,
    "right_ankle_pitch_joint": 3.0,
    "right_ankle_roll_joint": 3.0,
    # arms
    "left_shoulder_pitch_joint": 3.0,
    "left_shoulder_roll_joint": 3.0,
    "left_arm_pitch_joint": 3.0,
    "right_shoulder_pitch_joint": 3.0,
    "right_shoulder_roll_joint": 3.0,
    "right_arm_pitch_joint": 3.0
}

EFFORT_REAL={
    # left leg
    "left_hip_pitch_joint": 130.0,
    "left_hip_roll_joint": 130.0,
    "left_hip_yaw_joint": 90.0,
    "left_knee_joint": 150.0,
    "left_ankle_pitch_joint": 75.0,
    "left_ankle_roll_joint": 75.0,
    # right leg
    "right_hip_pitch_joint": 130.0,
    "right_hip_roll_joint": 130.0,
    "right_hip_yaw_joint": 90.0,
    "right_knee_joint": 150.0,
    "right_ankle_pitch_joint": 75.0,
    "right_ankle_roll_joint": 75.0,
    # arms
    "left_shoulder_pitch_joint": 36.0,
    "left_shoulder_roll_joint": 36.0,
    "left_arm_pitch_joint": 36.0,
    "right_shoulder_pitch_joint": 36.0,
    "right_shoulder_roll_joint": 36.0,
    "right_arm_pitch_joint": 36.0
}

##
# Configuration
##

R2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TRACKERLAB_USD_DIR}/r2/r2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False, 
            solver_position_iteration_count=4, 
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.665),
        joint_pos=INIT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "actuators": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=STIFFNESS_REAL,
            damping=DAMPING_REAL,
            effort_limit=EFFORT_REAL,
        ),
    },
)

"""Configuration for the R2 Humanoid Robot."""

