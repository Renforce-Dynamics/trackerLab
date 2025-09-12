import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from trackerLab import TRACKERLAB_USD_DIR

BOOSTER_K1SERIAL_22DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TRACKERLAB_USD_DIR}/booster_k1_rev/usd/K1_serial.usd",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "Head": ImplicitActuatorCfg(
            joint_names_expr=[".*Head_Yaw.*", ".*Head_Pitch.*"],
            effort_limit_sim=       7.0,
            stiffness=              50.0,
            damping=                2.0,
            armature=0.01,
        ),
        "Arms": ImplicitActuatorCfg(
            joint_names_expr=[".*Shoulder_Pitch.*", ".*Shoulder_Roll.*", ".*Shoulder_Yaw.*", ".*Elbow.*"],
            effort_limit_sim=       10.0,
            stiffness=              50.0,
            damping=                2.0,
            armature=0.01,
        ),
        "Legs": ImplicitActuatorCfg(
            joint_names_expr=[".*Hip_Pitch.*", ".*Hip_Roll.*", ".*Hip_Yaw.*", ".*Knee.*", ".*Ankle_Up.*", ".*Ankle_Down.*"],
            effort_limit_sim={
                ".*Hip_Pitch.*":    45.0,
                ".*Hip_Roll.*":     30.0,
                ".*Hip_Yaw.*":      30.0,
                ".*Knee.*":         45.0,
                ".*Ankle_Up.*":     20.0,
                ".*Ankle_Down.*":   20.0,
            },
            stiffness={
                ".*Hip_Pitch.*":    200.0,
                ".*Hip_Roll.*":     200.0,
                ".*Hip_Yaw.*":      200.0,
                ".*Knee.*":         200.0,
                ".*Ankle_Up.*":     200.0,
                ".*Ankle_Down.*":   200.0,
            },
            damping={
                ".*Hip_Pitch.*":    10.0,
                ".*Hip_Roll.*":     10.0,
                ".*Hip_Yaw.*":      10.0,
                ".*Knee.*":         10.0,
                ".*Ankle_Up.*":     10.0,
                ".*Ankle_Down.*":   10.0,
            },
            armature=0.01,
        ),
    }
)

BOOSTER_K1SERIAL_22DOF_POSREV_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TRACKERLAB_USD_DIR}/booster_k1_rev/usd/K1_serial.usd",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
                ".*Shoulder_Pitch.*": 0.25,
                ".*Left_Shoulder_Roll.*": -1.4,
                ".*Right_Shoulder_Roll.*": 1.4,
                ".*Left_Elbow.*": -0.5,
                ".*Right_Elbow.*": 0.5,
                ".*Hip_Pitch.*": -0.1,
                ".*Knee.*": 0.2,
                ".*Ankle_Up.*": -0.1,
            },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "Head": ImplicitActuatorCfg(
            joint_names_expr=[".*Head_Yaw.*", ".*Head_Pitch.*"],
            effort_limit_sim=       7.0,
            stiffness=              50.0,
            damping=                2.0,
            armature=0.01,
        ),
        "Arms": ImplicitActuatorCfg(
            joint_names_expr=[".*Shoulder_Pitch.*", ".*Shoulder_Roll.*", ".*Shoulder_Yaw.*", ".*Elbow.*"],
            effort_limit_sim=       10.0,
            stiffness=              50.0,
            damping=                2.0,
            armature=0.01,
        ),
        "Legs": ImplicitActuatorCfg(
            joint_names_expr=[".*Hip_Pitch.*", ".*Hip_Roll.*", ".*Hip_Yaw.*", ".*Knee.*", ".*Ankle_Up.*", ".*Ankle_Down.*"],
            effort_limit_sim={
                ".*Hip_Pitch.*":    45.0,
                ".*Hip_Roll.*":     30.0,
                ".*Hip_Yaw.*":      30.0,
                ".*Knee.*":         45.0,
                ".*Ankle_Up.*":     20.0,
                ".*Ankle_Down.*":   20.0,
            },
            stiffness={
                ".*Hip_Pitch.*":    200.0,
                ".*Hip_Roll.*":     200.0,
                ".*Hip_Yaw.*":      200.0,
                ".*Knee.*":         200.0,
                ".*Ankle_Up.*":     200.0,
                ".*Ankle_Down.*":   200.0,
            },
            damping={
                ".*Hip_Pitch.*":    10.0,
                ".*Hip_Roll.*":     10.0,
                ".*Hip_Yaw.*":      10.0,
                ".*Knee.*":         10.0,
                ".*Ankle_Up.*":     10.0,
                ".*Ankle_Down.*":   10.0,
            },
            armature=0.01,
        ),
    }
)

BOOSTER_K1SERIAL_22DOF_POSREV_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TRACKERLAB_USD_DIR}/booster_k1_rev/usd/K1_serial.usd",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
                ".*Shoulder_Pitch.*": 0.25,
                ".*Left_Shoulder_Roll.*": -1.4,
                ".*Right_Shoulder_Roll.*": 1.4,
                ".*Left_Elbow.*": -0.5,
                ".*Right_Elbow.*": 0.5,
                ".*Hip_Pitch.*": -0.1,
                ".*Knee.*": 0.2,
                ".*Ankle_Up.*": -0.1,
            },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "Head": ImplicitActuatorCfg(
            joint_names_expr=[".*Head_Yaw.*", ".*Head_Pitch.*"],
            effort_limit_sim=       7.0,
            stiffness=              3.5,
            damping=                0.3,
            armature=0.01,
        ),
        "Arms": ImplicitActuatorCfg(
            joint_names_expr=[".*Shoulder_Pitch.*", ".*Shoulder_Roll.*", ".*Shoulder_Yaw.*", ".*Elbow.*"],
            effort_limit_sim=       10.0,
            stiffness=              5.0,
            damping=                0.5,
            armature=0.01,
        ),
        "Legs": ImplicitActuatorCfg(
            joint_names_expr=[".*Hip_Pitch.*", ".*Hip_Roll.*", ".*Hip_Yaw.*", ".*Knee.*", ".*Ankle_Up.*", ".*Ankle_Down.*"],
            effort_limit_sim={
                ".*Hip_Pitch.*":    45.0,
                ".*Hip_Roll.*":     30.0,
                ".*Hip_Yaw.*":      30.0,
                ".*Knee.*":         45.0,
                ".*Ankle_Up.*":     20.0,
                ".*Ankle_Down.*":   20.0,
            },
            stiffness={
                ".*Hip_Pitch.*":    25.0,
                ".*Hip_Roll.*":     25.0,
                ".*Hip_Yaw.*":      25.0,
                ".*Knee.*":         25.0,
                ".*Ankle_Up.*":     25.0,
                ".*Ankle_Down.*":   25.0,
            },
            damping={
                ".*Hip_Pitch.*":    2.5,
                ".*Hip_Roll.*":     2.5,
                ".*Hip_Yaw.*":      2.5,
                ".*Knee.*":         2.5,
                ".*Ankle_Up.*":     2.5,
                ".*Ankle_Down.*":   2.5,
            },
            armature=0.01,
        ),
    }
)


BOOSTER_K1SERIAL_22DOF_POSREV_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TRACKERLAB_USD_DIR}/booster_k1_rev/usd/K1_serial.usd",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.53),
        joint_pos={
                ".*Shoulder_Pitch.*": 0.25,
                ".*Left_Shoulder_Roll.*": -1.4,
                ".*Right_Shoulder_Roll.*": 1.4,
                ".*Left_Elbow.*": -0.5,
                ".*Right_Elbow.*": 0.5,
                ".*Hip_Pitch.*": -0.1,
                ".*Knee.*": 0.2,
                ".*Ankle_Up.*": -0.1,
            },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "Head": IdealPDActuatorCfg(
            joint_names_expr=[".*Head_Yaw.*", ".*Head_Pitch.*"],
            effort_limit_sim=       7.0,
            stiffness=              20.0,
            damping=                0.2,
            armature=0.01,
        ),
        "Arms": IdealPDActuatorCfg(
            joint_names_expr=[".*Shoulder_Pitch.*", ".*Shoulder_Roll.*", ".*Shoulder_Yaw.*", ".*Elbow.*"],
            effort_limit_sim=       10.0,
            stiffness=              20.0,
            damping=                0.2,
            armature=0.01,
        ),
        "Legs": IdealPDActuatorCfg(
            joint_names_expr=[".*Hip_Pitch.*", ".*Hip_Roll.*", ".*Hip_Yaw.*", ".*Knee.*", ".*Ankle_Up.*", ".*Ankle_Down.*"],
            effort_limit_sim={
                ".*Hip_Pitch.*":    60.0,
                ".*Hip_Roll.*":     25.0,
                ".*Hip_Yaw.*":      30.0,
                ".*Knee.*":         60.0,
                ".*Ankle_Up.*":     24.0,
                ".*Ankle_Down.*":   15.0,
            },
            stiffness={
                ".*Hip_Pitch.*":    100.0,
                ".*Hip_Roll.*":     100.0,
                ".*Hip_Yaw.*":      100.0,
                ".*Knee.*":         100.0,
                ".*Ankle_Up.*":     50.0,
                ".*Ankle_Down.*":   50.0,
            },
            damping={
                ".*Hip_Pitch.*":    5.0,
                ".*Hip_Roll.*":     5.0,
                ".*Hip_Yaw.*":      5.0,
                ".*Knee.*":         5.0,
                ".*Ankle_Up.*":     3.0,
                ".*Ankle_Down.*":   3.0,
            },
            armature=0.01,
        ),
    }
)