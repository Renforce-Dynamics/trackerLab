import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from trackerLab import TRACKERLAB_USD_DIR

BOOSTER_K1SERIAL_22DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{TRACKERLAB_USD_DIR}/booster_k1/usd/K1_serial_rev.usd",
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
            joint_names_expr=["AAHead_yaw", "Head_pitch"],
            effort_limit_sim=7.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
        ),
        "Arms": ImplicitActuatorCfg(
            joint_names_expr=[".*Shoulder_Pitch", ".*Shoulder_Roll", ".*Elbow_Pitch", ".*Elbow_Yaw"],
            effort_limit_sim=10.0,
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
        ),
        "Legs": ImplicitActuatorCfg(
            joint_names_expr=[".*Hip_Pitch", ".*Hip_Roll", ".*Hip_Yaw", ".*Knee_Pitch", ".*Ankle_Pitch", ".*Ankle_Roll"],
            effort_limit_sim={
                ".*Hip_Pitch": 45.0,
                ".*Hip_Roll": 30.0,
                ".*Hip_Yaw": 30.0,
                ".*Knee_Pitch": 45.0,
                ".*Ankle_Pitch": 20.0, 
                ".*Ankle_Roll": 20.0,
            },
            stiffness={
                ".*Hip_Pitch": 200.0,
                ".*Hip_Roll": 200.0,
                ".*Hip_Yaw": 200.0,
                ".*Knee_Pitch": 200.0,
                ".*Ankle_Pitch": 200.0, 
                ".*Ankle_Roll": 200.0,
            },
            damping={
                ".*Hip_Pitch": 10.0,
                ".*Hip_Roll": 10.0,
                ".*Hip_Yaw": 10.0,
                ".*Knee_Pitch": 10.0,
                ".*Ankle_Pitch": 10.0,
                ".*Ankle_Roll": 10.0,
            },
            armature=0.01,
        ),
    }
)