from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots.smpl import SMPL_HUMANOID
from trackerTask.beyondMimic.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class HumanoidFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SMPL_HUMANOID.replace(
            actuators={
                "body": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    velocity_limit=100.0,
                    stiffness=None,
                    damping=None,
                ),
            },
        )

        self.commands.motion.anchor_body_name = "Torso"
        self.commands.motion.joint_names = [
            "L_Hip_x",
            "L_Hip_y",
            "L_Hip_z",
            "L_Knee_x",
            "L_Knee_y",
            "L_Knee_z",
            "L_Ankle_x",
            "L_Ankle_y",
            "L_Ankle_z",
            "L_Toe_x",
            "L_Toe_y",
            "L_Toe_z",
            "R_Hip_x",
            "R_Hip_y",
            "R_Hip_z",
            "R_Knee_x",
            "R_Knee_y",
            "R_Knee_z",
            "R_Ankle_x",
            "R_Ankle_y",
            "R_Ankle_z",
            "R_Toe_x",
            "R_Toe_y",
            "R_Toe_z",
            "Torso_x",
            "Torso_y",
            "Torso_z",
            "Spine_x",
            "Spine_y",
            "Spine_z",
            "Chest_x",
            "Chest_y",
            "Chest_z",
            "Neck_x",
            "Neck_y",
            "Neck_z",
            "Head_x",
            "Head_y",
            "Head_z",
            "L_Thorax_x",
            "L_Thorax_y",
            "L_Thorax_z",
            "L_Shoulder_x",
            "L_Shoulder_y",
            "L_Shoulder_z",
            "L_Elbow_x",
            "L_Elbow_y",
            "L_Elbow_z",
            "L_Wrist_x",
            "L_Wrist_y",
            "L_Wrist_z",
            "L_Hand_x",
            "L_Hand_y",
            "L_Hand_z",
            "R_Thorax_x",
            "R_Thorax_y",
            "R_Thorax_z",
            "R_Shoulder_x",
            "R_Shoulder_y",
            "R_Shoulder_z",
            "R_Elbow_x",
            "R_Elbow_y",
            "R_Elbow_z",
            "R_Wrist_x",
            "R_Wrist_y",
            "R_Wrist_z",
            "R_Hand_x",
            "R_Hand_y",
            "R_Hand_z",
        ]

        self.commands.motion.body_names = [
            "Pelvis",
            "L_Knee",
            "L_Ankle",
            "L_Toe",
            "R_Knee",
            "R_Ankle",
            "R_Toe",
            "Torso",
            "Spine",
            "Chest",
            "Neck",
            "Head",
            "L_Shoulder",
            "L_Elbow",
            "L_Hand",
            "R_Elbow",
            "R_Hand",
        ]


@configclass
class HumanoidFlatWalkEnvCfg(HumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = f"{ASSET_DIR}/smpl/motions/walk.npz"


@configclass
class HumanoidFlatWalkBackEnvCfg(HumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = f"{ASSET_DIR}/smpl/motions/walk_back.npz"


@configclass
class HumanoidFlatWalkBoxEnvCfg(HumanoidFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = f"{ASSET_DIR}/smpl/motions/walk_box.npz"
