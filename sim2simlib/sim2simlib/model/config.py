
from trackerLab.managers.motion_manager import MotionManagerCfg
from dataclasses import dataclass
import numpy as np

@dataclass
class Actions_Config:
    scale: float
    action_clip: tuple[float, float]

@dataclass
class Actions():
    joint_pos: np.ndarray = None
    joint_vel: np.ndarray = None
    joint_efforts: np.ndarray = None
    
@dataclass
class Observations_Config:
    base_observations_terms: list[str]
    scale: dict[str, float]
    motion_observations_terms: list[str] = None
    
@dataclass
class Motor_Config(): 
    motor_type: type = None
    joint_names: list[str] = None
    effort_limit: float | dict[str, float] = 0.0
    saturation_effort: float | dict[str, float] = 0.0
    velocity_limit: float | dict[str, float] = 0.0
    stiffness: float | dict[str, float] = 0.0
    damping: float | dict[str, float] = 0.0
    friction: float | dict[str, float] = 0.0
    

@dataclass
class Sim2Sim_Config:
    robot_name: str
    simulation_dt: float
    control_decimation: int
    xml_path: str
    policy_path: str
    policy_joint_names: list[str]
    default_pos: np.ndarray
    default_angles: np.ndarray
    
    observation_cfg: Observations_Config
    action_cfg: Actions_Config
    motor_cfg: Motor_Config
    
    motion_cfg: MotionManagerCfg = None