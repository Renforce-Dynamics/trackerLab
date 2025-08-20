import numpy as np
from sim2simlib.model.config import Actions, Motor_Config

class DC_Motor():
    def __init__(self, cfg:Motor_Config):
        self.cfg = cfg
        self._effort_limit = self.cfg.effort_limit
        self._saturation_effort = self.cfg.saturation_effort
        self._velocity_limit = self.cfg.velocity_limit
        self._stiffness = self.cfg.stiffness
        self._damping = self.cfg.damping
        self._friction = self.cfg.friction

    def compute(self, joint_pos:np.ndarray, joint_vel:np.ndarray, action: Actions) -> np.ndarray:
        self._joint_vel = joint_vel
        error_pos = action.joint_pos - joint_pos
        error_vel = action.joint_vel - joint_vel
        computed_effort = self._stiffness * error_pos + self._damping * error_vel + action.joint_efforts
        applied_effort = self._clip_effort(computed_effort)
        return applied_effort
    
     
    def _clip_effort(self, effort: np.ndarray) -> np.ndarray:
        # compute torque limits
        # -- max limit
        max_effort = self._saturation_effort * (1.0 - self._joint_vel / self._velocity_limit)
        max_effort = np.clip(max_effort, a_min=0, a_max=self._effort_limit)
        # -- min limit
        min_effort = self._saturation_effort * (-1.0 - self._joint_vel / self._velocity_limit)
        min_effort = np.clip(min_effort, a_min=-self._effort_limit, a_max=0)

        # clip the torques based on the motor limits
        return np.clip(effort, a_min=min_effort, a_max=max_effort)

"""

"""
class PID_Motor():
    def __init__(self, cfg:Motor_Config):
        self.cfg = cfg
        self._effort_limit = self.cfg.effort_limit
        self._saturation_effort = self.cfg.saturation_effort
        self._velocity_limit = self.cfg.velocity_limit
        self._stiffness = self.cfg.stiffness
        self._damping = self.cfg.damping
        self._friction = self.cfg.friction


    def compute(self, joint_pos:np.ndarray, joint_vel:np.ndarray, action: Actions) -> np.ndarray:
        self._joint_vel = joint_vel
        error_pos = action.joint_pos - joint_pos
        error_vel = action.joint_vel - joint_vel
        computed_effort = self._stiffness * error_pos + self._damping * error_vel + action.joint_efforts
        applied_effort = self._clip_effort(computed_effort)
        return applied_effort
     
    def _clip_effort(self, effort: np.ndarray) -> np.ndarray:
        # clip the torques based on the motor limits
        return np.clip(effort, a_min=-self._effort_limit, a_max=self._effort_limit)
