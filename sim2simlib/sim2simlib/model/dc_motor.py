import numpy as np
import re
from sim2simlib.model.config import Actions, Motor_Config

class DC_Motor():
    _effort_limit: float | np.ndarray
    _saturation_effort: float | np.ndarray
    _velocity_limit: float | np.ndarray
    _stiffness: float | np.ndarray
    _damping: float | np.ndarray
    _friction: float | np.ndarray

    def __init__(self, cfg:Motor_Config):
        self.cfg = cfg
        self.parse_cfg()
        
    def parse_cfg(self):
        """Parse motor configuration and convert to numpy arrays based on joint names."""        
        # Parse each parameter
        self._effort_limit = self._parse_parameter(self.cfg.effort_limit)
        self._saturation_effort = self._parse_parameter(self.cfg.saturation_effort)
        self._velocity_limit = self._parse_parameter(self.cfg.velocity_limit)
        self._stiffness = self._parse_parameter(self.cfg.stiffness)
        self._damping = self._parse_parameter(self.cfg.damping)
        self._friction = self._parse_parameter(self.cfg.friction)
        
        print(f"[INFO] Motor parameters for {self.cfg.joint_names}:")
        print(f"  Effort limit: {self._effort_limit}")
        print(f"  Saturation effort: {self._saturation_effort}")
        print(f"  Velocity limit: {self._velocity_limit}")
        print(f"  Stiffness: {self._stiffness}")
        print(f"  Damping: {self._damping}")
        print(f"  Friction: {self._friction}")

    def _parse_parameter(self, param: float | dict[str, float]) -> np.ndarray:
        """Parse a parameter that can be either float or dict with regex patterns."""
        num_joints = len(self.cfg.joint_names)
        result = np.zeros(num_joints)
        
        if isinstance(param, (float, int)):
            # If it's a single value, apply to all joints
            result.fill(param)
        elif isinstance(param, dict):
            # If it's a dict with regex patterns, match each joint name
            for i, joint_name in enumerate(self.cfg.joint_names):
                matched = False
                for pattern, value in param.items():
                    if re.match(pattern, joint_name):
                        result[i] = value
                        matched = True
                        break
                if not matched:
                    # If no pattern matches, use a default value (1e-7)
                    result[i] = 1e-7
        else:
            raise ValueError(f"Parameter must be float or dict, got {type(param)}")
            
        return result

    def compute(self, joint_pos:np.ndarray, joint_vel:np.ndarray, action: Actions) -> np.ndarray:
        """Compute motor torques based on current joint state and desired actions."""
        self._joint_vel = joint_vel
        
        # Calculate position and velocity errors
        error_pos = action.joint_pos - joint_pos
        error_vel = action.joint_vel - joint_vel
        
        # Compute desired effort using PD control with feedforward
        computed_effort = self._stiffness * error_pos + self._damping * error_vel + action.joint_efforts
        
        # Apply motor limits
        applied_effort = self._clip_effort(computed_effort)
        return applied_effort
    
     
    def _clip_effort(self, effort: np.ndarray) -> np.ndarray:
        """Clip motor efforts based on velocity-dependent torque limits."""
        # Ensure all parameters are numpy arrays for element-wise operations
        joint_vel = np.asarray(self._joint_vel)
        effort_limit = np.asarray(self._effort_limit)
        saturation_effort = np.asarray(self._saturation_effort)
        velocity_limit = np.asarray(self._velocity_limit)
        
        # Avoid division by zero
        velocity_ratio = np.divide(joint_vel, velocity_limit, 
                                 out=np.zeros_like(joint_vel), 
                                 where=velocity_limit!=0)
        
        # Compute torque limits based on motor characteristics
        # -- max limit (positive direction)
        max_effort = saturation_effort * (1.0 - velocity_ratio)
        max_effort = np.clip(max_effort, a_min=0, a_max=effort_limit)
        
        # -- min limit (negative direction)  
        min_effort = saturation_effort * (-1.0 - velocity_ratio)
        min_effort = np.clip(min_effort, a_min=-effort_limit, a_max=0)

        # Clip the torques based on the computed motor limits
        return np.clip(effort, a_min=min_effort, a_max=max_effort)

"""

"""
class PID_Motor():
    _effort_limit: np.ndarray
    _stiffness: np.ndarray
    _damping: np.ndarray

    def __init__(self, cfg:Motor_Config):
        self.cfg = cfg
        self.parse_cfg()
        
    def parse_cfg(self):
        """Parse motor configuration and convert to numpy arrays based on joint names."""        
        # Parse only the parameters needed for PID control
        self._effort_limit = self._parse_parameter(self.cfg.effort_limit)
        self._stiffness = self._parse_parameter(self.cfg.stiffness)
        self._damping = self._parse_parameter(self.cfg.damping)

        print(f"[INFO] Motor parameters for {self.cfg.joint_names}:")
        print(f"  Effort limit: {self._effort_limit}")
        print(f"  Stiffness: {self._stiffness}")
        print(f"  Damping: {self._damping}")

    def _parse_parameter(self, param: float | dict[str, float]) -> np.ndarray:
        """Parse a parameter that can be either float or dict with regex patterns."""
        num_joints = len(self.cfg.joint_names)
        result = np.zeros(num_joints)
        
        if isinstance(param, (float, int)):
            # If it's a single value, apply to all joints
            result.fill(param)
        elif isinstance(param, dict):
            # If it's a dict with regex patterns, match each joint name
            for i, joint_name in enumerate(self.cfg.joint_names):
                matched = False
                for pattern, value in param.items():
                    if re.match(pattern, joint_name):
                        result[i] = value
                        matched = True
                        break
                if not matched:
                    # If no pattern matches, use a default value (1e-7)
                    result[i] = 1e-7
        else:
            raise ValueError(f"Parameter must be float or dict, got {type(param)}")
            
        return result

    def compute(self, joint_pos:np.ndarray, joint_vel:np.ndarray, action: Actions) -> np.ndarray:
        """Compute motor torques based on current joint state and desired actions using simple PID control."""
        # Calculate position and velocity errors
        error_pos = action.joint_pos - joint_pos
        error_vel = action.joint_vel - joint_vel
        
        # Compute desired effort using PD control with feedforward
        computed_effort = self._stiffness * error_pos + self._damping * error_vel + action.joint_efforts
        
        # Apply simple torque limits
        applied_effort = self._clip_effort(computed_effort)
        return applied_effort
     
    def _clip_effort(self, effort: np.ndarray) -> np.ndarray:
        """Clip motor efforts based on simple torque limits."""
        # Ensure effort_limit is numpy array for element-wise operations
        effort_limit = np.asarray(self._effort_limit)
        
        # Clip the torques based on the motor limits
        return np.clip(effort, a_min=-effort_limit, a_max=effort_limit)
