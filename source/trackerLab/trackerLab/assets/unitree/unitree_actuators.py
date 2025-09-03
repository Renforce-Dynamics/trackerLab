from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class UnitreeActuator(IdealPDActuator):
    """Unitree actuator class that implements a torque-speed curve for the actuators.

    The torque-speed curve is defined as follows:

            Torque, N·m
                ^
    Y2──────────|
                |──────────────Y1
                |              │\
                |              │ \
                |              │  \
                |              |   \
    ------------+--------------|------> velocity: rad/s
                              X1   X2

    Y1: Peak Torque Test (Torque and Speed in the Same Direction)
    Y2: Peak Torque Test (Torque and Speed in the Opposite Direction)
    X1: Maximum Speed at Full Torque (T-N Curve Knee Point)
    X2: No-Load Speed Test
    """

    cfg: UnitreeActuatorCfg

    def __init__(self, cfg: UnitreeActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        assert cfg.X1 < cfg.X2, "X1 must be less than X2"
        assert cfg.Y1 <= cfg.Y2, "Y1 must be less than or equal to Y2"

        self._joint_vel = torch.zeros_like(self.computed_effort)
        self._effort_y1 = torch.tensor(cfg.Y1, dtype=self.computed_effort.dtype, device=self.computed_effort.device)
        self._effort_y2 = torch.tensor(cfg.Y2, dtype=self.computed_effort.dtype, device=self.computed_effort.device)
        self._velocity_x1 = torch.tensor(cfg.X1, dtype=self.computed_effort.dtype, device=self.computed_effort.device)
        self._velocity_x2 = torch.tensor(cfg.X2, dtype=self.computed_effort.dtype, device=self.computed_effort.device)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # calculate the desired joint torques
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # check if the effort is the same direction as the joint velocity
        same_direction = (self._joint_vel * effort) > 0
        max_effort = torch.where(same_direction, self._effort_y1, self._effort_y2)
        # check if the joint velocity is less than the max speed at full torque
        max_effort = torch.where(
            self._joint_vel.abs() < self._velocity_x1, max_effort, self._compute_effort_limit(max_effort)
        )
        return torch.clip(effort, -max_effort, max_effort)

    def _compute_effort_limit(self, max_effort):
        k = -max_effort / (self._velocity_x2 - self._velocity_x1)
        limit = k * (self._joint_vel.abs() - self._velocity_x1) + max_effort
        return limit.clip(min=0.0)


@configclass
class UnitreeActuatorCfg(IdealPDActuatorCfg):
    """
    Configuration for Unitree actuators.
    """

    class_type: type = UnitreeActuator

    X1: float = MISSING
    """Maximum Speed at Full Torque(T-N Curve Knee Point) Unit: rad/s"""

    X2: float = MISSING
    """No-Load Speed Test Unit: rad/s"""

    Y1: float = MISSING
    """Peak Torque Test(Torque and Speed in the Same Direction) Unit: N*m"""

    Y2: float = MISSING
    """Peak Torque Test(Torque and Speed in the Opposite Direction) Unit: N*m"""


@configclass
class UnitreeActuatorCfg_M107_15(UnitreeActuatorCfg):
    X1 = 14.0
    X2 = 25.6
    Y1 = 150.0
    Y2 = 102.8


@configclass
class UnitreeActuatorCfg_M107_24(UnitreeActuatorCfg):
    X1 = 8.8
    X2 = 16
    Y1 = 240
    Y2 = 292.5


@configclass
class UnitreeActuatorCfg_Go2HV(UnitreeActuatorCfg):
    X1 = 13.5
    X2 = 30
    Y1 = 20.2
    Y2 = 23.4


@configclass
class UnitreeActuatorCfg_N7520_14p3(UnitreeActuatorCfg):
    # Decimal point cannot be used as variable name, use `p` instead
    X1 = 22.63
    X2 = 35.52
    Y1 = 71
    Y2 = 83.3


@configclass
class UnitreeActuatorCfg_N7520_22p5(UnitreeActuatorCfg):
    # Decimal point cannot be used as variable name, use `p` instead
    X1 = 14.5
    X2 = 22.7
    Y1 = 111.0
    Y2 = 131.0


@configclass
class UnitreeActuatorCfg_N5010_16(UnitreeActuatorCfg):
    X1 = 27.0
    X2 = 41.5
    Y1 = 9.5
    Y2 = 17.0


@configclass
class UnitreeActuatorCfg_N5020_16(UnitreeActuatorCfg):
    X1 = 30.86
    X2 = 40.13
    Y1 = 24.8
    Y2 = 31.9


@configclass
class UnitreeActuatorCfg_W4010_25(UnitreeActuatorCfg):
    X1 = 15.3
    X2 = 24.76
    Y1 = 4.8
    Y2 = 8.6
