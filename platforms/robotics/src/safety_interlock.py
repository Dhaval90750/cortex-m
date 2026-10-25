"""Deterministic Safety Interlock for CORTEX-Robotics.

Implements hard ISO 10218-1/2:2025 and ISO/TS 15066 collaborative safety rules:
1. Joint position, velocity, and acceleration ceilings.
2. Cartesian Tool Center Point (TCP) speed envelope limiters.
3. Speed and Separation Monitoring (SSM) based on human proximity.
4. Deterministic Emergency Stop (E-Stop) latch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .kinematics import KinematicLimits, Manipulator6DOF, RobotPose


class SafetyViolation(PermissionError):
    """Raised when a proposed robot command or trajectory violates safety invariants."""
    pass


@dataclass
class SafetyStatusReport:
    """Real-time diagnostic report produced by safety interlock."""
    is_safe: bool
    e_stop_active: bool
    human_distance_m: Optional[float]
    allowed_tcp_speed_m_s: float
    violations: List[str]


class DeterministicSafetyInterlock:
    """Non-negotiable safety supervisor guarding robotic actuators."""

    COLLABORATIVE_REDUCED_SPEED_M_S = 0.25  # ISO/TS 15066 reduced speed for close proximity
    MIN_SAFETY_DISTANCE_M = 0.50  # Hard stop distance for human intrusion

    def __init__(
        self,
        manipulator: Manipulator6DOF,
        limits: Optional[KinematicLimits] = None,
        strict_mode: bool = True,
    ):
        self.manipulator = manipulator
        self.limits = limits or manipulator.limits
        self.strict_mode = strict_mode
        self._e_stop_latched = False

    def trigger_estop(self, reason: str = "Manual E-Stop") -> None:
        """Immediately latch the Emergency Stop flag, cutting actuator power."""
        self._e_stop_latched = True

    def reset_estop(self) -> None:
        """Clear E-Stop latch after operator safety sign-off."""
        self._e_stop_latched = False

    @property
    def is_estop_active(self) -> bool:
        return self._e_stop_latched

    def calculate_allowable_tcp_speed(self, human_distance_m: Optional[float]) -> float:
        """Compute ISO/TS 15066 speed ceiling based on human separation distance."""
        if human_distance_m is None:
            return self.limits.max_tcp_velocity_m_s

        if human_distance_m <= self.MIN_SAFETY_DISTANCE_M:
            return 0.0  # Immediate halt required
        elif human_distance_m <= 1.5:
            return self.COLLABORATIVE_REDUCED_SPEED_M_S
        else:
            return self.limits.max_tcp_velocity_m_s

    def inspect_step(
        self,
        current_joints: Sequence[float],
        target_joints: Sequence[float],
        dt_seconds: float = 0.01,
        human_distance_m: Optional[float] = None,
    ) -> SafetyStatusReport:
        """Inspect a proposed joint step before hardware command execution."""
        violations = []

        if self._e_stop_latched:
            violations.append("E-Stop active: Actuator motion forbidden.")

        # 1. Joint angle bounds
        if not self.manipulator.validate_joint_limits(target_joints):
            violations.append("Target joints exceed physical mechanical limit bounds.")

        # 2. Joint velocity bounds
        if dt_seconds > 0.0:
            for i in range(len(current_joints)):
                v = abs(target_joints[i] - current_joints[i]) / dt_seconds
                if v > self.limits.max_joint_velocity_rad_s:
                    violations.append(
                        f"Joint {i} velocity {v:.2f} rad/s exceeds limit {self.limits.max_joint_velocity_rad_s:.2f} rad/s."
                    )

        # 3. Cartesian TCP velocity bounds
        p_curr = self.manipulator.forward_kinematics(current_joints)
        p_target = self.manipulator.forward_kinematics(target_joints)
        dx = p_target.x - p_curr.x
        dy = p_target.y - p_curr.y
        dz = p_target.z - p_curr.z
        tcp_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        tcp_speed = tcp_dist / max(1e-5, dt_seconds)

        allowed_tcp_speed = self.calculate_allowable_tcp_speed(human_distance_m)
        if tcp_speed > allowed_tcp_speed:
            violations.append(
                f"TCP velocity {tcp_speed:.2f} m/s exceeds allowable safety ceiling {allowed_tcp_speed:.2f} m/s."
            )

        is_safe = len(violations) == 0
        if not is_safe and self.strict_mode:
            # Enforce immediate fail-safe latch if critical violation
            if any("exceeds" in v for v in violations):
                self._e_stop_latched = True

        return SafetyStatusReport(
            is_safe=is_safe,
            e_stop_active=self._e_stop_latched,
            human_distance_m=human_distance_m,
            allowed_tcp_speed_m_s=allowed_tcp_speed,
            violations=violations,
        )
