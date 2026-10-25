"""6-DOF Manipulator Kinematic Engine for CORTEX-Robotics.

Implements forward kinematics, joint limit validation, and TCP pose calculation
for standard 6-axis industrial articulating arms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RobotPose:
    """Cartesian Tool Center Point (TCP) pose in 3D workspace."""
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @property
    def position_xyz(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class KinematicLimits:
    """Joint physical boundaries and maximum velocity/acceleration constraints."""
    joint_min_rad: Tuple[float, ...] = (-math.pi, -math.pi, -math.pi, -math.pi, -math.pi, -math.pi)
    joint_max_rad: Tuple[float, ...] = (math.pi, math.pi, math.pi, math.pi, math.pi, math.pi)
    max_joint_velocity_rad_s: float = 3.14  # ~180 deg/s
    max_joint_accel_rad_s2: float = 10.0
    max_tcp_velocity_m_s: float = 1.5  # ISO 10218-1 maximum speed
    workspace_max_radius_m: float = 1.2


class Manipulator6DOF:
    """Standard 6-DOF industrial articulating robot arm kinematics."""

    # Nominal link lengths in meters (base to wrist)
    LINK_LENGTHS = (0.20, 0.40, 0.35, 0.15, 0.10, 0.05)

    def __init__(self, limits: Optional[KinematicLimits] = None):
        self.limits = limits or KinematicLimits()

    def forward_kinematics(self, joint_angles_rad: Sequence[float]) -> RobotPose:
        """Compute end-effector TCP pose from 6 joint angles (radians)."""
        if len(joint_angles_rad) != 6:
            raise ValueError(f"Manipulator requires exactly 6 joint angles, got {len(joint_angles_rad)}")

        q = [float(angle) for angle in joint_angles_rad]
        l1, l2, l3, l4, l5, l6 = self.LINK_LENGTHS

        # Geometric planar/spherical forward kinematics approximation for 6-axis arm
        # Base rotation q[0]
        c0, s0 = math.cos(q[0]), math.sin(q[0])
        # Arm elevation q[1] and elbow q[2]
        c1, s1 = math.cos(q[1]), math.sin(q[1])
        c12 = math.cos(q[1] + q[2])
        s12 = math.sin(q[1] + q[2])
        c123 = math.cos(q[1] + q[2] + q[3])
        s123 = math.sin(q[1] + q[2] + q[3])

        # Planar reach r in XY plane
        r = l2 * c1 + l3 * c12 + l4 * c123 + (l5 + l6) * math.cos(q[1] + q[2] + q[3] + q[4])
        x = r * c0
        y = r * s0
        z = l1 + l2 * s1 + l3 * s12 + l4 * s123 + (l5 + l6) * math.sin(q[1] + q[2] + q[3] + q[4])

        return RobotPose(
            x=round(x, 4),
            y=round(y, 4),
            z=round(z, 4),
            roll=round(q[3], 4),
            pitch=round(q[4], 4),
            yaw=round(q[5], 4),
        )

    def validate_joint_limits(self, joint_angles_rad: Sequence[float]) -> bool:
        """Verify that all joint angles lie within certified mechanical limits."""
        if len(joint_angles_rad) != 6:
            return False
        for i, angle in enumerate(joint_angles_rad):
            if angle < self.limits.joint_min_rad[i] or angle > self.limits.joint_max_rad[i]:
                return False
        return True
