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

    def _dh_transform(self, theta: float, d: float, a: float, alpha: float) -> np.ndarray:
        """Standard Denavit-Hartenberg 4x4 homogeneous transformation matrix."""
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array([
            [ct, -st * ca,  st * sa, a * ct],
            [st,  ct * ca, -ct * sa, a * st],
            [0.0,      sa,       ca,      d],
            [0.0,     0.0,      0.0,    1.0],
        ], dtype=np.float64)

    def forward_kinematics(self, joint_angles_rad: Sequence[float]) -> RobotPose:
        """Compute end-effector TCP pose from 6 joint angles using DH parameters."""
        if len(joint_angles_rad) != 6:
            raise ValueError(f"Manipulator requires exactly 6 joint angles, got {len(joint_angles_rad)}")

        q = [float(angle) for angle in joint_angles_rad]
        l1, l2, l3, l4, l5, l6 = self.LINK_LENGTHS

        # Standard 6-axis articulating arm DH parameters (theta, d, a, alpha)
        dh_params = [
            (q[0], l1, 0.0, math.pi / 2),
            (q[1], 0.0, l2, 0.0),
            (q[2], 0.0, l3, math.pi / 2),
            (q[3], l4, 0.0, -math.pi / 2),
            (q[4], 0.0, l5, math.pi / 2),
            (q[5], l6, 0.0, 0.0),
        ]

        T = np.eye(4, dtype=np.float64)
        for theta, d, a, alpha in dh_params:
            T = np.dot(T, self._dh_transform(theta, d, a, alpha))

        # Position
        x = float(T[0, 3])
        y = float(T[1, 3])
        z = float(T[2, 3])

        # Euler angles (roll, pitch, yaw) from rotation matrix
        r11, r21, r31 = T[0, 0], T[1, 0], T[2, 0]
        r32, r33 = T[2, 1], T[2, 2]
        pitch = math.atan2(-r31, math.sqrt(r11**2 + r21**2))
        yaw = math.atan2(r21, r11)
        roll = math.atan2(r32, r33)

        return RobotPose(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )

    def compute_jacobian(self, joint_angles_rad: Sequence[float], delta: float = 1e-4) -> np.ndarray:
        """Compute 3x6 translational Jacobian matrix via finite difference."""
        q = np.array(joint_angles_rad, dtype=np.float64)
        base_pose = self.forward_kinematics(q)
        base_pos = np.array([base_pose.x, base_pose.y, base_pose.z])
        J = np.zeros((3, 6), dtype=np.float64)

        for i in range(6):
            q_pert = q.copy()
            q_pert[i] += delta
            pert_pose = self.forward_kinematics(q_pert)
            pert_pos = np.array([pert_pose.x, pert_pose.y, pert_pose.z])
            J[:, i] = (pert_pos - base_pos) / delta

        return J

    def manipulability_index(self, joint_angles_rad: Sequence[float]) -> float:
        """Calculate Yoshikawa Manipulability Index w = sqrt(det(J * J^T))."""
        J = self.compute_jacobian(joint_angles_rad)
        JJT = np.dot(J, J.T)
        det = np.linalg.det(JJT)
        return float(np.sqrt(max(0.0, det)))

    def inverse_kinematics(
        self,
        target_xyz: Tuple[float, float, float],
        initial_guess: Optional[Sequence[float]] = None,
        max_iterations: int = 100,
        tolerance_m: float = 1e-3,
        damping_lambda: float = 0.05,
    ) -> Tuple[bool, List[float], float]:
        """Solve inverse kinematics via Damped Least Squares (Levenberg-Marquardt Jacobian pseudo-inverse).
        
        Returns:
            Tuple of (converged: bool, joint_angles_rad: List[float], final_error_m: float)
        """
        target = np.array(target_xyz, dtype=np.float64)
        q = np.array(initial_guess if initial_guess is not None else [0.0] * 6, dtype=np.float64)

        for iteration in range(max_iterations):
            current_pose = self.forward_kinematics(q)
            current_pos = np.array([current_pose.x, current_pose.y, current_pose.z])
            error = target - current_pos
            error_norm = float(np.linalg.norm(error))

            if error_norm <= tolerance_m:
                return True, [round(float(a), 4) for a in q], round(error_norm, 5)

            J = self.compute_jacobian(q)
            # Damped Least Squares: delta_q = J^T (J J^T + lambda^2 I)^(-1) * error
            JJT = np.dot(J, J.T)
            damped_inv = np.linalg.inv(JJT + (damping_lambda ** 2) * np.eye(3))
            delta_q = np.dot(np.dot(J.T, damped_inv), error)

            # Limit step size to avoid oscillation
            max_step = 0.25
            step_norm = np.linalg.norm(delta_q)
            if step_norm > max_step:
                delta_q = (delta_q / step_norm) * max_step

            q = q + delta_q

            # Clamp to physical joint limits
            for j in range(6):
                q[j] = np.clip(q[j], self.limits.joint_min_rad[j], self.limits.joint_max_rad[j])

        final_pose = self.forward_kinematics(q)
        final_error = float(np.linalg.norm(target - np.array([final_pose.x, final_pose.y, final_pose.z])))
        converged = final_error <= tolerance_m
        return converged, [round(float(a), 4) for a in q], round(final_error, 5)

    def validate_joint_limits(self, joint_angles_rad: Sequence[float]) -> bool:
        """Verify that all joint angles lie within certified mechanical limits."""
        if len(joint_angles_rad) != 6:
            return False
        for i, angle in enumerate(joint_angles_rad):
            if angle < self.limits.joint_min_rad[i] or angle > self.limits.joint_max_rad[i]:
                return False
        return True
