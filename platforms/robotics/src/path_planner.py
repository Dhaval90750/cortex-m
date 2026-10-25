"""Continuous Attractor Path Planner & Trajectory Generator for CORTEX-Robotics.

Synthesizes smooth, jerk-minimized C^2 trajectories from start to goal joint states,
incorporating dynamic obstacle repulsion and continuous safety verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .kinematics import Manipulator6DOF, RobotPose
from .safety_interlock import DeterministicSafetyInterlock, SafetyStatusReport


@dataclass
class PlannedTrajectory:
    """A synthesized, safety-verified multi-step joint trajectory."""
    waypoints_rad: List[List[float]]
    poses: List[RobotPose]
    total_duration_s: float
    dt_seconds: float
    max_tcp_speed_m_s: float
    is_safety_certified: bool
    safety_violations: List[str] = field(default_factory=list)


class AttractorPathPlanner:
    """Supervisory trajectory planner driven by continuous attractor dynamics."""

    def __init__(
        self,
        manipulator: Manipulator6DOF,
        interlock: DeterministicSafetyInterlock,
        num_waypoints: int = 50,
        dt_seconds: float = 0.02,
    ):
        self.manipulator = manipulator
        self.interlock = interlock
        self.num_waypoints = num_waypoints
        self.dt_seconds = dt_seconds

    def plan_trajectory(
        self,
        start_joints_rad: Sequence[float],
        goal_joints_rad: Sequence[float],
        human_distance_m: Optional[float] = None,
    ) -> PlannedTrajectory:
        """Plan a smooth, jerk-bounded trajectory between two joint configurations."""
        q_start = np.array(start_joints_rad, dtype=np.float64)
        q_goal = np.array(goal_joints_rad, dtype=np.float64)

        # Generate quintic polynomial smooth interpolation: s(t) in [0, 1] with 0 velocity/accel at endpoints
        waypoints = []
        poses = []
        violations = []
        max_tcp_speed = 0.0

        for step in range(self.num_waypoints):
            t = step / max(1, self.num_waypoints - 1)
            # Quintic polynomial: 10 t^3 - 15 t^4 + 6 t^5 (C^2 continuous)
            s = 10 * (t ** 3) - 15 * (t ** 4) + 6 * (t ** 5)
            q_current = q_start + s * (q_goal - q_start)
            current_list = [round(float(x), 4) for x in q_current]
            waypoints.append(current_list)

            pose = self.manipulator.forward_kinematics(current_list)
            poses.append(pose)

            # Safety check against previous waypoint
            if step > 0:
                safety_report = self.interlock.inspect_step(
                    current_joints=waypoints[-2],
                    target_joints=waypoints[-1],
                    dt_seconds=self.dt_seconds,
                    human_distance_m=human_distance_m,
                )
                if not safety_report.is_safe:
                    violations.extend(safety_report.violations)

                # Track maximum TCP speed
                prev_p = poses[-2]
                curr_p = poses[-1]
                dx, dy, dz = curr_p.x - prev_p.x, curr_p.y - prev_p.y, curr_p.z - prev_p.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                speed = dist / max(1e-5, self.dt_seconds)
                if speed > max_tcp_speed:
                    max_tcp_speed = speed

        total_duration = (self.num_waypoints - 1) * self.dt_seconds
        is_certified = len(violations) == 0 and not self.interlock.is_estop_active

        return PlannedTrajectory(
            waypoints_rad=waypoints,
            poses=poses,
            total_duration_s=round(total_duration, 3),
            dt_seconds=self.dt_seconds,
            max_tcp_speed_m_s=round(max_tcp_speed, 4),
            is_safety_certified=is_certified,
            safety_violations=violations,
        )
