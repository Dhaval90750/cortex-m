"""Unit tests for CORTEX-Robotics platform."""

import math
import unittest

from platforms.robotics.src.kinematics import KinematicLimits, Manipulator6DOF, RobotPose
from platforms.robotics.src.path_planner import AttractorPathPlanner, PlannedTrajectory
from platforms.robotics.src.safety_interlock import DeterministicSafetyInterlock, SafetyStatusReport


class TestCortexRoboticsPlatform(unittest.TestCase):
    def setUp(self):
        self.manipulator = Manipulator6DOF()
        self.interlock = DeterministicSafetyInterlock(
            manipulator=self.manipulator,
            strict_mode=False,
        )
        self.planner = AttractorPathPlanner(
            manipulator=self.manipulator,
            interlock=self.interlock,
            num_waypoints=25,
            dt_seconds=0.04,
        )

    def test_forward_kinematics(self):
        # Zero configuration
        zero_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pose = self.manipulator.forward_kinematics(zero_joints)
        self.assertIsInstance(pose, RobotPose)
        self.assertGreater(pose.x, 0.0)  # Arm extends forward in X
        self.assertGreater(pose.z, 0.0)  # Base elevation

        # Joint limits check
        self.assertTrue(self.manipulator.validate_joint_limits(zero_joints))
        out_of_bounds = [4.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Exceeds pi
        self.assertFalse(self.manipulator.validate_joint_limits(out_of_bounds))

    def test_safety_interlock_speed_and_separation_monitoring(self):
        q_curr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # Moderate safe step
        q_target = [0.01, 0.01, 0.0, 0.0, 0.0, 0.0]

        # 1. Normal clearance (distance = 2.5m)
        report_safe = self.interlock.inspect_step(q_curr, q_target, dt_seconds=0.05, human_distance_m=2.5)
        self.assertTrue(report_safe.is_safe)
        self.assertEqual(len(report_safe.violations), 0)

        # 2. Critical human intrusion (distance = 0.3m <= 0.50m) requires immediate halt
        report_intrusion = self.interlock.inspect_step(q_curr, q_target, dt_seconds=0.05, human_distance_m=0.3)
        self.assertFalse(report_intrusion.is_safe)
        self.assertEqual(report_intrusion.allowed_tcp_speed_m_s, 0.0)

        # 3. Excessive joint velocity violation
        q_jump = [1.5, 0.0, 0.0, 0.0, 0.0, 0.0]  # Jump 1.5 rad in 0.01s = 150 rad/s
        report_jump = self.interlock.inspect_step(q_curr, q_jump, dt_seconds=0.01)
        self.assertFalse(report_jump.is_safe)
        self.assertTrue(any("velocity" in v for v in report_jump.violations))

    def test_estop_latching(self):
        self.interlock.trigger_estop()
        self.assertTrue(self.interlock.is_estop_active)

        q_curr = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        q_target = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]
        report = self.interlock.inspect_step(q_curr, q_target, dt_seconds=0.1)
        self.assertFalse(report.is_safe)
        self.assertTrue(any("E-Stop active" in v for v in report.violations))

        self.interlock.reset_estop()
        self.assertFalse(self.interlock.is_estop_active)

    def test_attractor_trajectory_planning(self):
        q_start = [0.0, 0.2, -0.2, 0.0, 0.0, 0.0]
        q_goal = [0.4, 0.5, 0.1, 0.2, 0.1, 0.0]

        traj = self.planner.plan_trajectory(q_start, q_goal, human_distance_m=2.0)
        self.assertIsInstance(traj, PlannedTrajectory)
        self.assertEqual(len(traj.waypoints_rad), 25)
        self.assertEqual(len(traj.poses), 25)
        self.assertTrue(traj.is_safety_certified)
        self.assertEqual(len(traj.safety_violations), 0)

        # Verify start and goal endpoints
        self.assertAlmostEqual(traj.waypoints_rad[0][0], q_start[0], places=3)
        self.assertAlmostEqual(traj.waypoints_rad[-1][0], q_goal[0], places=3)


if __name__ == "__main__":
    unittest.main()
