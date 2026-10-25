"""
Unit test suite for CORTEX-M Core Runtime and Architecture Contracts.
"""

import unittest
from cortex_m.core import (
    CortexConfig,
    CortexRuntime,
    EventBus,
    CortexEvent,
    EventType,
    DeploymentProfile,
)


class TestCoreRuntime(unittest.TestCase):
    def setUp(self):
        self.runtime = CortexRuntime()

    def test_runtime_learn_and_ask_grounding(self):
        fact = "Antigravity is Google DeepMind agentic pair programmer"
        self.runtime.learn(fact)
        result = self.runtime.ask("What is Antigravity?")
        self.assertTrue(result["is_grounded"])
        self.assertIn(result["evidence_status"], ["VERIFIED", "PARTIALLY_SUPPORTED"])
        self.assertIn("Antigravity", result["text"])

    def test_event_bus_pubsub(self):
        bus = EventBus()
        received_events = []

        bus.subscribe(EventType.TASK_STARTED.value, lambda e: received_events.append(e))
        bus.publish(CortexEvent(
            event_type=EventType.TASK_STARTED,
            payload={"task": "git_patch"},
            task_id="task-001",
        ))

        self.assertEqual(len(received_events), 1)
        self.assertEqual(received_events[0].task_id, "task-001")
        self.assertEqual(received_events[0].payload["task"], "git_patch")

    def test_config_profiles(self):
        cfg = CortexConfig(profile=DeploymentProfile.EDGE)
        self.assertEqual(cfg.profile.value, "edge")
        cfg_dict = cfg.to_dict()
        self.assertEqual(cfg_dict["profile"], "edge")


if __name__ == "__main__":
    unittest.main()
