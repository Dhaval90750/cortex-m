"""Verification Test: Dynamic Cognitive Skills & Synthesis without Hardcoding."""

import sys
sys.path.insert(0, "src")

import unittest
from cortex_m import (
    CortexV2,
    MultiAgentTaskCompleter,
    PlanningSkill,
    AskingSkill,
    SuggestingSkill,
    GrillingSkill,
    SchedulingSkill,
    GoalSkill,
    AlignerSkill,
    SkillRegistry,
)
from cortex_m.agents.code_synthesizer import DynamicCodeSynthesizer


class TestDynamicCognitiveSkills(unittest.TestCase):
    def setUp(self):
        self.cortex = CortexV2(dim=64, gist_dim=32, seed=42)
        self.completer = MultiAgentTaskCompleter()
        self.skills = self.cortex.skills

    def test_planning_skill_dynamic_and_no_lru_hardcode(self):
        """Verify planning skill produces custom, tailored milestones for diverse domains."""
        # Domain 1: Distributed Consensus
        res1 = self.skills.execute("planning", "distributed consensus with raft leader election")
        self.assertTrue(res1.success)
        self.assertIn("Raft", res1.output)
        self.assertNotIn("LRU", res1.output)
        self.assertNotIn("lru", res1.output.lower())
        self.assertGreaterEqual(len(res1.metadata.get("milestones", [])), 4)

        # Domain 2: Audio DSP
        res2 = self.skills.execute("planning", "audio waveform resampler with sinc interpolation")
        self.assertTrue(res2.success)
        self.assertIn("Resampler", res2.output)
        self.assertNotIn("LRU", res2.output)
        self.assertNotIn("lru", res2.output.lower())

    def test_asking_skill_dynamic(self):
        """Verify asking skill formulates options referencing actual prompt entities."""
        res = self.skills.execute("asking", "distributed rate limiter with token bucket")
        self.assertTrue(res.success)
        self.assertIn("Rate", res.output)
        self.assertGreaterEqual(res.metadata.get("count", 0), 2)

    def test_suggesting_skill_dynamic(self):
        """Verify suggesting skill dynamically recommends optimizations."""
        sample_code = """
class OrderProcessor:
    def process(self, orders):
        res = []
        for o in orders:
            res.append(o['id'])
        return res
"""
        res = self.skills.execute("suggesting", sample_code)
        self.assertTrue(res.success)
        self.assertIn("RECOMMENDATIONS", res.output)
        self.assertGreaterEqual(len(res.metadata.get("suggestions", [])), 2)

    def test_grilling_skill_dynamic(self):
        """Verify grilling skill interrogates failure modes of the target component."""
        res = self.skills.execute("grilling", "in-memory payment settlement ledger")
        self.assertTrue(res.success)
        self.assertIn("Settlement", res.output)
        self.assertIn("INTERROGATION", res.output)
        self.assertGreaterEqual(len(res.metadata.get("interrogations", [])), 5)

    def test_scheduling_skill_dynamic(self):
        """Verify scheduling skill executes sleep consolidation and timer."""
        res_sleep = self.skills.execute("scheduling", "sleep consolidation")
        self.assertTrue(res_sleep.success)
        self.assertIn("Sleep Consolidation", res_sleep.output)

        res_timer = self.skills.execute("scheduling", "remind in 15m to check memory")
        self.assertTrue(res_timer.success)
        self.assertIn("15m", res_timer.output)

    def test_aligner_skill_dynamic(self):
        """Verify aligner detects matching concepts and omissions."""
        intent = "Implement a bank account with deposit, withdraw, and transaction history"
        code_good = """
class BankAccount:
    def __init__(self):
        self.balance = 0.0
        self.history = []
    def deposit(self, amount):
        self.balance += amount
        self.history.append(('deposit', amount))
    def withdraw(self, amount):
        self.balance -= amount
        self.history.append(('withdraw', amount))
"""
        res_good = self.skills.execute("aligner", code_good, context={"intent": intent, "code": code_good})
        self.assertTrue(res_good.success)
        self.assertGreaterEqual(res_good.metadata.get("alignment_score", 0), 80.0)

    def test_code_synthesizer_dynamic_bank_account(self):
        """Verify DynamicCodeSynthesizer synthesizes real BankAccount with working methods."""
        synth = DynamicCodeSynthesizer()
        code = synth.synthesize("implement BankAccount with deposit and withdraw")
        self.assertIn("class BankAccount", code)
        self.assertIn("def deposit(", code)
        self.assertIn("def withdraw(", code)
        self.assertIn("def transfer(", code)
        self.assertIn("def get_balance(", code)
        self.assertNotIn("set_item", code)  # No dummy stubs!
        
        # Test that synthesized code actually executes cleanly
        namespace = {}
        exec(code, namespace)
        BankAccount = namespace["BankAccount"]
        acc = BankAccount("test_01", initial_balance=100.0)
        acc.deposit(50.0)
        self.assertEqual(acc.get_balance(), 150.0)
        acc.withdraw(30.0)
        self.assertEqual(acc.get_balance(), 120.0)

    def test_code_synthesizer_dynamic_sensor_monitor(self):
        """Verify DynamicCodeSynthesizer synthesizes real SensorMonitor with working methods."""
        synth = DynamicCodeSynthesizer()
        code = synth.synthesize("create SensorMonitor with record_reading and alert")
        self.assertIn("class SensorMonitor", code)
        self.assertIn("def record_reading(", code)
        self.assertIn("def get_average(", code)
        self.assertNotIn("set_item", code)
        
        namespace = {}
        exec(code, namespace)
        SensorMonitor = namespace["SensorMonitor"]
        sensor = SensorMonitor("S-1", alert_threshold=50.0)
        sensor.record_reading(20.0)
        sensor.record_reading(60.0)
        self.assertEqual(sensor.get_average(), 40.0)
        self.assertEqual(len(sensor.get_alerts()), 1)


if __name__ == "__main__":
    unittest.main()
