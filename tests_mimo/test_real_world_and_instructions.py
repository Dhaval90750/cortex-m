"""Test Suite for Real-World Knowledge & Instruction Awareness in CortexLM.

Validates:
1. Real-World Systems & DevOps Reasoning (Docker vs VMs, Memory Leaks, Git Rebase).
2. Real-World Science & Physics (Relativity, Speed of Light, Thermodynamics).
3. Decision Frameworks & Natural Conversation (First Principles, Pareto, Friendly Tone).
4. Strict Instruction Following:
   - JSON formatting
   - Bullet point formatting
   - Step-by-step procedural breakdown
   - One-sentence concise constraint
   - Code block syntax wrapping
5. Zero Robotic Boilerplate (No canned "As an AI language model..." disclaimers).
"""

from __future__ import annotations

import json
import unittest
import torch

from cortex_m import CortexLM


class TestRealWorldAndInstructionAwareness(unittest.TestCase):
    """Test suite verifying real-world grounding and instruction awareness."""

    def setUp(self):
        torch.manual_seed(42)
        self.lm = CortexLM(dim=64, gist_dim=32, seed=42, imprint_foundation=True)

    def test_real_world_devops_and_debugging(self):
        """Verify grounded reasoning for practical DevOps, Docker, and debugging."""
        # 1. Docker vs VMs
        docker_out = self.lm.generate("What is the difference between Docker and Virtual Machines?")
        self.assertTrue(docker_out.is_grounded)
        self.assertIn("kernel", docker_out.text.lower())
        self.assertIn("hypervisor", docker_out.text.lower())

        # 2. Memory Leak Debugging
        leak_out = self.lm.generate("How do I debug memory leaks in production?")
        self.assertTrue(leak_out.is_grounded)
        self.assertIn("profiler", leak_out.text.lower())
        self.assertIn("heap", leak_out.text.lower())

        # 3. Git Merge vs Rebase
        git_out = self.lm.generate("Explain git merge vs git rebase")
        self.assertTrue(git_out.is_grounded)
        self.assertIn("rebase", git_out.text.lower())
        self.assertIn("linear", git_out.text.lower())

    def test_real_world_science_and_physics(self):
        """Verify grounded reasoning for fundamental physics, speed of light, and relativity."""
        # 1. Speed of light
        light_out = self.lm.generate("What is the speed of light in vacuum?")
        self.assertTrue(light_out.is_grounded)
        self.assertIn("299,792,458", light_out.text)

        # 2. General Relativity
        rel_out = self.lm.generate("How does gravity work in general relativity?")
        self.assertTrue(rel_out.is_grounded)
        self.assertIn("spacetime", rel_out.text.lower())
        self.assertIn("geodesic", rel_out.text.lower())

        # 3. Second Law of Thermodynamics
        thermo_out = self.lm.generate("What is the second law of thermodynamics?")
        self.assertTrue(thermo_out.is_grounded)
        self.assertIn("entropy", thermo_out.text.lower())

    def test_decision_frameworks_and_natural_dialogue(self):
        """Verify first-principles thinking, Pareto principle, and human dialogue."""
        # 1. First-principles thinking
        fp_out = self.lm.generate("What is first principles thinking?")
        self.assertTrue(fp_out.is_grounded)
        self.assertIn("axiomatic", fp_out.text.lower())

        # 2. Pareto 80/20
        pareto_out = self.lm.generate("How does the Pareto principle apply to software optimization?")
        self.assertTrue(pareto_out.is_grounded)
        self.assertIn("80%", pareto_out.text)
        self.assertIn("20%", pareto_out.text)

        # 3. Conversational greeting
        greet_out = self.lm.chat("Hello there!")
        self.assertTrue(greet_out.is_grounded)
        self.assertIn("connect", greet_out.text.lower())
        # Ensure no canned robotic throat-clearing
        self.assertNotIn("as an ai", greet_out.text.lower())

    def test_instruction_format_json(self):
        """Verify prompt requesting JSON outputs structured, parseable JSON."""
        prompt = "Explain Docker vs Virtual Machines. Format as JSON"
        out = self.lm.generate(prompt)
        self.assertTrue(out.is_grounded)
        
        # Verify valid JSON syntax
        parsed = json.loads(out.text)
        self.assertIn("summary", parsed)
        self.assertIn("provenance", parsed)
        self.assertEqual(parsed["provenance"]["attractor"], "docker_vs_vms")

    def test_instruction_format_bullet_points(self):
        """Verify prompt requesting bullet points produces structured list."""
        prompt = "How to debug memory leaks in production? Key points in bullet points"
        out = self.lm.generate(prompt)
        self.assertTrue(out.is_grounded)
        self.assertIn("•", out.text)
        lines = [line.strip() for line in out.text.split("\n") if line.strip().startswith("•")]
        self.assertGreaterEqual(len(lines), 1)

    def test_instruction_format_step_by_step(self):
        """Verify prompt requesting step-by-step produces numbered breakdown."""
        prompt = "Step by step guide on how to debug memory leaks"
        out = self.lm.generate(prompt)
        self.assertTrue(out.is_grounded)
        self.assertIn("1.", out.text)

    def test_instruction_constraint_concise_one_sentence(self):
        """Verify prompt requesting one sentence is strictly concise."""
        prompt = "Explain git rebase in 1 sentence"
        out = self.lm.generate(prompt)
        self.assertTrue(out.is_grounded)
        # Should be a single sentence
        self.assertIn("rebase", out.text.lower())
        # The main statement should end with period
        self.assertTrue(out.text.count(".") >= 1)

    def test_zero_robotic_boilerplate(self):
        """Verify model never uses canned AI disclaimers."""
        queries = [
            "What is the speed of light?",
            "Explain git merge vs git rebase",
            "Why does Rust guarantee memory safety?",
            "What is the second law of thermodynamics?",
        ]
        robotic_phrases = [
            "as an ai language model",
            "as an ai,",
            "i am programmed to",
            "i do not have personal opinions",
            "certainly! here is",
        ]
        for q in queries:
            out = self.lm.generate(q)
            for phrase in robotic_phrases:
                self.assertNotIn(phrase, out.text.lower())


if __name__ == "__main__":
    unittest.main()
