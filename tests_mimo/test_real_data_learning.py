"""Tests for Real Data Ingestion & Everyday Normal User Intelligence in CortexLM."""

import os
import unittest
import torch

from cortex_m import CortexLM, RealDataTrainer, EVERYDAY_USER_KNOWLEDGE


class TestRealDataLearning(unittest.TestCase):
    """Test suite for real data ingestion, everyday lifestyle intelligence, and normal user flows."""

    @classmethod
    def setUpClass(cls):
        # Initialize CortexLM with everyday normal user knowledge auto-imprinted
        cls.lm = CortexLM(dim=64, gist_dim=32, device="cpu", seed=42, imprint_foundation=True)

        # Ingest real episodic dataset from data/generated/dev.jsonl
        dev_path = os.path.join(os.path.dirname(__file__), "..", "data", "generated", "dev.jsonl")
        cls.dev_path = os.path.abspath(dev_path)
        cls.ingest_stats = cls.lm.learn_real_data(cls.dev_path, max_samples=20)

    def test_everyday_lifestyle_knowledge_imprinted(self):
        """Verify normal user life, health, finance, and productivity facts are grounded."""
        # 1. 50/30/20 Budgeting rule
        budget_out = self.lm.generate("How does the 50/30/20 budgeting rule work?")
        self.assertTrue(budget_out.is_grounded)
        self.assertIn("50", budget_out.text)
        self.assertIn("30", budget_out.text)
        self.assertIn("20", budget_out.text)

        # 2. Sleep hygiene & circadian rhythm
        sleep_out = self.lm.generate("What are tips for good sleep hygiene and circadian rhythms?")
        self.assertTrue(sleep_out.is_grounded)
        self.assertIn("circadian", sleep_out.text.lower())

        # 3. Pomodoro Technique
        pomo_out = self.lm.generate("How does the Pomodoro Technique boost productivity with 25-minute intervals?")
        self.assertTrue(pomo_out.is_grounded)
        self.assertIn("25", pomo_out.text)

    def test_real_jsonl_dataset_recall(self):
        """Verify real user episodic preferences from dev.jsonl are retrieved with zero hallucination."""
        # Ingested keys from dev.jsonl: 'preferred editor' (Vim), 'usual drink' (green tea), 'travel month' (October)
        self.assertGreater(self.ingest_stats["total_imprinted"], 0)

        # Query 1: Preferred editor
        editor_out = self.lm.generate("What is my preferred editor?")
        self.assertTrue(editor_out.is_grounded)
        self.assertIn("vim", editor_out.text.lower())

        # Query 2: Usual drink
        drink_out = self.lm.generate("What is my usual drink?")
        self.assertTrue(drink_out.is_grounded)
        self.assertIn("green tea", drink_out.text.lower())

    def test_normal_user_conversational_chat(self):
        """Verify multi-turn natural dialogue for a normal user asking about daily life and habits."""
        # Turn 1: Ask about hydration
        turn1 = self.lm.chat("How much daily water is recommended for healthy hydration guidelines?")
        self.assertTrue(turn1.is_grounded)
        self.assertIn("liters", turn1.text.lower())

        # Turn 2: Follow-up question using working memory
        turn2 = self.lm.chat("Does that daily amount change with physical activity?")
        self.assertTrue(turn2.is_grounded)

    def test_real_data_targeted_anti_hebbian_unlearning(self):
        """Verify instant anti-Hebbian unlearning for real user preferences."""
        # Create a fresh mini LM for clean isolation
        fresh_lm = CortexLM(dim=64, gist_dim=32, device="cpu", seed=100, imprint_foundation=False)
        fresh_lm.learn("User preference for 'favorite sport' is 'badminton'.", memory_id="realdata_favorite_sport")

        pre_out = fresh_lm.generate("What is my favorite sport?")
        self.assertTrue(pre_out.is_grounded)
        self.assertIn("badminton", pre_out.text.lower())

        # Instant Anti-Hebbian depression
        fresh_lm.forget("realdata_favorite_sport")

        post_out = fresh_lm.generate("What is my favorite sport?")
        # Should cleanly abstain after unlearning
        self.assertFalse(post_out.is_grounded)


if __name__ == "__main__":
    unittest.main()
