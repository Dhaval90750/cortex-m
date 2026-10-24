"""Test Suite for Database Mastery, Adaptive Language, and Metacognitive Reasoning.

Validates:
1. SQL DDL Schema Ingestion into Synaptic Weights (< 1 ms).
2. Grounded Database Reasoning (B-Tree vs LSM, ACID vs BASE, Query Optimization).
3. Adaptive Human-Understandable Explanations (ELI5 Analogies on User Demand).
4. 5W1H Metacognitive Assessment (When, Why, What, Where, How).
5. Proactive Suggestion Engine & Active Learning Pattern Reinforcement.
"""

from __future__ import annotations

import unittest
import torch

from cortex_m import CortexLM, DatabaseKnowledgeTrainer


SAMPLE_ECOMMERCE_DDL = """
CREATE TABLE users (
    id INT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    created_at TIMESTAMP
);

CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    user_id INT NOT NULL,
    total_amount DECIMAL(10, 2),
    status VARCHAR(50),
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE order_items (
    item_id INT PRIMARY KEY,
    order_id INT NOT NULL,
    product_sku VARCHAR(100),
    quantity INT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);
"""


class TestDatabaseAndMetacognition(unittest.TestCase):
    """Test suite verifying database mastery, adaptive explanations, and metacognition."""

    def setUp(self):
        torch.manual_seed(42)
        self.lm = CortexLM(dim=64, gist_dim=32, seed=42, imprint_foundation=True)

    def test_database_systems_theory_reasoning(self):
        """Verify grounded reasoning across storage engines, ACID, and query optimization."""
        # 1. B-Tree vs LSM-Tree
        btree_out = self.lm.generate("Compare B-Tree and LSM-Tree database storage engines")
        self.assertTrue(btree_out.is_grounded)
        self.assertIn("sstable", btree_out.text.lower())
        self.assertIn("write", btree_out.text.lower())

        # 2. ACID vs BASE
        acid_out = self.lm.generate("What is the difference between ACID and BASE in database transactions?")
        self.assertTrue(acid_out.is_grounded)
        self.assertIn("atomicity", acid_out.text.lower())
        self.assertIn("eventual consistency", acid_out.text.lower())

        # 3. SQL Query Optimization
        opt_out = self.lm.generate("How does SQL query optimization work with indexes?")
        self.assertTrue(opt_out.is_grounded)
        self.assertIn("planner", opt_out.text.lower())

    def test_sql_ddl_schema_synaptic_ingestion(self):
        """Verify parsing SQL DDL and directly querying the learned schema."""
        # 1. Ingest DDL schema into synaptic memory
        res = self.lm.connect_database_schema(SAMPLE_ECOMMERCE_DDL, database_name="shop_db")
        self.assertEqual(res["status"], "schema_imprinted_in_synapses")
        self.assertEqual(res["total_tables_ingested"], 3)
        self.assertIn("orders", res["tables"])

        # 2. Query the ingested schema directly
        query_out = self.lm.generate("What columns are in the orders table in shop_db?")
        self.assertTrue(query_out.is_grounded)
        self.assertIn("total_amount", query_out.text.lower())
        self.assertIn("status", query_out.text.lower())

        # 3. Query foreign key relationship
        rel_out = self.lm.generate("What does orders user_id reference in shop_db?")
        self.assertTrue(rel_out.is_grounded)
        self.assertIn("users", rel_out.text.lower())

    def test_adaptive_human_understandable_language_eli5(self):
        """Verify model dynamically adapts to user demand for plain-English analogies."""
        # Query with explicit ELI5 demand
        prompt = "Explain Docker vs Virtual Machines like I'm 5 with a simple analogy"
        out = self.lm.generate(prompt)
        self.assertTrue(out.is_grounded)
        # Should produce the apartment vs house analogy rather than technical kernel jargon
        self.assertIn("apartment", out.text.lower())
        self.assertIn("house", out.text.lower())

        # Query B-Tree with ELI5 demand
        db_prompt = "Explain B-Trees vs LSM-Trees in simple terms for a beginner"
        db_out = self.lm.generate(db_prompt)
        self.assertTrue(db_out.is_grounded)
        self.assertIn("library", db_out.text.lower())

    def test_metacognitive_5w1h_assessment(self):
        """Verify 5W1H metacognitive trace is populated on generation."""
        out = self.lm.generate("How does SQL query optimization work with indexes?")
        self.assertTrue(out.is_grounded)
        self.assertIsNotNone(out.metacognition)
        
        meta = out.metacognition
        self.assertIn("when_mode", meta)
        self.assertIn("why_rationale", meta)
        self.assertIn("what_topic", meta)
        self.assertIn("where_attractor", meta)
        self.assertIn("how_comprehension_level", meta)
        self.assertEqual(meta["when_mode"], "DELIBERATIVE_PREFRONTAL")
        self.assertEqual(out.comprehension_level, "STANDARD_EXPERT")

    def test_proactive_suggestions_and_active_learning(self):
        """Verify CortexLM offers proactive followups and learns user stack from feedback."""
        # 1. Query generates contextual suggestions
        out = self.lm.generate("How should I optimize my SQL database queries?")
        self.assertTrue(len(out.suggestions) > 0)
        self.assertTrue(any("index" in s.lower() or "explain" in s.lower() for s in out.suggestions))

        # 2. Active learning: User tells CortexLM their stack in chat
        chat_turn1 = self.lm.chat("We use PostgreSQL in our production database stack.")
        self.assertIn("database", self.lm.metacognition.user_learned_patterns)
        self.assertEqual(self.lm.metacognition.user_learned_patterns["database"], "Postgresql")

        # 3. CortexLM immediately recalls learned user preference from synapses
        chat_turn2 = self.lm.chat("What is our production database engine?")
        self.assertTrue(chat_turn2.is_grounded)
        self.assertIn("Postgresql", chat_turn2.text)


if __name__ == "__main__":
    unittest.main()
