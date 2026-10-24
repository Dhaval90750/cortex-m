"""Tests for Cortex-M Multi-Agent Task Completer and Synaptic Blackboard."""

import unittest

from cortex_m.agents.blackboard import AgentMessage, AgentRole, SynapticBlackboard
from cortex_m.agents.coding_agent import CortexCodingAgent
from cortex_m.agents.task_completer import MultiAgentTaskCompleter


class TestMultiAgentTaskCompleter(unittest.TestCase):
    """Verify autonomous multi-agent task completion, synaptic blackboard, and self-repair."""

    def setUp(self):
        self.blackboard = SynapticBlackboard(dim=64, beta=10.0, seed=42)
        self.completer = MultiAgentTaskCompleter(blackboard=self.blackboard)

    def test_synaptic_blackboard_posting_and_associative_query(self):
        """Verify zero-database multi-agent message exchange through synaptic attractors."""
        # 1. Post messages from distinct agent roles
        msg1 = self.blackboard.post(
            sender_role=AgentRole.ARCHITECT,
            topic="architecture_decision",
            content="Use continuous Modern Hopfield network for fast associative memory.",
        )
        msg2 = self.blackboard.post(
            sender_role=AgentRole.CODER,
            topic="implementation_detail",
            content="Implemented LRUCache using collections.OrderedDict with capacity bounds.",
        )
        msg3 = self.blackboard.post(
            sender_role=AgentRole.CRITIC,
            topic="security_audit",
            content="Flagged insecure dynamic eval usage in authentication handler.",
        )

        self.assertEqual(len(self.blackboard.messages), 3)

        # 2. Query blackboard associatively via Lyapunov energy descent
        recalled = self.blackboard.query("What cache data structure did the coder implement?")
        self.assertTrue(len(recalled) > 0)
        self.assertIn("lrucache", recalled[0].content.lower())

        # 3. Query with role filter
        critic_msgs = self.blackboard.query("security issues", filter_role=AgentRole.CRITIC)
        self.assertTrue(len(critic_msgs) > 0)
        self.assertEqual(critic_msgs[0].sender_role, AgentRole.CRITIC)

    def test_coding_agent_plan_and_synthesis(self):
        """Verify architect planning and coder code synthesis with AST validation."""
        agent = CortexCodingAgent(AgentRole.CODER, blackboard=self.blackboard)

        # 1. Architect plan
        plan = agent.plan("Design a high-throughput quicksort algorithm")
        self.assertIn("milestones", plan)
        self.assertGreaterEqual(len(plan["milestones"]), 3)

        # 2. Synthesize code
        artifact = agent.synthesize_code(
            task_description="Implement quicksort algorithm in python",
            filename="quicksort.py",
        )
        self.assertTrue(artifact.is_valid_syntax)
        self.assertIn("quicksort", artifact.functions)
        self.assertTrue(any("List" in imp for imp in artifact.imports))

    def test_critic_review_and_security_inspection(self):
        """Verify critic static analysis and security risk detection."""
        critic = CortexCodingAgent(AgentRole.CRITIC, blackboard=self.blackboard)
        coder = CortexCodingAgent(AgentRole.CODER, blackboard=self.blackboard)

        # 1. Safe artifact
        safe_art = coder.synthesize_code("Implement quicksort", filename="safe.py")
        safe_review = critic.review_code(safe_art)
        self.assertTrue(safe_review.is_approved)
        self.assertEqual(safe_review.risk_score, 0.0)

        # 2. Insecure artifact with dynamic eval
        insecure_art = coder._analyze_code_ast(
            filename="insecure.py",
            code="def run_code(user_input: str):\n    return eval(user_input)\n",
            language="python",
        )
        insecure_review = critic.review_code(insecure_art)
        self.assertFalse(insecure_review.is_approved)
        self.assertGreater(insecure_review.risk_score, 0.5)
        self.assertTrue(any("eval" in issue for issue in insecure_review.issues_found))

    def test_end_to_end_task_completer_lru_cache(self):
        """Verify complete multi-agent workflow: plan -> code -> review -> execute -> audit."""
        res = self.completer.execute_task(
            goal="Implement an LRU cache with capacity limit",
            filename="lru_cache.py",
            verify_executable=True,
            test_fn=lambda scope: "LRUCache" in scope and scope["LRUCache"](2) is not None,
        )

        self.assertEqual(res.status, "completed")
        self.assertTrue(len(res.artifacts) > 0)
        self.assertTrue(res.artifacts[0].is_valid_syntax)
        self.assertIn("LRUCache", res.artifacts[0].classes)
        self.assertTrue(res.audit_summary["runtime_verified"])
        self.assertGreater(res.synaptic_memories_imprinted, 0)
        self.assertGreater(res.execution_time_sec, 0.0)


if __name__ == "__main__":
    unittest.main()
