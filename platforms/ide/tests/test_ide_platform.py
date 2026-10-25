"""Unit tests for CORTEX-IDE / CortexCode platform."""

import tempfile
import unittest
from pathlib import Path

from platforms.ide.src.ast_parser import ASTSymbolTableParser
from platforms.ide.src.diff_engine import ASTDiffEngine
from platforms.ide.src.sandbox import SubprocessSandboxRunner
from platforms.ide.src.swarm import CortexCodeSwarm, SwarmTaskPlan


class TestCortexIDEPlatform(unittest.TestCase):
    def setUp(self):
        self.parser = ASTSymbolTableParser()
        self.diff_engine = ASTDiffEngine()
        self.sandbox = SubprocessSandboxRunner(default_timeout_seconds=2.0)

    def test_ast_symbol_table_parsing(self):
        code = '''"""Module docstring."""

class Calculator:
    """Class docstring."""
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

def main():
    calc = Calculator()
    return calc.add(1, 2)
'''
        table = self.parser.parse_source(code, filename="calc.py")
        self.assertTrue(table.syntax_valid)
        self.assertIn("Calculator", table.classes)
        self.assertEqual(table.classes["Calculator"].docstring, "Class docstring.")
        self.assertIn("Calculator.add", table.functions)
        self.assertIn("main", table.functions)

    def test_ast_security_issue_detection(self):
        unsafe_code = '''
def run_dangerous(payload):
    try:
        eval(payload)
    except:
        pass
'''
        table = self.parser.parse_source(unsafe_code)
        self.assertTrue(table.syntax_valid)
        self.assertGreaterEqual(len(table.security_issues), 2)
        # Should flag both eval() and bare except:
        flags_text = " ".join(table.security_issues)
        self.assertIn("eval", flags_text)
        self.assertIn("bare", flags_text.lower())

    def test_ast_syntax_error_handling(self):
        bad_code = "def broken_syntax(:\n    pass"
        table = self.parser.parse_source(bad_code)
        self.assertFalse(table.syntax_valid)
        self.assertIn("SyntaxError", table.error_message)

    def test_sandbox_subprocess_execution_success(self):
        script = 'print("CORTEX_SANDBOX_OK")'
        res = self.sandbox.run_python_code(script)
        self.assertTrue(res.success)
        self.assertEqual(res.returncode, 0)
        self.assertIn("CORTEX_SANDBOX_OK", res.stdout)
        self.assertFalse(res.timed_out)

    def test_sandbox_subprocess_timeout_termination(self):
        # Infinite loop must be terminated at timeout limit
        infinite_script = "import time\nwhile True:\n    time.sleep(0.1)"
        res = self.sandbox.run_python_code(infinite_script, timeout_seconds=0.8)
        self.assertFalse(res.success)
        self.assertTrue(res.timed_out)
        self.assertIn("SandboxTimeoutExpired", res.stderr)

    def test_diff_engine_surgical_diff_and_ast_check(self):
        orig = "def compute(x):\n    return x * 2\n"
        valid_mod = "def compute(x):\n    # Optimized\n    return x << 1\n"
        invalid_mod = "def compute(x):\n    return x *\n"

        # Valid patch
        res_valid = self.diff_engine.validate_and_patch(orig, valid_mod)
        self.assertTrue(res_valid.success)
        self.assertTrue(res_valid.ast_valid)
        self.assertIn("+    return x << 1", res_valid.diff_text)

        # Invalid patch syntax
        res_invalid = self.diff_engine.validate_and_patch(orig, invalid_mod)
        self.assertFalse(res_invalid.success)
        self.assertFalse(res_invalid.ast_valid)
        self.assertIn("AST validation failed", res_invalid.error_message)

    def test_cortex_code_swarm_end_to_end_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            target = tmppath / "math_util.py"
            target.write_text("def multiply(a, b):\n    return a * b\n", encoding="utf-8")

            swarm = CortexCodeSwarm(sandbox_timeout=2.0, working_directory=tmppath)

            def synthesizer(plan: SwarmTaskPlan) -> str:
                return "def multiply(a, b):\n    \"\"\"Multiply two numbers.\"\"\"\n    return a * b\n"

            test_script = "import math_util\nassert math_util.multiply(3, 4) == 12"

            # Execute with human approval granted
            approved_calls = []

            def approval_gate(diff_res):
                approved_calls.append(True)
                return True

            res = swarm.execute_engineering_workflow(
                goal="Add docstring to multiply function",
                target_file="math_util.py",
                code_synthesizer=synthesizer,
                test_script=test_script,
                human_approval_callback=approval_gate,
            )

            self.assertTrue(res.success)
            self.assertTrue(res.critic_audit_passed)
            self.assertTrue(res.human_approved)
            self.assertEqual(len(approved_calls), 1)

            # Verify file on disk was modified safely
            content = target.read_text(encoding="utf-8")
            self.assertIn('"""Multiply two numbers."""', content)


if __name__ == "__main__":
    unittest.main()
