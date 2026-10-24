"""Complexity & Big-O Profiling Skill for Cortex-M & CortexCode.

Evaluates:
1. Time Complexity (Best, Average, Worst case Big-O).
2. Space Complexity (Auxiliary heap memory and recursion stack depth).
3. Loop nesting and asymptotic bottleneck identification.
4. Algorithmic trade-offs (e.g. In-place vs copy overhead).
"""

from __future__ import annotations

import ast
import re
import time
from typing import Any, Dict, List, Optional

from .base import BaseSkill, SkillResult


class ComplexityAnalysisSkill(BaseSkill):
    """Calculates theoretical Big-O time and space complexity."""

    def __init__(self):
        super().__init__(
            name="complexity",
            description="Analyzes theoretical Big-O time and space complexity, loop invariants, and asymptotic bottlenecks.",
            category="performance",
        )

    def execute(self, target: str, context: Optional[Dict[str, Any]] = None) -> SkillResult:
        t0 = time.perf_counter()
        target_lower = target.lower()

        time_comp = "O(n)"
        space_comp = "O(1)"
        bottlenecks: List[str] = []
        optimizations: List[str] = []

        # 1. Pattern & AST analysis
        max_loop_depth = 0
        has_recursion = False
        has_binary_division = False
        has_sorting = "sort(" in target_lower or "sorted(" in target_lower or "quicksort" in target_lower

        try:
            tree = ast.parse(target)

            def get_loop_depth(node: ast.AST, current_depth: int = 0) -> int:
                depths = [current_depth]
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.For, ast.While)):
                        depths.append(get_loop_depth(child, current_depth + 1))
                    else:
                        depths.append(get_loop_depth(child, current_depth))
                return max(depths)

            max_loop_depth = get_loop_depth(tree)

            # Check recursion
            func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in func_names:
                        has_recursion = True
        except SyntaxError:
            pass

        # Check binary division (mid = (low + high) // 2)
        if "// 2" in target or ">> 1" in target or "/ 2" in target:
            has_binary_division = True

        # 2. Derive Complexity
        if "lru" in target_lower or "hash" in target_lower or "dict" in target_lower and max_loop_depth == 0:
            time_comp = "O(1) amortized"
            space_comp = "O(C) where C is cache capacity"
            bottlenecks.append("Cache eviction occurs in O(1) time using linked/ordered hash tables.")

        elif has_binary_division and not has_recursion and max_loop_depth == 1:
            time_comp = "O(log n)"
            space_comp = "O(1)"
            bottlenecks.append("Divide-and-conquer search interval halves in each iteration.")

        elif has_sorting or (has_recursion and has_binary_division):
            time_comp = "O(n log n) average, O(n^2) worst case (if unbalanced partition)"
            space_comp = "O(log n) call stack" if not ("[" in target and "]" in target) else "O(n) auxiliary partition slices"
            bottlenecks.append("Pivot selection determines partition balance.")
            optimizations.append("Use median-of-three pivot selection to prevent worst-case O(n^2) degradation on sorted inputs.")

        elif max_loop_depth >= 3:
            time_comp = f"O(n^{max_loop_depth})"
            space_comp = "O(n) auxiliary"
            bottlenecks.append(f"Deeply nested loop structure ({max_loop_depth} levels). High polynomial scaling cost.")
            optimizations.append("Consider memoization, hash table lookups, or sliding-window algorithms to reduce loop depth.")

        elif max_loop_depth == 2:
            time_comp = "O(n^2) quadratic"
            space_comp = "O(1) auxiliary"
            bottlenecks.append("Nested loop iteration over inputs.")
            optimizations.append("Check if inner loop search can be replaced with a hash set or dictionary for O(n) linear time.")

        elif max_loop_depth == 1:
            time_comp = "O(n) linear"
            space_comp = "O(1) auxiliary" if "append" not in target else "O(n) auxiliary space"

        # 3. Format Report
        report = f"""Complexity Analysis Report:
• Time Complexity : {time_comp}
• Space Complexity: {space_comp}
• Loop Nesting    : {max_loop_depth} levels deep
• Recursion       : {'Detected' if has_recursion else 'None'}

Key Observations:
""" + "\n".join(f"  - {b}" for b in bottlenecks)

        if optimizations:
            report += "\n\nOptimization Suggestions:\n" + "\n".join(f"  * {opt}" for opt in optimizations)

        dt = (time.perf_counter() - t0) * 1000.0

        return SkillResult(
            skill_name=self.name,
            success=True,
            output=report,
            metadata={
                "time_complexity": time_comp,
                "space_complexity": space_comp,
                "loop_depth": max_loop_depth,
                "has_recursion": has_recursion,
            },
            execution_time_ms=round(dt, 2),
            resonance_score=0.91,
            findings=bottlenecks,
            suggestions=optimizations,
        )
