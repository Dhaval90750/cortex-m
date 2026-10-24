from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .assistant import RuleBasedCortex
from .store import MemoryStore


class CortexMIMOSystem:
    """One-input orchestrator for Cortex gist/memory and the M2M-35D specialist.

    MIMO is dispatched only for an explicit JSON relational-task request. The
    gist is retrieved and recorded beside the solver trace. The archived MIMO
    model does not consume gist embeddings or evidence, so the result labels
    this as orchestration rather than claiming gist-conditioned computation.
    """

    def __init__(self, *, keys=(), values=(), store: MemoryStore | None = None,
                 checkpoint_path: str | Path | None = None,
                 gist_checkpoint_path: str | Path | None = None) -> None:
        self.gist_checkpoint_path = Path(gist_checkpoint_path) if gist_checkpoint_path else None
        extra_keys = ("mode",) if self.gist_checkpoint_path else ()
        extra_values = ("alpha", "beta") if self.gist_checkpoint_path else ()
        self.cortex = RuleBasedCortex((*keys, *extra_keys), (*values, *extra_values), store=store)
        self.store = self.cortex.store
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model = None
        self._gist_model = None
        self._device = "cpu"

    def run(self, prompt: str) -> dict[str, Any]:
        """Handle either ordinary memory chat or the documented MIMO task JSON."""
        try:
            request = json.loads(prompt)
        except (json.JSONDecodeError, TypeError):
            return self._run_memory_prompt(prompt)
        if not isinstance(request, dict) or request.get("task") != "mimo_n8":
            if isinstance(request, dict) and request.get("task") == "mimo_gist_n8":
                return self._run_gist_mimo(request)
            if isinstance(request, dict) and request.get("task") == "support_gist_route":
                return self._run_support_gist(request)
            return self._run_memory_prompt(prompt)
        required = {"values", "role_ids"}
        if not required.issubset(request):
            return {"kind": "error", "response": "MIMO request requires values and role_ids.",
                    "stages": [{"name": "Task validation", "status": "failed",
                                "detail": "Missing required fields: values and/or role_ids."}]}
        query = str(request.get("gist_query", request.get("instruction", "MIMO N=8 relational task")))
        gist = self.cortex.trace_query(query)
        if self.checkpoint_path is None:
            return {"kind": "error", "response": "MIMO checkpoint path is not configured.",
                    "stages": gist["stages"] + [{"name": "MIMO dispatch", "status": "failed",
                        "detail": "Provide a trained M2M-SparseRoute checkpoint path."}],
                    "gist": gist, "gist_used_for_computation": False}
        if self._model is None:
            try:
                from cortex_m.mimo.m2m_sparse_route import load_model
            except ImportError as exc:
                return {"kind": "error", "response": f"Optional MIMO runtime unavailable: {exc}",
                        "stages": gist["stages"] + [{"name": "MIMO dispatch", "status": "failed",
                            "detail": "Install the optional research dependencies in .venv-mimo."}],
                        "gist": gist, "gist_used_for_computation": False}
            self._model, _ = load_model(self.checkpoint_path, device=self._device)
        from cortex_m.mimo.m2m_sparse_route import infer
        try:
            output = infer(self._model, request["values"], request["role_ids"], device=self._device)
        except (ValueError, TypeError, RuntimeError) as exc:
            return {"kind": "error", "response": f"MIMO input rejected: {exc}",
                    "stages": gist["stages"] + [{"name": "MIMO input validation", "status": "failed",
                        "detail": str(exc)}], "gist": gist, "gist_used_for_computation": False}
        stage = {"name": "M2M-SparseRoute v1 inference", "status": "done",
                 "detail": "Trained CPU checkpoint produced numeric outputs and top-K physical routes.",
                 "data": output}
        readable = "M2M-SparseRoute outputs: " + json.dumps(output["outputs"]) + "; selected physical input positions: " + json.dumps(output["selected_physical_positions"])
        return {"kind": "integrated_mimo", "response": readable, "solver_output": output,
                "gist": gist,
                "stages": gist["stages"] + [stage],
                "action": "mimo-inference", "facts": gist["facts"],
                "gist_used_for_computation": False,
                "integration_status": "gist-retrieval-and-MIMO-orchestration; no gist conditioning"}

    def _run_gist_mimo(self, request: dict[str, Any]) -> dict[str, Any]:
        required = {"values", "role_ids"}
        if not required.issubset(request):
            return {"kind": "error", "action": "invalid-task",
                    "response": "Gist MIMO request requires values and role_ids.", "facts": [], "stages": []}
        query = str(request.get("gist_query", "what is my mode?"))
        gist = self.cortex.trace_query(query)
        if self.gist_checkpoint_path is None:
            return {"kind": "error", "action": "solver-unavailable",
                    "response": "No gist-conditioned MIMO checkpoint is configured.",
                    "gist": gist, "facts": gist["facts"], "stages": gist["stages"],
                    "gist_used_for_computation": False}
        evidence_stage = next((stage for stage in gist["stages"]
                               if stage["name"] == "Evidence ranking"), None)
        mode_items = [] if not evidence_stage else [item["fact"] for item in evidence_stage["data"]
                                                    if item["fact"]["key"].casefold() == "mode"]
        if not mode_items or any(item["status"] == "conflicted" for item in mode_items):
            reason = "No single, non-conflicted mode fact was retrieved; solver was not run."
            return {"kind": "abstain", "action": "clarify-or-abstain", "response": reason,
                    "gist": gist, "facts": gist["facts"], "stages": gist["stages"] +
                    [{"name": "Gist validation", "status": "failed", "detail": reason}],
                    "gist_used_for_computation": False}
        mode = mode_items[0]["value"].casefold()
        mode_map = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}
        if mode not in mode_map:
            reason = f"Retrieved mode value {mode!r} is outside this checkpoint's trained task schema."
            return {"kind": "abstain", "action": "unsupported-gist", "response": reason,
                    "gist": gist, "facts": gist["facts"], "stages": gist["stages"] +
                    [{"name": "Gist validation", "status": "failed", "detail": reason}],
                    "gist_used_for_computation": False}
        try:
            from cortex_m.mimo.gist_conditioned import infer_gist, load_gist_model
            if self._gist_model is None:
                self._gist_model, _ = load_gist_model(self.gist_checkpoint_path, device=self._device)
            output = infer_gist(self._gist_model, request["values"], request["role_ids"],
                                mode_map[mode], device=self._device)
        except (ImportError, OSError, ValueError, TypeError, RuntimeError) as exc:
            return {"kind": "error", "action": "solver-error", "response": f"Gist-conditioned MIMO failed: {exc}",
                    "gist": gist, "facts": gist["facts"], "stages": gist["stages"] +
                    [{"name": "Gist-conditioned MIMO", "status": "failed", "detail": str(exc)}],
                    "gist_used_for_computation": False}
        source_fact = mode_items[0]
        detail = f"Retrieved mode={mode} from stored evidence and encoded it into router context."
        stages = gist["stages"] + [
            {"name": "Dynamic gist selection", "status": "done", "detail": detail,
             "data": {"fact": source_fact, "conditioning_vector": mode_map[mode]}},
            {"name": "Gist-conditioned MIMO inference", "status": "done",
             "detail": "The retrieved fact changed the MIMO router context used for edge scores.",
             "data": output},
        ]
        readable = "Gist-conditioned MIMO outputs: " + json.dumps(output["outputs"]) + "; selected physical input positions: " + json.dumps(output["selected_physical_positions"])
        return {"kind": "integrated_gist_mimo", "action": "mimo-inference", "response": readable,
                "solver_output": output, "gist": gist, "facts": gist["facts"], "stages": stages,
                "gist_used_for_computation": True,
                "integration_status": "retrieved fact encoded into MIMO router context"}

    def _run_support_gist(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dynamic demonstration-gist baseline; deliberately not the neural MIMO model."""
        try:
            import torch
            from cortex_m.mimo.sparse_router import SparseEvidenceRouter, route_labels
            support_x = torch.as_tensor(request["support_x"], dtype=torch.float32)
            support_y = torch.as_tensor(request["support_y"], dtype=torch.float32)
            query_x = torch.as_tensor(request["query_x"], dtype=torch.float32)
            result = SparseEvidenceRouter().route(support_x, support_y, query_x)
            labels = route_labels(support_x.shape[1], result.route_indices.tolist())
        except (ImportError, KeyError, ValueError, TypeError, RuntimeError) as exc:
            return {"kind": "error", "action": "invalid-support-gist",
                    "response": f"Support-example gist could not be routed: {exc}", "stages": [],
                    "gist_used_for_computation": False}
        data = {"prediction": result.prediction.tolist(), "route_labels": labels,
                "confidence_margin": result.confidence_margin.tolist(),
                "support_example_count": int(support_x.shape[0]),
                "gist_source": "variable-size input/output examples in the current request"}
        return {"kind": "support_gist_route", "action": "route-by-support-examples",
                "response": "Support examples selected a route from the configured relational library.",
                "solver_output": data, "stages": [{"name": "Dynamic support-example gist",
                    "status": "done", "detail": "Variable-size examples were evaluated against 168 explicit candidate relations.",
                    "data": data}], "gist_used_for_computation": True,
                "integration_status": "non-neural support-router comparator; operation library is finite"}

    def _run_memory_prompt(self, prompt: str) -> dict[str, Any]:
        if self._looks_like_question(prompt):
            return self.cortex.trace_query(prompt)
        trace = self.cortex.trace_observe(prompt)
        if not trace["accepted_count"]:
            trace.update({"kind": "unsupported", "action": "not-supported",
                          "response": ("This prototype does not support that request yet. "
                                       "It can store supported explicit facts and answer from "
                                       "saved evidence; it does not provide general chat or "
                                       "generate or execute code.")})
        return trace

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        return bool(re.search(
            r"^\s*(?:what|who|when|where|why|how|which|whose|whom|can|could|do|does|did|is|are|am|was|were|will|would|should|has|have|had)\b|\?\s*$",
            text, re.IGNORECASE,
        ))

    def trace_observe(self, text: str) -> dict[str, Any]:
        return self.cortex.trace_observe(text)

    def trace_query(self, text: str) -> dict[str, Any]:
        return self.cortex.trace_query(text)

    def observe(self, statement: str) -> int:
        return self.cortex.observe(statement)

    def close(self) -> None:
        self.cortex.close()
