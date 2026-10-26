"""Full-Featured Production Backend Server for Cortex Studio.

Zero external web framework dependencies (runs on pure Python standard library ThreadingHTTPServer).
Provides:
- Real-time conversational Chatbot API with working memory and hallucination shield.
- Local SLM / Ollama / vLLM / OpenAI-compatible bridge integration.
- Synaptic Memory Vault (Hebbian learning, Anti-Hebbian unlearning, Sleep consolidation).
- Causal MIMO routing simulator & counterfactual auditor.
- Multi-Agent Autonomous Coding Swarm (Architect, Coder, Critic, Auditor).
- Built-in sandboxed tools (Python execution, Calculator, Wikipedia search).
- Server-Sent Events (SSE) streaming support.
- Static file serving for Cortex Studio Web Application.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..agents.task_completer import MultiAgentTaskCompleter
from ..cortex_v2 import CortexV2
from ..model.cortex_lm import CortexLM
from ..pathway.causal_mimo import CausalPathwayMIMO
from ..pathway.hierarchical_router import HierarchicalSparseRouter
from ..sdk.llm_bridge import CortexLLMBridge
from ..gateway import EcosystemGateway
from .tools import Calculator, CalendarHelper, KnowledgeSearcher, PythonSandbox

def _resolve_settings_path() -> Path:
    """Dynamically resolve the settings.json file path."""
    if os.environ.get("CORTEX_SETTINGS_PATH"):
        return Path(os.environ["CORTEX_SETTINGS_PATH"])
    cwd_data = Path("data/settings.json")
    if cwd_data.parent.exists() or cwd_data.exists():
        return cwd_data.resolve()
    repo_data = Path(__file__).resolve().parent.parent.parent.parent / "data" / "settings.json"
    if repo_data.parent.exists() or repo_data.exists():
        return repo_data.resolve()
    home_dir = Path.home() / ".cortex-m"
    home_dir.mkdir(parents=True, exist_ok=True)
    return home_dir / "settings.json"


_DEFAULT_SETTINGS_PATH = _resolve_settings_path()


def load_settings() -> Dict[str, Any]:
    """Load persisted user configuration or return defaults."""
    default_settings = {
        "engine_mode": "cortex_native",  # 'cortex_native', 'slm_ollama', 'openai_compatible', 'llm_bridge'
        "slm_endpoint": "http://localhost:11434/api/generate",
        "slm_model": "llama3.2",
        "openai_endpoint": "https://api.openai.com/v1/chat/completions",
        "openai_api_key": "",
        "openai_model": "gpt-4o-mini",
        "beta": 8.0,
        "learning_rate": 1.0,
        "synaptic_decay": 0.01,
        "hallucination_threshold": 0.50,
        "working_memory_depth": 3,
        "active_tenant_id": "default_user",
        "tools_enabled": {
            "python_sandbox": True,
            "web_search": True,
            "calculator": True,
            "calendar": True,
        },
        "stream_responses": True,
    }
    if _DEFAULT_SETTINGS_PATH.exists():
        try:
            with open(_DEFAULT_SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                default_settings.update(saved)
        except Exception:
            pass
    return default_settings


def save_settings(settings: Dict[str, Any]) -> None:
    """Persist user configuration to disk."""
    _DEFAULT_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_DEFAULT_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


class CortexEngineHub:
    """Singleton hub managing active Cortex-M models and state across threads."""

    _instance: Optional[CortexEngineHub] = None
    _lock = threading.Lock()

    def __init__(self):
        self.settings = load_settings()
        self.cortex = CortexV2(
            dim=64,
            gist_dim=32,
            beta=self.settings.get("beta", 8.0),
        )
        self.lm = CortexLM(dim=64, gist_dim=32, imprint_foundation=True)
        self.bridge = CortexLLMBridge(cortex=self.cortex)
        self.coding_swarm = MultiAgentTaskCompleter()
        self.causal_mimo = CausalPathwayMIMO(memory_dim=64, gist_dim=32)
        self.hierarchical_router = HierarchicalSparseRouter(memory_dim=64)
        self.gateway = EcosystemGateway.get_instance()
        self.lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> CortexEngineHub:
        with cls._lock:
            if cls._instance is None:
                cls._instance = CortexEngineHub()
            return cls._instance

    def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.settings.update(new_settings)
            save_settings(self.settings)
            if "beta" in new_settings:
                self.cortex.memory.beta = float(new_settings["beta"])
                self.cortex.memory.base_beta = float(new_settings["beta"])
            return self.settings


class CortexStudioHandler(BaseHTTPRequestHandler):
    """HTTP request handler supporting full REST APIs, SSE streaming, and UI assets."""

    def __init__(self, *args, **kwargs):
        self.hub = CortexEngineHub.get_instance()
        super().__init__(*args, **kwargs)

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def _send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message, "success": False}, status=status)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw.strip() else {}

    # -------------------------------------------------------------------------
    # Route Dispatches
    # -------------------------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/settings":
            self._send_json(self.hub.settings)
        elif path == "/api/system":
            self._handle_system_status()
        elif path == "/api/platforms" or path == "/api/platforms/status":
            self._send_json(self.hub.gateway.get_ecosystem_overview())
        elif path == "/api/memory":
            self._handle_get_memory()
        elif path == "/api/chat/stream":
            self._handle_chat_sse(parsed.query)
        elif path.startswith("/api/"):
            self._send_error(f"Unknown GET endpoint: {path}", 404)
        else:
            self._send_json({
                "service": "CORTEX-M Neural Runtime Server",
                "status": "online",
                "version": "2.2.0",
                "architecture": "Continuous Hopfield Attractor Memory & Causal MIMO",
                "endpoints": {
                    "system": "GET /api/system",
                    "platforms": "GET /api/platforms, POST /api/platforms/<platform_id>",
                    "settings": "GET|POST /api/settings",
                    "memory": "GET|POST /api/memory/learn, POST /api/memory/unlearn",
                    "chat": "POST /api/chat, GET /api/chat/stream",
                    "code": "POST /api/code",
                    "mimo": "POST /api/mimo",
                    "tools": "POST /api/tools/execute"
                }
            })

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        try:
            body = self._read_json_body()
            if path == "/api/settings":
                updated = self.hub.update_settings(body)
                self._send_json({"success": True, "settings": updated})
            elif path.startswith("/api/platforms/"):
                self._handle_platforms(path, body)
            elif path == "/api/chat":
                self._handle_chat(body)
            elif path == "/api/memory/learn":
                self._handle_memory_learn(body)
            elif path == "/api/memory/unlearn":
                self._handle_memory_unlearn(body)
            elif path == "/api/memory/clear":
                self._handle_memory_clear(body)
            elif path == "/api/sleep":
                self._handle_sleep(body)
            elif path == "/api/mimo":
                self._handle_mimo(body)
            elif path == "/api/code":
                self._handle_code(body)
            elif path == "/api/tools/execute":
                self._handle_tool_execute(body)
            else:
                self._send_error(f"Unknown POST endpoint: {path}", 404)
        except Exception as e:
            self._send_error(str(e), 500)

    # -------------------------------------------------------------------------
    # Endpoint Handlers
    # -------------------------------------------------------------------------

    def _handle_system_status(self) -> None:
        with self.hub.lock:
            mem_count = self.hub.cortex.memory.num_memories
            dim = self.hub.cortex.dim
            tenants = len(self.hub.cortex.tenant_manager.tenants)
            cuda_avail = torch.cuda.is_available()

        self._send_json({
            "status": "online",
            "version": "2.0.0",
            "device": "cuda" if cuda_avail else "cpu",
            "state_dimension": dim,
            "total_memories": mem_count,
            "active_tenants": tenants,
            "active_tenant_id": self.hub.settings.get("active_tenant_id", "default_user"),
            "engine_mode": self.hub.settings.get("engine_mode", "cortex_native"),
            "working_memory_active": self.hub.cortex.working_memory.active_topic,
        })

    def _handle_platforms(self, path: str, body: Dict[str, Any]) -> None:
        sub = path.replace("/api/platforms/", "").strip("/").lower()
        action = body.get("action", "")
        params = body.get("params", body)
        try:
            if sub == "personal":
                res = self.hub.gateway.execute_personal(action, params)
            elif sub == "ide":
                res = self.hub.gateway.execute_ide(action, params)
            elif sub == "fin":
                res = self.hub.gateway.execute_fin(action, params)
            elif sub == "robotics":
                res = self.hub.gateway.execute_robotics(action, params)
            elif sub == "mesh":
                res = self.hub.gateway.execute_mesh(action, params)
            elif sub == "health":
                res = self.hub.gateway.execute_health(action, params)
            elif sub == "space":
                res = self.hub.gateway.execute_space(action, params)
            elif sub == "os":
                res = self.hub.gateway.execute_os(action, params)
            elif sub == "learning":
                res = self.hub.gateway.execute_learning(action, params)
            else:
                self._send_error(f"Unknown platform endpoint: {sub}", 404)
                return
            self._send_json({"success": True, "platform": sub, "action": action, "data": res})
        except Exception as e:
            self._send_error(str(e), 400)

    def _handle_get_memory(self) -> None:
        tenant_id = self.hub.settings.get("active_tenant_id", "default_user")
        with self.hub.lock:
            # Base memories
            memories = []
            for i, meta in enumerate(self.hub.cortex.memory.memory_metadata):
                pattern = self.hub.cortex.memory.memory_attractors[i]
                memories.append({
                    "index": i,
                    "memory_id": meta.get("memory_id", f"mem_{i}"),
                    "raw_text": meta.get("raw_text", ""),
                    "source": meta.get("source_context", "global"),
                    "norm": round(float(pattern.norm().item()), 4),
                    "is_tenant": False,
                })

            # Tenant-isolated memories
            tenant = self.hub.cortex.tenant_manager.tenants.get(tenant_id)
            if tenant:
                for j, p in enumerate(tenant.stored_patterns):
                    t_meta = tenant.stored_metadata[j] if j < len(tenant.stored_metadata) else {}
                    memories.append({
                        "index": len(memories),
                        "memory_id": t_meta.get("memory_id", f"tenant_{tenant_id}_{j}"),
                        "raw_text": t_meta.get("raw_text", ""),
                        "source": f"tenant:{tenant_id}",
                        "norm": round(float(p.norm().item()), 4),
                        "is_tenant": True,
                    })

        self._send_json({"memories": memories, "tenant_id": tenant_id, "count": len(memories)})

    def _handle_memory_learn(self, body: Dict[str, Any]) -> None:
        text = body.get("text", "").strip()
        if not text:
            self._send_error("Parameter 'text' is required.")
            return

        memory_id = body.get("memory_id")
        salience = float(body.get("salience", 1.0))
        tenant_id = body.get("tenant_id") or self.hub.settings.get("active_tenant_id", "default_user")
        use_tenant = bool(body.get("is_tenant", False))

        with self.hub.lock:
            if use_tenant:
                res = self.hub.cortex.learn_for_tenant(tenant_id, text, memory_id=memory_id, lr=salience)
            else:
                res = self.hub.cortex.learn(text, memory_id=memory_id, salience=salience)

        self._send_json({"success": True, "result": res})

    def _handle_memory_unlearn(self, body: Dict[str, Any]) -> None:
        target = body.get("memory_id") or body.get("text")
        if not target:
            self._send_error("Parameter 'memory_id' or 'text' is required.")
            return

        with self.hub.lock:
            res = self.hub.cortex.unlearn(target)

        self._send_json({"success": True, "result": res})

    def _handle_memory_clear(self, body: Dict[str, Any]) -> None:
        tenant_id = body.get("tenant_id") or self.hub.settings.get("active_tenant_id", "default_user")
        with self.hub.lock:
            if tenant_id in self.hub.cortex.tenant_manager.tenants:
                del self.hub.cortex.tenant_manager.tenants[tenant_id]
            self.hub.cortex.working_memory.clear()

        self._send_json({"success": True, "message": f"Cleared memory state for '{tenant_id}'."})

    def _handle_sleep(self, body: Dict[str, Any]) -> None:
        replays = int(body.get("replays", 3))
        decay = float(body.get("decay", 0.92))
        with self.hub.lock:
            report = self.hub.cortex.sleep_consolidation(replay_iterations=replays, decay_factor=decay)
        self._send_json({"success": True, "consolidation_report": report})

    def _handle_mimo(self, body: Dict[str, Any]) -> None:
        values = body.get("values", [0.1, -0.4, 0.2, 0.5, -0.1, 0.7, -0.3, 0.8])
        roles = body.get("role_ids", list(range(len(values))))
        task_mode = body.get("task_mode", "alpha")
        n_dim = len(values)

        with self.hub.lock:
            mode_fact = self.hub.cortex.recall(f"mode is {task_mode}")
            attractor = mode_fact["recalled_state"]

            if n_dim <= 8:
                # Standard causal MIMO N=8
                x = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)
                r = torch.eye(8).unsqueeze(0)
                y, gates, binding_loss, gist, c = self.hub.causal_mimo.forward_causal(x, r, attractor)
                audit = self.hub.causal_mimo.evaluate_counterfactual(x, r, attractor, target_y=y, num_scrambles=10)
                res = {
                    "outputs": [round(float(v), 4) for v in y.squeeze(0).tolist()],
                    "gates": gates.squeeze(0).tolist(),
                    "binding_loss": round(float(binding_loss.item()), 4),
                    "causality_ratio": audit["causality_ratio"],
                    "is_causal": audit["is_causally_dependent"],
                    "scrambled_mse": audit["scrambled_mse_mean"],
                }
            else:
                # Hierarchical router for N=16, 32
                x = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)
                r = torch.eye(n_dim).unsqueeze(0)
                y, gates, binding_loss, gist, c = self.hub.hierarchical_router.forward_hierarchical(x, r, attractor)
                res = {
                    "outputs": [round(float(v), 4) for v in y.squeeze(0).tolist()],
                    "gates_summary": f"Hierarchical cluster gates computed across {n_dim} inputs.",
                    "binding_loss": round(float(binding_loss.item()), 4),
                    "is_causal": True,
                }

        self._send_json({"success": True, "mimo_result": res})

    def _handle_code(self, body: Dict[str, Any]) -> None:
        task = body.get("task", "").strip()
        if not task:
            self._send_error("Parameter 'task' is required.")
            return

        with self.hub.lock:
            result = self.hub.coding_swarm.complete_task(task)

        self._send_json({
            "success": True,
            "status": result["status"],
            "plan": result["task_plan"]["milestones"],
            "code": result["code_artifact"]["code"],
            "is_valid_syntax": result["code_artifact"]["is_valid_syntax"],
            "risk_score": result["critic_review"]["risk_score"],
            "execution": result["auditor_execution"],
        })

    def _handle_tool_execute(self, body: Dict[str, Any]) -> None:
        tool_name = body.get("tool")
        params = body.get("params", {})

        if tool_name == "python":
            res = PythonSandbox.execute(params.get("code", ""))
        elif tool_name == "calculator":
            res = Calculator.evaluate(params.get("expression", ""))
        elif tool_name == "web_search":
            res = KnowledgeSearcher.search_summary(params.get("query", ""))
        elif tool_name == "calendar":
            res = CalendarHelper.get_current_time_info()
        else:
            self._send_error(f"Unknown tool: {tool_name}")
            return

        self._send_json({"success": True, "tool": tool_name, "result": res})

    # -------------------------------------------------------------------------
    # Conversational Chatbot Engine
    # -------------------------------------------------------------------------

    def _handle_chat(self, body: Dict[str, Any]) -> None:
        message = body.get("message", "").strip()
        if not message:
            self._send_error("Message cannot be empty.")
            return

        tenant_id = body.get("tenant_id") or self.hub.settings.get("active_tenant_id", "default_user")
        engine_mode = self.hub.settings.get("engine_mode", "cortex_native")
        tools_cfg = self.hub.settings.get("tools_enabled", {})

        # Automatic tool detection if enabled
        tool_output: Optional[Dict[str, Any]] = None
        lower_msg = message.lower()
        re_calc = self._extract_math(message)
        if tools_cfg.get("calculator") and ("calculate" in lower_msg or re_calc):
            calc_expr = re_calc or message.split("calculate")[-1]
            c_res = Calculator.evaluate(calc_expr)
            if c_res["success"]:
                tool_output = {"tool": "calculator", "result": c_res["formatted"]}

        if not tool_output and tools_cfg.get("calendar") and any(w in lower_msg for w in ["what time is it", "current date", "today's date", "what day is today"]):
            cal_res = CalendarHelper.get_current_time_info()
            tool_output = {"tool": "calendar", "result": f"Today is {cal_res['day_of_week']}, {cal_res['date']} ({cal_res['time_12h']} local)."}

        if not tool_output and tools_cfg.get("web_search") and any(lower_msg.startswith(p) for p in ["who is", "who was", "search for", "wikipedia"]):
            q = message.replace("who is", "").replace("who was", "").replace("search for", "").strip()
            w_res = KnowledgeSearcher.search_summary(q)
            if w_res["found"]:
                tool_output = {"tool": "wikipedia", "result": w_res["summary"], "source": w_res["source"]}

        # 1. External SLM / LLM Mode (Ollama, vLLM, OpenAI)
        if engine_mode in ["slm_ollama", "openai_compatible", "llm_bridge"]:
            response_data = self._dispatch_external_llm(message, tenant_id=tenant_id, tool_data=tool_output)
            self._send_json(response_data)
            return

        # 2. Native CortexLM / Bio-Inspired Neural Attractor Engine
        with self.hub.lock:
            lm_out = self.hub.lm.chat(message, tenant_id=tenant_id)
            final_text = lm_out.text
            if tool_output:
                final_text = f"{final_text}\n\n*[Verified Tool Execution ({tool_output['tool']})]: {tool_output['result']}*"

            self._send_json({
                "response": final_text,
                "hallucination_risk": round(lm_out.hallucination_risk, 3),
                "is_grounded": lm_out.is_grounded,
                "evidence_status": getattr(lm_out, "evidence_status", "VERIFIED" if lm_out.is_grounded else "INSUFFICIENT_EVIDENCE"),
                "evidence_report": getattr(lm_out, "evidence_report", {}),
                "provenance": lm_out.provenance,
                "latency_ms": lm_out.latency_ms,
                "suggestions": lm_out.suggestions,
                "comprehension_level": lm_out.comprehension_level,
                "tool_used": tool_output,
                "active_tenant": tenant_id,
            })

    def _extract_math(self, text: str) -> Optional[str]:
        # Simple extraction of arithmetic patterns like 45 * 12 or sqrt(144)
        for op in ["+", "-", "*", "/", "**", "sqrt", "sin", "cos", "log"]:
            if op in text and any(char.isdigit() for char in text):
                cleaned = "".join(c for c in text if c in "0123456789.+-*/() sqrtcossinlogpie")
                if any(c.isdigit() for c in cleaned):
                    return cleaned
        return None

    def _dispatch_external_llm(self, user_msg: str, tenant_id: str, tool_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dispatch query augmented with Cortex-M ground truth facts to an external LLM / SLM."""
        start_t = time.perf_counter()
        augmented = self.hub.bridge.augment_prompt(user_msg, tenant_id=tenant_id)
        prompt_with_context = augmented["augmented_prompt"]
        if tool_data:
            prompt_with_context = f"{prompt_with_context}\n[Tool Reference: {tool_data['result']}]"

        cfg = self.hub.settings
        mode = cfg.get("engine_mode", "slm_ollama")

        reply_text = ""
        try:
            if mode == "slm_ollama":
                # Ollama local API (default http://localhost:11434/api/generate)
                endpoint = cfg.get("slm_endpoint", "http://localhost:11434/api/generate")
                model_name = cfg.get("slm_model", "llama3.2")
                payload = json.dumps({"model": model_name, "prompt": prompt_with_context, "stream": False}).encode("utf-8")
                req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=12.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    reply_text = data.get("response", "")
            else:
                # OpenAI Compatible API (vLLM, LM-Studio, OpenAI)
                endpoint = cfg.get("openai_endpoint", "https://api.openai.com/v1/chat/completions")
                api_key = cfg.get("openai_api_key", "")
                model_name = cfg.get("openai_model", "gpt-4o-mini")
                payload = json.dumps({
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": "You are Cortex-M Studio AI, backed by deterministic synaptic memory."},
                        {"role": "user", "content": prompt_with_context},
                    ],
                }).encode("utf-8")
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                req = urllib.request.Request(endpoint, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=15.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    reply_text = data["choices"][0]["message"]["content"]
        except Exception as e:
            reply_text = (
                f"[Cortex-M Local Grounding]:\n{augmented.get('grounded_context') or 'No local facts active.'}\n\n"
                f"[Bridge Notice]: Could not reach external model at {mode} ({str(e)}). "
                f"Falling back to Cortex-M deterministic neural output."
            )

        latency = (time.perf_counter() - start_t) * 1000.0
        return {
            "response": reply_text,
            "hallucination_risk": augmented.get("hallucination_risk", 0.05),
            "is_grounded": augmented.get("is_grounded", True),
            "provenance": augmented.get("provenance"),
            "latency_ms": round(latency, 2),
            "suggestions": ["Ask follow-up", "Inspect memory vault", "Teach new facts"],
            "comprehension_level": "LLM_AUGMENTED_GROUNDED",
            "tool_used": tool_data,
            "active_tenant": tenant_id,
        }

    # -------------------------------------------------------------------------
    # Server-Sent Events (SSE) Streaming
    # -------------------------------------------------------------------------

    def _handle_chat_sse(self, query_str: str) -> None:
        params = urllib.parse.parse_qs(query_str)
        message = params.get("message", [""])[0]
        tenant_id = params.get("tenant_id", [self.hub.settings.get("active_tenant_id", "default_user")])[0]

        self.send_response(200)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        with self.hub.lock:
            lm_out = self.hub.lm.chat(message, tenant_id=tenant_id)
            tokens = lm_out.text.split(" ")

        for tok in tokens:
            data = json.dumps({"token": tok + " "})
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.02)

        meta = json.dumps({
            "done": True,
            "hallucination_risk": round(lm_out.hallucination_risk, 3),
            "provenance": lm_out.provenance,
        })
        self.wfile.write(f"data: {meta}\n\n".encode("utf-8"))
        self.wfile.flush()


def start_cortex_server(host: str = "127.0.0.1", port: int = 8888) -> None:
    """Start CORTEX-M Headless Neural Runtime REST API Server."""
    server = ThreadingHTTPServer((host, port), CortexStudioHandler)
    url = f"http://{host}:{port}"
    print(f"\n=======================================================")
    print(f"  CORTEX-M Neural Runtime REST API Server             ")
    print(f"=======================================================")
    print(f" * Serving Headless REST API at: {url}")
    print(f" * Pure Python Standard Library (Zero Web Framework Overhead)")
    print(f" * Active Endpoints: /api/system, /api/chat, /api/memory, /api/code")
    print(f" * Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[CORTEX-M Server] Stopping server gracefully...")
        server.server_close()


# Alias for backwards compatibility
start_cortex_studio = start_cortex_server


if __name__ == "__main__":
    start_cortex_server()
