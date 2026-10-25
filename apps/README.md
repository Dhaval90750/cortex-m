# CORTEX-M Application Entry Points

This directory houses the programmatic and command-line entry points for CORTEX-M. All applications interface directly with the underlying [`cortex_m.core.runtime`](../src/cortex_m/core/runtime.py) without relying on any web browser or HTML interface.

---

## 1. `apps/cli/` — Universal Terminal CLI
Universal terminal command-line interface for interactive conversation, on-the-go learning, memory inspection, and benchmark execution.
```bash
# Interactive conversational interface (learns and interacts on the go)
python -m apps.cli.main chat

# Associative memory inspection (Continuous Hopfield attractor state)
python -m apps.cli.main memory status
```

## 2. `apps/api/` — Headless Neural REST API Server
High-throughput headless HTTP / REST API server for programmatic integration, IDE extensions, daemon services, and microservice pipelines.
```bash
python -m apps.api.server
```
Active Endpoints:
- `GET /api/system` — System status, memory dimension, active tenants
- `GET|POST /api/settings` — Runtime configuration & profile management
- `GET|POST /api/memory` — Associative pattern recall & Hebbian imprinting
- `POST /api/chat` — Calibrated conversational cognition & evidence reports
- `POST /api/code` — MultiAgentTaskCompleter autonomous software engineering
- `POST /api/mimo` — Causal MIMO sparse routing execution
- `POST /api/tools/execute` — Sandboxed tool execution
