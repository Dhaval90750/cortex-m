# CORTEX-M Application Entry Points

This directory houses the user-facing application entry points for CORTEX-M. All applications share the same underlying [`cortex_m.core.runtime`](../src/cortex_m/core/runtime.py) without duplicating application logic.

---

## 1. `apps/studio/` — Cortex Studio Web UI
Launches the browser-based neural cognition workbench, synaptic memory visualizer, and chat interface.
```bash
python -m apps.studio.run
```

## 2. `apps/cli/` — Unified CLI
Universal terminal command-line interface for chat, code generation, memory inspection, and benchmark execution.
```bash
python -m apps.cli.main chat
python -m apps.cli.main code "Add unit tests"
python -m apps.cli.main memory search "project plan"
```

## 3. `apps/api/` — Headless API Server
Headless HTTP / REST API server for external microservice integration, IDE plugins, and third-party tools.
```bash
python -m apps.api.server
```
