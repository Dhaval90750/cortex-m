# Changelog

All notable changes to the CORTEX-M architecture and runtime will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.0] - 2026-10-25

### Added
- **Ecosystem UI/UX Gallery**: Integrated high-resolution Figma-grade visual mockups across 10 platform surfaces (`References/` and `assets/ui_previews/`).
- **Interactive Ecosystem Showcase in Studio**: Added dedicated `🌐 Industrial Ecosystem` workbench tab in Cortex Studio with live telemetry specs.
- **Typed Core Architecture**: Formal abstract contracts (`ModelGateway`, `MemoryProvider`, `ToolRegistry`, `PolicyEngine`, `TaskOrchestrator`).
- **Deployment Profiles**: Added `developer_workstation`, `local_research`, and `edge` configuration presets in `configs/`.
- **Open Source Governance Suite**: Added `.github/workflows/ci.yml`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, and issue/PR templates.

---

## [2.0.0] - 2026-10-24

### Added
- **Calibrated Epistemic Evidence Verifier**: Replaced raw Lyapunov energy assertions with an evidence status pipeline (`VERIFIED`, `PARTIALLY_SUPPORTED`, `CONFLICTING_EVIDENCE`, `INSUFFICIENT_EVIDENCE`).
- **Comparative Empirical Benchmark Suite**: Added `benchmarks/run_comparison_benchmark.py` evaluating Continuous Hopfield vs. BM25 vs. Dense Cosine Vector Similarity.
- **CortexCode Flagship Hardening**:
  - `GitWorkspaceContext`: Live Git branch inspection, uncommitted changes, and AST symbol extraction.
  - `PatchSandboxRunner`: Unified diff generator and isolated 5.0-second subprocess test execution runner.
  - Interactive Human Approval Gate for consequential file modifications.
- **Universal Node Runner**: Added `bin/cortex-m.js` and `package.json` for one-command execution via `npx cortex-m`.

---

## [1.0.0] - 2026-10-20

### Added
- Initial CORTEX-M research release: Continuous Hopfield synaptic memory substrate, Hebbian potentiation, and anti-Hebbian unlearning.
- Causal MIMO Sparse Routing across $N=8, 16, 32, 1024$ dimensions.
- 128-D Factored Gist bottleneck with orthogonal functional heads.
- Pure Python standard library HTTP server and Cortex Studio visualizer.
