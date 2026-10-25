# Security Policy

## 1. Supported Versions

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 2.x     | :white_check_mark: | Active production release |
| 1.x     | :x:                | Legacy research substrate |

---

## 2. Reporting a Vulnerability

The CORTEX-M project takes security seriously, particularly around:
- **Untrusted Code Execution**: Escape from the 5.0-second Python sandbox (`src/cortex_m/agents/patch_sandbox.py`).
- **Prompt & Data Injection**: Indirect prompt injection via ingested web documents, PDFs, or external model responses.
- **Cross-Tenant Memory Leakage**: Unauthorized access across tenant low-rank memory factors (`src/cortex_m/pathway/tenant_paging.py`).
- **Policy Engine Bypass**: Circumvention of the human approval gate for file modifications or Git operations.

### How to Report
Please do **not** report security vulnerabilities through public GitHub issues.

Instead, please send a responsible disclosure email describing the vulnerability, proof of concept, and affected versions to the repository maintainer through GitHub Security Advisories or direct private contact.

We acknowledge receipt of reports within 48 hours and aim to provide a resolution or patch timeline within 7 days.

---

## 3. Sandboxing & Tool Safety Notice
CortexCode and Python tools execute generated code inside an isolated subprocess with strict CPU, memory, and timeout constraints (default: 5.0s). While this prevents accidental runaway processes, running untrusted code on production workstations without container or VM boundaries is not recommended for untrusted multi-user environments.
