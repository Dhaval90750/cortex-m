# CORTEX-IDE / CortexCode

<p align="center">
  <img src="../../assets/ui_previews/cortex_code_ide.png" alt="CORTEX-IDE Dark Workspace" width="100%" />
</p>

## Overview
**CORTEX-IDE** (powered by **CortexCode**) is an autonomous, repository-aware software engineering workspace. Unlike naive LLM code assistants that overwrite arbitrary files or output untested code blocks, CortexCode inspects Git branch state, parses AST symbols, produces formal unified diff patches, executes tests inside a 5.0-second sandbox, and halts for explicit human approval before any commit or push.

---

## Key Capabilities
- **AST Symbol Indexing**: Real-time extraction of classes, functions, and call hierarchies directly from workspace source files.
- **Unified Diff Reviewer**: Visual side-by-side patch inspection highlighting deletions and additions with syntax highlighting.
- **5.0s Sandboxed Execution**: Automated subprocess runner executing unit tests and syntax checks with hard real-time timeouts.
- **Human Approval Gate**: Cryptographically bound tokens for consequential file edits, branch creations, and git pushes.

---

## Safety & Qualification Boundary
* **Execution Boundary**: Subprocess sandboxing with isolated environment variables, restricted network egress, and memory ceiling.
* **Deterministic Interlock**: Patch generation does not automatically write to disk until human approval is registered (`[y/N]`).
