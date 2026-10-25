# CORTEX-IDE / CortexCode


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

---

## 🎨 Visual Identity & Chromatic Environment: Forest Graphite & Fern
Governed by the [Ecosystem Design System](../../docs/architecture/design_system.md#3-cortex-ide-cortexcode-engineering-workspace--forest-graphite--fern):
* **Editor Ink** (`#171A17`): Deep non-glare editor background (70%).
* **Forest Graphite** (`#29332C`): File trees, diff panels, and borders (20%).
* **Fern** (`#8EBF76`): **Signature Accent** — AST symbols, diff additions, and active cursor (10%).
* **Debug Brass** (`#E7AF63`): Breakpoints, test warnings, and auditor gates.
* **Linen** (`#F0F1E8`): High-contrast monospace code text.

