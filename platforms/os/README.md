# CORTEX-OS: Cognitive Userspace Environment (Specification)

## Overview
**CORTEX-OS** is an architectural research specification for a userspace cognitive operating environment. Rather than relying solely on static folder hierarchies, CORTEX-OS is designed to dynamically organize files, documents, and notes by **semantic associative affinity clusters** while providing local associative memory retrieval.

---

## Planned Capabilities
- **Associative Omnibar Search**: Design specification for spotlight search finding local documents by conceptual meaning rather than exact keywords.
- **Cognitive Affinity Clusters**: Dynamic groupings of active projects based on working memory state.
- **Zero-Footprint Runtime**: Dormant low-rank factor footprint ($<2.5\text{ MB}$) that wakes on demand without degrading OS responsiveness.
- **Userspace Sandboxing**: Operates strictly as a non-privileged userspace daemon on Linux, macOS, and Windows.

---

## 🎨 Visual Identity & Chromatic Environment: Brutalist Graphite & Signal Orange
Governed by the [Ecosystem Design System](../../docs/architecture/design_system.md#6-cortex-os-cognitive-operating-environment--brutalist-graphite--signal-orange):
* **Carbon** (`#131313`): Deep structural canvas (70%).
* **Graphite** (`#292B29`): Window frames and dock separators (20%).
* **Concrete** (`#777B73`): Inactive file boundaries and structural metadata.
* **Signal Orange** (`#E56B46`): **Signature Accent** — Omnibar focus and high-priority attention (10%).
* **Acid Lime** (`#C5D75C`): Dynamic process and live affinity cluster markers.

