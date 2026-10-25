# CORTEX-Personal (Architectural Specification)

## Overview
**CORTEX-Personal** is an offline, privacy-first personal intelligence specification for desktop and mobile edge devices. Backed by CORTEX-Synapse continuous attractor memory, it retains user preferences, project notes, and episodic facts in physical synaptic connection weights without transmitting data to external cloud servers or relying on third-party vector databases.

---

## Key Capabilities (Planned)
- **Continuous Hopfield Recall**: Sub-millisecond associative recall ($<500\ \mu\text{s}$) over personal memories.
- **Epistemic Evidence Scoring**: Calibrated verification reports (`VERIFIED`, `PARTIALLY_SUPPORTED`, `INSUFFICIENT_EVIDENCE`) informing the user of ground truth.
- **Tenant & Device Paging**: 2 KB low-rank factor isolation ensuring private personal memories never leak across profiles.
- **Zero Cloud Egress**: Strict offline local runtime execution on user hardware.

---

## 🎨 Visual Identity & Chromatic Environment: Avocado, Seed & Terracotta
Governed by the [Ecosystem Design System](../../docs/architecture/design_system.md#4-cortex-personal-edge-intelligence--avocado-seed--terracotta):
* **Cream** (`#F6F0DF`): Warm, calm daylight neutral foundation (70%).
* **Avocado Flesh** (`#DDE5B6`): Contextual cards and soft containers (20%).
* **Olive Leaf** (`#87985A`): **Signature Accent** — Personal identity and primary controls (10%).
* **Deep Olive** (`#4B603B`): High-legibility text and structural icons.
* **Seed Brown** (`#8B5B40`): Terracotta warmth for episodic memory tags.

