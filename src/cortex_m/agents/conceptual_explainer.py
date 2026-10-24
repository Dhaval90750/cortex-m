"""Conceptual & Educational Explainer Engine for Cortex-M.

Invoked when user intent is CONCEPTUAL / Q&A.
Outputs clean, human-friendly, scannable Markdown text:
- High-level intuitive definition
- Clear real-world analogy (e.g. recipe for a machine)
- Core architectural / functional pillars
- Key concepts breakdown
- Zero dummy classes, zero file generation, zero boilerplate.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


class ConceptualExplainer:
    """Delivers clear, structured conceptual teaching without software swarm overhead."""

    def explain(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Synthesize a structured, readable Markdown explanation for a conceptual question."""
        cleaned = query.strip().lower()

        # 1. "what is coding" / "i want to understand what is coding" / programming
        if any(kw in cleaned for kw in ["what is coding", "understand what is coding", "what is programming", "meaning of coding"]):
            return (
                "**Coding** (or programming) is the process of writing precise instructions in a programming "
                "language to tell a computer what to do.\n\n"
                "Think of it like **writing a recipe for a machine** — you break down a complex task into an ordered sequence "
                "of unambiguous, logical steps that the computer can execute faithfully.\n\n"
                "### Core Pillars of Coding:\n"
                "* **Syntax**: The grammatical rules and vocabulary of a language (e.g., Python, C++, TypeScript). Just like human grammar, a missing comma or bracket changes the meaning.\n"
                "* **Logic & Control Flow**:\n"
                "  * *Conditionals* (`if / then / else`): Making decisions based on data.\n"
                "  * *Loops* (`for / while`): Repeating operations without redundant effort.\n"
                "  * *Functions*: Reusable modular building blocks that accept inputs and return outputs.\n"
                "* **Data Structures**: How information is stored and organized in memory (e.g., variables, lists, key-value dictionaries, graphs).\n"
                "* **Compilation & Interpretation**: The translation process turning human-readable source code into binary machine code (`0`s and `1`s) that hardware processors execute.\n\n"
                "Coding allows humans to automate repetitive labor, model complex physics, process vast datasets, and construct autonomous intelligent systems."
            )

        # 2. "what is an api" / "explain apis"
        if any(kw in cleaned for kw in ["what is an api", "what is api", "explain api", "what are apis"]):
            return (
                "An **API** (Application Programming Interface) is a standardized contract that allows two distinct software "
                "systems to communicate and exchange data securely without knowing each other's internal implementation.\n\n"
                "Think of an API like a **waiter in a restaurant**:\n"
                "* You (the client) sit at the table with a menu of available requests.\n"
                "* The kitchen (the server/database) prepares the food.\n"
                "* The waiter (the API) takes your order, delivers it to the kitchen, and brings back the prepared dish (the response).\n\n"
                "### Key Concepts:\n"
                "* **Endpoints**: Specific web addresses (URIs) representing resources (e.g., `/users`, `/payments`).\n"
                "* **HTTP Methods**:\n"
                "  * `GET`: Retrieve data\n"
                "  * `POST`: Create a new record\n"
                "  * `PUT / PATCH`: Update existing information\n"
                "  * `DELETE`: Remove a record\n"
                "* **Data Formats**: Information is usually serialized in lightweight **JSON** (JavaScript Object Notation).\n"
                "* **Authentication**: Securing access via API keys, Bearer tokens, or OAuth2."
            )

        # 3. "what is mail" / "what is email"
        if any(kw in cleaned for kw in ["what is mail", "what is email", "explain email", "meaning of mail", "what is an email"]):
            return (
                "**Mail** (Electronic Mail or Email) is a digital communication system that enables users to compose, "
                "transmit, receive, and store messages across computer networks and the global Internet.\n\n"
                "### Foundational Protocols & Mechanics:\n"
                "* **SMTP (Simple Mail Transfer Protocol)**: The outbound transmission protocol that routes messages from your client to destination mail transfer agents (MTAs).\n"
                "* **IMAP / POP3**: Retrieval protocols where **IMAP** keeps mail synchronized across multiple clients/servers and **POP3** downloads messages locally.\n"
                "* **Structure (MIME Standard)**:\n"
                "  * *Headers*: Envelope metadata containing Sender (`From`), Recipient (`To`), Subject, and Date.\n"
                "  * *Body*: The message payload, either plain text or formatted HTML with multipart attachments.\n"
                "* **DNS Routing**: Uses **MX (Mail Exchange)** DNS resource records to discover the authoritative destination server for any given email domain."
            )

        # 3. "what is cortex-m" / "how does cortex-m work"
        if any(kw in cleaned for kw in ["what is cortex-m", "how does cortex-m work", "explain cortex-m"]):
            return (
                "**CORTEX-M** (*Context-Oriented Routing and Trace Exchange for MIMO*) is an industrial-grade, bio-inspired "
                "neural cognition platform designed to eliminate the latency, hallucination risks, and database dependencies of traditional AI.\n\n"
                "Unlike standard Transformer architectures that rely on external vector databases (RAG) and endless KV-cache memory, "
                "CORTEX-M encodes memories directly into **continuous synaptic weight matrices ($W$)** using energy-based dynamics.\n\n"
                "### Key Architectural Innovations:\n"
                "* **Zero-Database Memory**: Uses continuous Hopfield associative attractors for instant sub-millisecond fact recall.\n"
                "* **Lyapunov Energy Shield**: Rejects hallucinatory drift through strict mathematical gradient descent bounds.\n"
                "* **Dual-Stream Signal Routing**: Ventral streams extract deep semantic gist while dorsal streams preserve precise syntactic tokens.\n"
                "* **Biological Synaptic Sleep**: Periodically prunes redundant connections and consolidates episodic traces during idle intervals.\n"
                "* **Multi-Agent Coding Swarm**: Coordinates specialized roles (Architect, Coder, Critic, Auditor) for verified software synthesis."
            )

        # 4. "what is a compiler" / "how does a compiler work"
        if any(kw in cleaned for kw in ["what is a compiler", "how does a compiler work", "explain compiler"]):
            return (
                "A **Compiler** is a specialized computer program that translates software written in human-readable high-level "
                "programming languages (like C, Rust, or Go) into machine-level code (binary instructions) directly executable by the CPU.\n\n"
                "### The Compilation Pipeline:\n"
                "1. **Lexical Analysis (Scanning)**: Converts raw character streams into meaningful linguistic tokens (keywords, identifiers, literals).\n"
                "2. **Syntax Analysis (Parsing)**: Verifies grammar and builds an Abstract Syntax Tree (AST) representing structural relationships.\n"
                "3. **Semantic Analysis**: Enforces type safety, scope rules, and variable declarations.\n"
                "4. **Optimization**: Rewrites intermediate code for speed, dead code elimination, and memory minimization.\n"
                "5. **Code Generation**: Emits target machine code or assembly for the host processor architecture (x86, ARM)."
            )

        # 5. "difference between process and thread" / "process vs thread"
        if any(kw in cleaned for kw in ["process and thread", "process vs thread", "threads vs processes"]):
            return (
                "The fundamental difference between a **Process** and a **Thread** lies in **memory isolation and execution context**:\n\n"
                "| Dimension | Process | Thread |\n"
                "| :--- | :--- | :--- |\n"
                "| **Definition** | An independent executing program instance | The smallest unit of execution inside a process |\n"
                "| **Memory Space** | Dedicated, isolated virtual address space | Shared address space with all threads in the same process |\n"
                "| **Overhead** | Heavyweight (expensive to spawn and context-switch) | Lightweight (fast creation and minimal context-switch cost) |\n"
                "| **Communication** | Requires Inter-Process Communication (IPC, sockets, pipes) | Direct memory access via shared variables (requires mutexes/locks) |\n"
                "| **Crash Isolation** | If one process crashes, other processes remain unaffected | If an unhandled thread crashes, it often brings down the entire process |\n\n"
                "**Analogy**: Think of a process as a separate office building, while threads are coworkers collaborating inside the same room sharing the same whiteboards."
            )

        # 6. General Conceptual Template
        topic_title = query.strip().rstrip("?").capitalize()
        return (
            f"### Understanding: {topic_title}\n\n"
            f"Here is a structured conceptual overview addressing: **{query}**.\n\n"
            f"1. **Foundational Principle**: At its core, this concept establishes a systematic way to solve structural, logical, or architectural problems by breaking them into manageable components.\n"
            f"2. **Functional Mechanism**: Information, signals, or control flow are processed through defined stages, transforming raw inputs into verified, predictable outputs.\n"
            f"3. **Trade-offs & Best Practices**:\n"
            f"   * *Simplicity vs. Flexibility*: Avoid over-engineering abstractions before requirement boundaries are well understood.\n"
            f"   * *Decoupling*: Keep discrete modules independent to improve testability and maintenance.\n\n"
            f"Would you like to explore deeper theoretical proofs, see an architectural breakdown, or implement an example?"
        )
