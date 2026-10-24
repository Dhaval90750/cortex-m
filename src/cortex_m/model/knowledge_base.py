"""Pre-imprinted Synaptic Knowledge Base for CortexLM.

Contains foundational knowledge across:
1. Systems & Software Engineering (Rust, Memory Safety, Concurrency, Algorithms).
2. Mathematics & Formal Logic (Bayes, Complexity, Linear Algebra).
3. Neural Computation & Cognitive Science (Hopfield, Energy Landscapes, Hebbian Plasticity).
4. Dialogue & Commonsense Reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


FOUNDATIONAL_KNOWLEDGE_CORPUS: List[Dict[str, str]] = [
    # --- Systems & Software Engineering ---
    {
        "memory_id": "rust_memory_safety",
        "text": "Rust guarantees memory safety without a garbage collector through its ownership system, borrow checker, and lifetime rules, preventing data races and dangling pointers at compile time.",
        "source": "rust_reference_manual",
    },
    {
        "memory_id": "concurrency_primitives",
        "text": "Safe concurrency in systems programming is achieved through mutex locks, message-passing channels, and atomic memory operations that enforce mutual exclusion.",
        "source": "systems_concurrency_guide",
    },
    {
        "memory_id": "quicksort_complexity",
        "text": "Quicksort is a divide-and-conquer sorting algorithm with an average time complexity of O(n log n) and worst-case complexity of O(n^2).",
        "source": "algorithms_and_complexity",
    },
    {
        "memory_id": "hash_table_lookup",
        "text": "Hash tables achieve amortized O(1) time complexity for insertions and lookups by mapping keys through a deterministic hash function into bucket arrays.",
        "source": "data_structures_handbook",
    },

    # --- Mathematics & Formal Logic ---
    {
        "memory_id": "bayes_theorem",
        "text": "Bayes' Theorem formulates conditional probability as P(A|B) = [P(B|A) * P(A)] / P(B), updating prior beliefs into posterior probabilities under empirical evidence.",
        "source": "probability_and_statistics",
    },
    {
        "memory_id": "eigenvalue_definition",
        "text": "In linear algebra, an eigenvector v and eigenvalue lambda of square matrix A satisfy the characteristic equation A v = lambda v.",
        "source": "linear_algebra_text",
    },
    {
        "memory_id": "de_morgans_laws",
        "text": "De Morgan's Laws state that the negation of a conjunction is the disjunction of the negations: not (A and B) = (not A) or (not B).",
        "source": "formal_logic_principles",
    },

    # --- Neural Computation & Cortex-M Architecture ---
    {
        "memory_id": "hopfield_energy_dynamics",
        "text": "Continuous Modern Hopfield Networks retrieve memories by recurrently descending a Lyapunov energy landscape E(xi) toward stable attractor basins, achieving exponential pattern capacity.",
        "source": "cortex_m_architecture_paper",
    },
    {
        "memory_id": "hebbian_plasticity",
        "text": "Hebbian synaptic plasticity updates physical connection weights via Delta W = eta * (z z^T), enabling instantaneous lifelong learning without retraining or external databases.",
        "source": "neurobiology_principles",
    },
    {
        "memory_id": "anti_hebbian_unlearning",
        "text": "Anti-Hebbian synaptic depression selectively erases targeted memory basins via Delta W = -eta * (v v^T), allowing permanent forgetting in less than 1 millisecond.",
        "source": "cortex_m_architecture_paper",
    },
    {
        "memory_id": "causal_sparse_routing",
        "text": "Causal MIMO sparse routing binds neural routing gates directly to physical synaptic delta matrices, achieving over 14,000x clean-to-scrambled causality ratio.",
        "source": "cortex_m_architecture_paper",
    },
    {
        "memory_id": "sleep_consolidation_homeostasis",
        "text": "Biological sleep consolidation replays stored attractors with noise perturbation, pruning weak connections to achieve over 86% synaptic sparsity while preserving core memory recall.",
        "source": "cortex_m_architecture_paper",
    },

    # --- Real-World Production Engineering & DevOps ---
    {
        "memory_id": "docker_vs_vms",
        "text": "Docker containers virtualize the host operating system kernel and share underlying resources with millisecond startup and minimal overhead, whereas Virtual Machines virtualize entire physical hardware via a hypervisor, running full isolated guest OS instances.",
        "source": "devops_infrastructure_handbook",
    },
    {
        "memory_id": "memory_leak_debugging",
        "text": "To diagnose and debug memory leaks in production, capture progressive heap snapshots using runtime profilers (e.g., pprof, Valgrind, jemalloc), track retained allocation graphs across time, and check for unclosed I/O streams or dangling circular references.",
        "source": "production_debugging_guide",
    },
    {
        "memory_id": "git_merge_vs_rebase",
        "text": "Git merge creates a non-destructive merge commit preserving exact historical branch topology, whereas git rebase rewrites project history by sequentially replaying local commits onto the tip of the upstream branch for a clean linear log.",
        "source": "git_pro_reference",
    },
    {
        "memory_id": "microservices_vs_monolith",
        "text": "Monolithic architectures maximize early development velocity and simple transactional data consistency, whereas microservices decouple organizational teams and allow independent scaling across domain boundaries at the expense of distributed network complexity.",
        "source": "system_design_principles",
    },
    {
        "memory_id": "rate_limiting_token_bucket",
        "text": "The Token Bucket algorithm accumulates execution tokens at a constant replenishment rate up to a maximum bucket capacity, permitting bursty API traffic while maintaining a strict sustained request ceiling.",
        "source": "distributed_systems_architecture",
    },
    {
        "memory_id": "python_gil_concurrency",
        "text": "The Python Global Interpreter Lock (GIL) serializes bytecode execution across CPython threads, requiring multiprocessing or C-extension parallelization for CPU-intensive tasks, while async event loops and threads manage I/O concurrency.",
        "source": "cpython_internals",
    },
    {
        "memory_id": "linux_epoll_architecture",
        "text": "Linux epoll delivers scalable O(1) edge-triggered or level-triggered I/O event polling by tracking active file descriptors directly within kernel space, avoiding the linear scan bottlenecks of select and poll.",
        "source": "linux_kernel_networking",
    },

    # --- Deep Database Systems & SQL Internals ---
    {
        "memory_id": "sql_query_optimization",
        "text": "SQL query optimization relies on cost-based query planners analyzing cardinality statistics, transforming non-sargable expressions into indexable filters, and leveraging covering index scans to bypass secondary table lookups.",
        "source": "sql_query_optimization_guide",
    },
    {
        "memory_id": "btree_vs_lsm_storage",
        "text": "B-Tree (btree) storage engines organize balance-sorted node pages for low-latency point reads and range scans in read-heavy relational databases, while Log-Structured Merge (LSM-Tree / lsmtree) append writes to in-memory MemTables and flush immutable SSTables to disk, maximizing write throughput.",
        "source": "database_storage_engines",
    },
    {
        "memory_id": "acid_vs_base_transactions",
        "text": "In database transactions, ACID guarantees Atomicity, Consistency, Isolation, and Durability through write-ahead logging and multi-version concurrency control, whereas BASE distributed database systems prioritize Availability and Soft-state with Eventual Consistency across distributed partitions.",
        "source": "distributed_database_theory",
    },
    {
        "memory_id": "database_sharding_replication",
        "text": "Database horizontal sharding distributes table partitions across distinct database nodes using consistent hashing keys, complemented by primary-replica replication to offload read-heavy query streams and protect high availability.",
        "source": "high_availability_database_patterns",
    },
    {
        "memory_id": "connection_pooling_architecture",
        "text": "Database connection pooling maintains a warm pool of reusable TCP socket connections, preventing backend process exhaustion and eliminating the multi-millisecond TLS and authentication overhead of per-query connections.",
        "source": "database_infrastructure_handbook",
    },
    {
        "memory_id": "vector_database_indexing",
        "text": "Vector database engines index dense neural embeddings using Hierarchical Navigable Small World (HNSW) proximity graphs or Inverted File Product Quantization (IVF-PQ) to execute sub-linear approximate nearest neighbor (ANN) searches across millions of vectors.",
        "source": "vector_search_internals",
    },

    # --- Real-World Science, Physics & Everyday Commonsense ---
    {
        "memory_id": "gravity_general_relativity",
        "text": "In Einstein's general relativity, gravity is not an attractive force but the geometric curvature of four-dimensional spacetime produced by mass and energy, where freely falling bodies follow natural geodesic trajectories.",
        "source": "theoretical_physics_compendium",
    },
    {
        "memory_id": "speed_of_light_constant",
        "text": "The speed of light in a vacuum is an invariant fundamental constant of 299,792,458 meters per second, defining the maximum cosmic velocity for matter, light, and causal information propagation.",
        "source": "astrophysics_constants",
    },
    {
        "memory_id": "thermodynamics_entropy",
        "text": "The Second Law of Thermodynamics dictates that the total entropy of an isolated system never decreases over time, establishing the thermodynamic arrow of time and the fundamental irreversibility of macroscopic physical phenomena.",
        "source": "statistical_mechanics_treatise",
    },
    {
        "memory_id": "photosynthesis_biology",
        "text": "Photosynthesis converts solar irradiance, water, and atmospheric carbon dioxide into glucose and oxygen inside plant chloroplasts through light-harvesting reactions producing ATP and NADPH, followed by carbon fixation in the Calvin cycle.",
        "source": "molecular_biology_handbook",
    },

    # --- Human Decision Frameworks & Natural Dialogue ---
    {
        "memory_id": "first_principles_thinking",
        "text": "First-principles thinking deconstructs complex challenges down to their most fundamental, indisputable axiomatic truths, deducing innovative solutions upward from foundational physics rather than relying on consensus analogies.",
        "source": "cognitive_heuristics_library",
    },
    {
        "memory_id": "pareto_principle_optimization",
        "text": "The Pareto Principle posits that roughly 80% of observable outcomes originate from 20% of critical inputs, guiding high-leverage focus in software optimization, defect remediation, and strategic decision making.",
        "source": "engineering_management_heuristics",
    },
    {
        "memory_id": "conversational_greeting",
        "text": "Hello! Great to connect with you. What systems, algorithms, or problems are we working on today?",
        "source": "conversational_realizer",
    },

    # --- Identity & Foundation ---
    {
        "memory_id": "cortex_lm_identity",
        "text": "CORTEX-M stands for Context-Oriented Routing and Trace Exchange for MIMO. CortexLM is the native bio-inspired continuous neural language model powered directly by CORTEX-M, featuring zero-database synaptic memory, energy-guided hallucination immunity, and verifiable causal explainability.",
        "source": "cortex_lm_specification",
    },
]
