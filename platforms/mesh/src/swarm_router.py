"""Causal Sparse Swarm Router & Task Allocation for CORTEX-Mesh.

Ranks and assigns cooperative civilian swarm tasks using proximity, battery reserve,
and node load balancing, supervised by BFT consensus ratification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .consensus import BFTConsensusEngine, ConsensusProposal
from .node import MeshNode, NodeStatus


@dataclass
class SwarmTaskAllocation:
    """Task allocation record for a swarm logistics mission."""
    task_id: str
    target_xy: Tuple[float, float]
    priority: int
    assigned_node_id: Optional[str]
    distance_km: float
    consensus_committed: bool
    status: str  # 'ASSIGNED', 'UNFEASIBLE_PARTITIONED'


class CortexSwarmRouter:
    """Routes and balances tasks across decentralized mesh nodes."""

    def __init__(self, consensus_engine: BFTConsensusEngine):
        self.consensus = consensus_engine
        self.nodes: Dict[str, MeshNode] = {}
        self.active_allocations: Dict[str, SwarmTaskAllocation] = {}
        self.vector_clock: Dict[str, int] = {}

    def register_node(self, node: MeshNode) -> None:
        """Register a participating swarm node."""
        self.nodes[node.node_id] = node

    def find_best_candidate(self, target_xy: Tuple[float, float]) -> Optional[Tuple[str, float]]:
        """Identify optimal node based on distance, battery reserve, and available capacity."""
        best_node_id: Optional[str] = None
        best_cost = float("inf")
        best_dist = 0.0

        for nid, node in self.nodes.items():
            if not node.is_available:
                continue

            # Calculate Euclidean distance
            dx = target_xy[0] - node.position_xy[0]
            dy = target_xy[1] - node.position_xy[1]
            dist = math.sqrt(dx * dx + dy * dy)

            # Cost function: distance / (battery_factor) + load_penalty
            battery_factor = max(0.1, node.battery_percent / 100.0)
            load_penalty = (node.active_load / node.max_capacity) * 2.0
            cost = (dist / battery_factor) + load_penalty

            if cost < best_cost:
                best_cost = cost
                best_node_id = nid
                best_dist = dist

        if best_node_id is not None:
            return best_node_id, best_dist
        return None

    def allocate_task(
        self,
        task_id: str,
        target_xy: Tuple[float, float],
        priority: int = 1,
    ) -> SwarmTaskAllocation:
        """Allocate a task to the swarm, submitting it to BFT consensus ratification."""
        candidate = self.find_best_candidate(target_xy)
        if candidate is None:
            alloc = SwarmTaskAllocation(
                task_id=task_id,
                target_xy=target_xy,
                priority=priority,
                assigned_node_id=None,
                distance_km=0.0,
                consensus_committed=False,
                status="UNFEASIBLE_PARTITIONED",
            )
            self.active_allocations[task_id] = alloc
            return alloc

        node_id, dist = candidate

        # Propose via BFT consensus engine
        proposal = self.consensus.create_proposal(
            proposal_id=f"alloc_{task_id}",
            payload={"task_id": task_id, "node_id": node_id, "target_xy": target_xy},
        )

        # Update node local load
        node = self.nodes[node_id]
        node.active_load += 1
        node.assigned_tasks.append(task_id)

        alloc = SwarmTaskAllocation(
            task_id=task_id,
            target_xy=target_xy,
            priority=priority,
            assigned_node_id=node_id,
            distance_km=round(dist, 3),
            consensus_committed=False,  # Ratification occurs upon commit vote quorum
            status="ASSIGNED",
        )
        self.active_allocations[task_id] = alloc
        return alloc

    def tick_clock(self, node_id: str) -> Dict[str, int]:
        """Advance local vector clock for a specific node during task or route event."""
        self.vector_clock[node_id] = self.vector_clock.get(node_id, 0) + 1
        return dict(self.vector_clock)

    def merge_clock(self, incoming_clock: Dict[str, int]) -> Dict[str, int]:
        """Merge incoming vector clock via elementwise supremum to preserve Lamport/Mattern causality."""
        for nid, count in incoming_clock.items():
            self.vector_clock[nid] = max(self.vector_clock.get(nid, 0), count)
        return dict(self.vector_clock)

    @staticmethod
    def is_causally_dependent(clock_a: Dict[str, int], clock_b: Dict[str, int]) -> bool:
        """Check if clock_a happened before clock_b (a <= b across all keys and a != b)."""
        all_keys = set(clock_a.keys()).union(set(clock_b.keys()))
        strictly_less = False
        for k in all_keys:
            val_a = clock_a.get(k, 0)
            val_b = clock_b.get(k, 0)
            if val_a > val_b:
                return False
            if val_a < val_b:
                strictly_less = True
        return strictly_less
