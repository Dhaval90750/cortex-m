"""Causal Sparse Swarm Router & Task Allocation for CORTEX-Mesh.

Ranks and assigns cooperative civilian swarm tasks using proximity, battery reserve,
and node load balancing, supervised by BFT consensus ratification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

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

    def reconcile_gossip_digest(
        self,
        peer_node_id: str,
        peer_clock: Dict[str, int],
        peer_known_tasks: Sequence[str],
    ) -> Dict[str, Any]:
        """Anti-entropy epidemic gossip reconciliation between two mesh swarm nodes.
        
        Calculates set differences and causal lag to synchronize state over lossy RF links.
        """
        # Tasks we have that peer lacks
        peer_task_set = set(peer_known_tasks)
        missing_on_peer = [tid for tid in self.active_allocations if tid not in peer_task_set]

        # Causal divergence analysis
        local_ahead = []
        peer_ahead = []
        for nid in set(self.vector_clock.keys()).union(peer_clock.keys()):
            loc_t = self.vector_clock.get(nid, 0)
            peer_t = peer_clock.get(nid, 0)
            if loc_t > peer_t:
                local_ahead.append({"node_id": nid, "delta": loc_t - peer_t})
            elif peer_t > loc_t:
                peer_ahead.append({"node_id": nid, "delta": peer_t - loc_t})

        # Advance our clock by merging
        self.merge_clock(peer_clock)

        return {
            "peer_node_id": peer_node_id,
            "tasks_sent_to_peer": missing_on_peer,
            "num_tasks_propagated": len(missing_on_peer),
            "causal_divergence": {
                "local_ahead_events": local_ahead,
                "peer_ahead_events": peer_ahead,
            },
            "reconciled_vector_clock": dict(self.vector_clock),
            "sync_status": "SYNCHRONIZED" if not missing_on_peer and not peer_ahead else "DELTA_TRANSFERRED",
        }

    def detect_byzantine_nodes(
        self,
        vote_audit_log: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Identify and isolate Byzantine / adversarial swarm nodes attempting equivocation or invalid signatures."""
        equivocations: Dict[str, Set[str]] = {}
        signature_faults: Dict[str, int] = {}
        faulty_nodes: Set[str] = set()

        for record in vote_audit_log:
            node_id = record.get("voter_node_id")
            if not node_id:
                continue

            seq = record.get("sequence_number", 0)
            digest = record.get("digest", "")
            sig = record.get("signature", "")

            # Check signature integrity
            if not sig or node_id not in sig:
                signature_faults[node_id] = signature_faults.get(node_id, 0) + 1
                faulty_nodes.add(node_id)

            # Check equivocation (double-voting different digests for same sequence number)
            key = f"{node_id}:{seq}"
            if key not in equivocations:
                equivocations[key] = set()
            equivocations[key].add(digest)
            if len(equivocations[key]) > 1:
                faulty_nodes.add(node_id)

        # De-register or mark unavailable any byzantine nodes
        isolated = []
        for fn in faulty_nodes:
            if fn in self.nodes:
                self.nodes[fn].is_available = False
                isolated.append(fn)

        return {
            "audited_records": len(vote_audit_log),
            "byzantine_nodes_detected": list(faulty_nodes),
            "isolated_from_routing": isolated,
            "signature_fault_counts": signature_faults,
            "swarm_security_status": "COMPROMISED_ISOLATED" if faulty_nodes else "NOMINAL",
        }
