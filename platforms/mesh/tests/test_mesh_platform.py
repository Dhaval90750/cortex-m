"""Unit tests for CORTEX-Mesh platform."""

import unittest

from platforms.mesh.src.consensus import (
    BFTConsensusEngine,
    ConsensusPhase,
    ConsensusProposal,
    ConsensusVote,
)
from platforms.mesh.src.node import MeshNode, NodeStatus
from platforms.mesh.src.swarm_router import CortexSwarmRouter, SwarmTaskAllocation


class TestCortexMeshPlatform(unittest.TestCase):
    def setUp(self):
        self.consensus = BFTConsensusEngine(node_id="node_0", total_nodes=4)
        self.router = CortexSwarmRouter(self.consensus)

    def test_mesh_node_lifecycle_and_availability(self):
        node = MeshNode(node_id="rover_1", position_xy=(10.0, 10.0), battery_percent=85.0)
        self.assertTrue(node.is_available)

        # Deplete battery
        node.update_heartbeat(battery=8.0)
        self.assertFalse(node.is_available)
        self.assertEqual(node.status, NodeStatus.DEGRADED)

    def test_bft_3phase_consensus_lifecycle(self):
        # 4 nodes: n=4 => f=1 => quorum=3
        self.assertEqual(self.consensus.quorum_size, 3)

        # 1. Primary creates proposal (Pre-Prepare)
        prop = self.consensus.create_proposal(
            proposal_id="prop_mission_01",
            payload={"zone": "sector_b4", "rover": "rover_1"},
        )
        digest = prop.digest
        self.assertTrue(len(digest) == 64)

        # 2. Node receives Prepare votes from peers
        vote_1 = ConsensusVote(
            phase=ConsensusPhase.PREPARE,
            view_number=0,
            sequence_number=1,
            proposal_digest=digest,
            voter_node_id="node_1",
        )
        vote_2 = ConsensusVote(
            phase=ConsensusPhase.PREPARE,
            view_number=0,
            sequence_number=1,
            proposal_digest=digest,
            voter_node_id="node_2",
        )

        commit_emit_1 = self.consensus.receive_prepare_vote(vote_1)
        self.assertIsNone(commit_emit_1)  # Only 2 votes (node_0, node_1), need 3

        commit_emit_2 = self.consensus.receive_prepare_vote(vote_2)
        # Quorum of 3 reached (node_0, node_1, node_2) -> emits Commit vote
        self.assertIsNotNone(commit_emit_2)
        self.assertEqual(commit_emit_2.phase, ConsensusPhase.COMMIT)

        # 3. Collect Commit votes from peers
        c_vote_1 = ConsensusVote(
            phase=ConsensusPhase.COMMIT,
            view_number=0,
            sequence_number=1,
            proposal_digest=digest,
            voter_node_id="node_1",
        )
        c_vote_2 = ConsensusVote(
            phase=ConsensusPhase.COMMIT,
            view_number=0,
            sequence_number=1,
            proposal_digest=digest,
            voter_node_id="node_2",
        )

        committed_1 = self.consensus.receive_commit_vote(c_vote_1)
        self.assertFalse(committed_1)

        committed_2 = self.consensus.receive_commit_vote(c_vote_2)
        # Quorum of 3 commits reached -> permanently committed
        self.assertTrue(committed_2)
        self.assertIn(prop, self.consensus.committed_log)

    def test_tampered_proposal_rejected(self):
        prop = ConsensusProposal(
            proposal_id="tampered_01",
            view_number=0,
            sequence_number=1,
            proposer_node_id="attacker_node",
            payload={"action": "drop_supplies"},
            digest="invalid_sha256_hash",
        )
        res = self.consensus.receive_pre_prepare(prop)
        self.assertIsNone(res)

    def test_swarm_router_task_allocation(self):
        # Register 3 nodes
        n1 = MeshNode(node_id="rover_near", position_xy=(2.0, 2.0), battery_percent=90.0)
        n2 = MeshNode(node_id="rover_far", position_xy=(50.0, 50.0), battery_percent=95.0)
        n3 = MeshNode(node_id="rover_dead", position_xy=(1.0, 1.0), battery_percent=5.0)  # Low battery

        self.router.register_node(n1)
        self.router.register_node(n2)
        self.router.register_node(n3)

        # Allocate task near (3.0, 3.0) -> rover_near should be chosen
        alloc = self.router.allocate_task(task_id="rescue_mission_alpha", target_xy=(3.0, 3.0))
        self.assertEqual(alloc.status, "ASSIGNED")
        self.assertEqual(alloc.assigned_node_id, "rover_near")
        self.assertLess(alloc.distance_km, 2.0)
        self.assertEqual(n1.active_load, 1)

    def test_partitioned_swarm_unfeasible(self):
        # Empty router has no available nodes
        alloc = self.router.allocate_task(task_id="orphaned_task", target_xy=(10.0, 10.0))
        self.assertEqual(alloc.status, "UNFEASIBLE_PARTITIONED")
        self.assertIsNone(alloc.assigned_node_id)


if __name__ == "__main__":
    unittest.main()
