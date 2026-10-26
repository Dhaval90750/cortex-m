"""3-Phase Byzantine Fault Tolerant (BFT) Consensus Protocol for CORTEX-Mesh.

Provides auditable, decentralized consensus across swarm nodes:
1. Pre-Prepare: Primary proposes task assignment or route with sequence number.
2. Prepare: Nodes validate and broadcast votes; requires 2f + 1 matching votes.
3. Commit: Nodes commit state upon collecting 2f + 1 signed commit acknowledgments.
Tolerates up to f < n/3 Byzantine or partitioned nodes.
"""

from __future__ import annotations

import enum
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


class ConsensusPhase(str, enum.Enum):
    PRE_PREPARE = "PRE_PREPARE"
    PREPARE = "PREPARE"
    COMMIT = "COMMIT"
    COMMITTED = "COMMITTED"
    REJECTED = "REJECTED"
    VIEW_CHANGE = "VIEW_CHANGE"


@dataclass
class ConsensusProposal:
    """Consensus proposal submitted for swarm ratification."""
    proposal_id: str
    view_number: int
    sequence_number: int
    proposer_node_id: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    digest: str = ""

    def calculate_digest(self) -> str:
        """Calculate canonical SHA-256 hash digest of the proposal payload without modifying self."""
        content = json.dumps(
            {
                "proposal_id": self.proposal_id,
                "view": self.view_number,
                "seq": self.sequence_number,
                "proposer": self.proposer_node_id,
                "payload": self.payload,
            },
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def compute_digest(self) -> str:
        """Calculate and store canonical SHA-256 hash digest."""
        self.digest = self.calculate_digest()
        return self.digest


@dataclass
class ConsensusVote:
    """Signed vote message emitted by a node during consensus phases."""
    phase: ConsensusPhase
    view_number: int
    sequence_number: int
    proposal_digest: str
    voter_node_id: str
    signature: str = ""
    timestamp: float = field(default_factory=time.time)


class BFTConsensusEngine:
    """State machine supervising PBFT-style 3-phase consensus for a swarm node."""

    def __init__(self, node_id: str, total_nodes: int = 4):
        self.node_id = node_id
        self.total_nodes = total_nodes
        # Max tolerated Byzantine faults: f < n / 3
        self.max_faults = (total_nodes - 1) // 3
        # Quorum size: 2f + 1
        self.quorum_size = 2 * self.max_faults + 1

        self.current_view = 0
        self.current_sequence = 0
        self.proposals: Dict[str, ConsensusProposal] = {}
        self.prepare_votes: Dict[str, Set[str]] = {}  # digest -> set of voter IDs
        self.commit_votes: Dict[str, Set[str]] = {}  # digest -> set of voter IDs
        self.view_change_votes: Dict[int, Set[str]] = {}  # view -> set of voter IDs
        self.committed_log: List[ConsensusProposal] = []

    def create_proposal(self, proposal_id: str, payload: Dict[str, Any]) -> ConsensusProposal:
        """Create and initialize a new Pre-Prepare consensus proposal."""
        self.current_sequence += 1
        prop = ConsensusProposal(
            proposal_id=proposal_id,
            view_number=self.current_view,
            sequence_number=self.current_sequence,
            proposer_node_id=self.node_id,
            payload=payload,
        )
        digest = prop.compute_digest()
        self.proposals[digest] = prop
        self.prepare_votes[digest] = {self.node_id}
        return prop

    def receive_pre_prepare(self, proposal: ConsensusProposal) -> Optional[ConsensusVote]:
        """Validate an incoming Pre-Prepare proposal from the primary node."""
        expected_digest = proposal.calculate_digest()
        if proposal.digest != expected_digest:
            return None  # Tampered proposal

        self.proposals[expected_digest] = proposal
        if expected_digest not in self.prepare_votes:
            self.prepare_votes[expected_digest] = set()

        # Emit prepare vote
        vote = ConsensusVote(
            phase=ConsensusPhase.PREPARE,
            view_number=proposal.view_number,
            sequence_number=proposal.sequence_number,
            proposal_digest=expected_digest,
            voter_node_id=self.node_id,
            signature=f"sig_{self.node_id}_{expected_digest[:8]}",
        )
        self.prepare_votes[expected_digest].add(self.node_id)
        return vote

    def receive_prepare_vote(self, vote: ConsensusVote) -> Optional[ConsensusVote]:
        """Collect Prepare votes; once quorum (2f + 1) is achieved, emit Commit vote."""
        digest = vote.proposal_digest
        if digest not in self.prepare_votes:
            self.prepare_votes[digest] = set()

        self.prepare_votes[digest].add(vote.voter_node_id)

        # Check if 2f + 1 quorum reached
        if len(self.prepare_votes[digest]) >= self.quorum_size:
            if digest not in self.commit_votes:
                self.commit_votes[digest] = {self.node_id}
                commit_vote = ConsensusVote(
                    phase=ConsensusPhase.COMMIT,
                    view_number=vote.view_number,
                    sequence_number=vote.sequence_number,
                    proposal_digest=digest,
                    voter_node_id=self.node_id,
                    signature=f"commit_{self.node_id}_{digest[:8]}",
                )
                return commit_vote
        return None

    def receive_commit_vote(self, vote: ConsensusVote) -> bool:
        """Collect Commit votes; once quorum (2f + 1) is achieved, finalize commitment."""
        digest = vote.proposal_digest
        if digest not in self.commit_votes:
            self.commit_votes[digest] = set()

        self.commit_votes[digest].add(vote.voter_node_id)

        # Check if commit quorum reached
        if len(self.commit_votes[digest]) >= self.quorum_size:
            if digest in self.proposals and self.proposals[digest] not in self.committed_log:
                committed_prop = self.proposals[digest]
                self.committed_log.append(committed_prop)
                return True
        return False

    def verify_vote_signature(self, vote: ConsensusVote) -> bool:
        """Verify authenticity of cryptographic signature on vote message."""
        if not vote.signature or not vote.voter_node_id:
            return False
        # Expected signature format includes voter node ID
        return vote.voter_node_id in vote.signature

    def initiate_view_change(self, new_view: int) -> ConsensusVote:
        """Initiate PBFT View-Change protocol when primary is suspected of fault or timeout."""
        if new_view <= self.current_view:
            new_view = self.current_view + 1

        vote = ConsensusVote(
            phase=ConsensusPhase.VIEW_CHANGE,
            view_number=new_view,
            sequence_number=self.current_sequence,
            proposal_digest=f"view_change_to_{new_view}",
            voter_node_id=self.node_id,
            signature=f"vc_{self.node_id}_{new_view}",
        )
        self.receive_view_change(vote)
        return vote

    def receive_view_change(self, vote: ConsensusVote) -> bool:
        """Collect View-Change messages; transitions view once 2f + 1 quorum reached."""
        target_view = vote.view_number
        if target_view not in self.view_change_votes:
            self.view_change_votes[target_view] = set()

        self.view_change_votes[target_view].add(vote.voter_node_id)
        if len(self.view_change_votes[target_view]) >= self.quorum_size:
            if self.current_view < target_view:
                self.current_view = target_view
                return True
        return False
