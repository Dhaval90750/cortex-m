"""Mesh Node State & Telemetry for CORTEX-Mesh.

Represents an individual physical or simulated rover, drone, or sensor node in a cooperative swarm.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class NodeStatus(str, enum.Enum):
    ONLINE = "ONLINE"
    DEGRADED = "DEGRADED"
    FAULTY = "FAULTY"
    OFFLINE = "OFFLINE"


@dataclass
class MeshNode:
    """State descriptor for a participating swarm node."""
    node_id: str
    position_xy: Tuple[float, float]
    battery_percent: float = 100.0
    status: NodeStatus = NodeStatus.ONLINE
    active_load: int = 0
    max_capacity: int = 5
    assigned_tasks: List[str] = field(default_factory=list)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_available(self) -> bool:
        """Return True if node is online and has capacity for new tasks."""
        return (
            self.status == NodeStatus.ONLINE
            and self.battery_percent > 15.0
            and self.active_load < self.max_capacity
        )

    def update_heartbeat(self, battery: Optional[float] = None, position: Optional[Tuple[float, float]] = None) -> None:
        """Update node heartbeat and telemetry."""
        self.last_heartbeat = time.time()
        if battery is not None:
            self.battery_percent = max(0.0, min(100.0, battery))
            if self.battery_percent < 10.0:
                self.status = NodeStatus.DEGRADED
        if position is not None:
            self.position_xy = position
