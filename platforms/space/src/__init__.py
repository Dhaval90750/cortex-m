"""CORTEX-Space Autonomous Orbital Simulation & Telemetry System."""

from platforms.space.src.ccsds_telemetry import (
    CCSDSTelemetryProcessor,
    DecommutatedTelemetry,
    SpacePacketHeader,
    SyncState,
    crc16_ccitt,
)
from platforms.space.src.orbit_propagator import (
    DEFAULT_GROUND_STATIONS,
    GroundStation,
    OrbitalElements,
    OrbitalStatePoint,
    OrbitPropagator,
)
from platforms.space.src.safety_supervisor import (
    AnomalySeverity,
    SpaceSafetySupervisor,
    TelemetryAlert,
)

__all__ = [
    "CCSDSTelemetryProcessor",
    "DecommutatedTelemetry",
    "SpacePacketHeader",
    "SyncState",
    "crc16_ccitt",
    "DEFAULT_GROUND_STATIONS",
    "GroundStation",
    "OrbitalElements",
    "OrbitalStatePoint",
    "OrbitPropagator",
    "AnomalySeverity",
    "SpaceSafetySupervisor",
    "TelemetryAlert",
]
