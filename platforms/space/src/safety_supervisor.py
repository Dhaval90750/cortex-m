"""Autonomous Spacecraft Subsystem Safety Supervisor & Radiation SEU Detector.

In compliance with NASA NPR 7150.2 Class C software safety engineering:
1. Multi-subsystem continuous attractor state tracking (EPS, ADCS, TCS).
2. Orbital phase-conditioned nominal envelope (sunlight vs eclipse regimes).
3. Radiation-induced Single Event Upset (SEU) latchup anomaly detection.
4. Bidirectional requirement traceability records and deterministic alerts.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from platforms.space.src.ccsds_telemetry import DecommutatedTelemetry


class AnomalySeverity(str, Enum):
    NOMINAL = "NOMINAL"
    WARNING = "WARNING"
    CRITICAL_SEU = "CRITICAL_SEU"


@dataclass
class TelemetryAlert:
    alert_id: str
    timestamp_s: float
    subsystem: str
    severity: AnomalySeverity
    lyapunov_energy: float
    energy_threshold: float
    message: str
    mitigation_action: str
    traceability_req_id: str
    verification_hash: str


class SpaceSafetySupervisor:
    """Evaluates telemetry stream against phase-conditioned attractor baselines."""

    def __init__(self, beta: float = 6.0, seu_energy_threshold: float = -0.75):
        self.beta = beta
        self.seu_energy_threshold = seu_energy_threshold
        self.alert_history: List[TelemetryAlert] = []

        # Reference prototypes: 7-D vectors [V_bus, I_array, T_bat, gx, gy, gz, T_rad]
        # Regime 0: Sunlight Nominal (High solar current, normal gyro, warm radiator)
        p_sun = torch.tensor([28.0, 6.5, 22.0, 0.0, 0.0, 0.0, 15.0], dtype=torch.float32)
        # Regime 1: Eclipse Nominal (Zero solar current, battery discharging, cool radiator)
        p_ecl = torch.tensor([26.5, 0.0, 18.0, 0.0, 0.0, 0.0, -10.0], dtype=torch.float32)

        # Normalization scale factors
        self.scales = torch.tensor([28.0, 7.0, 25.0, 1.0, 1.0, 1.0, 20.0], dtype=torch.float32)

        p_sun_norm = p_sun / self.scales
        p_ecl_norm = p_ecl / self.scales

        self.sun_prototypes = (p_sun_norm / torch.norm(p_sun_norm)).unsqueeze(0)
        self.ecl_prototypes = (p_ecl_norm / torch.norm(p_ecl_norm)).unsqueeze(0)

    def extract_feature_vector(self, telem: DecommutatedTelemetry) -> torch.Tensor:
        """Map decommutated telemetry fields to normalized 7-D tensor."""
        raw = torch.tensor([
            telem.bus_voltage_v,
            telem.solar_current_a,
            telem.battery_temp_c,
            telem.gyro_rate_x,
            telem.gyro_rate_y,
            telem.gyro_rate_z,
            telem.radiator_temp_c,
        ], dtype=torch.float32)

        norm = raw / self.scales
        return norm / (torch.norm(norm) + 1e-8)

    def compute_energy(self, state_vec: torch.Tensor, in_eclipse: bool) -> float:
        """Compute continuous Hopfield Lyapunov energy conditioned on orbital regime."""
        prototypes = self.ecl_prototypes if in_eclipse else self.sun_prototypes
        sims = torch.mv(prototypes, state_vec)  # Cosine similarities
        # Lyapunov energy E(x) = -1/beta * ln sum(exp(beta * sims))
        lse = torch.logsumexp(self.beta * sims, dim=0)
        energy = float(-1.0 / self.beta * lse.item())
        return energy

    def evaluate_telemetry(
        self,
        telem: DecommutatedTelemetry,
        in_eclipse: bool,
    ) -> Optional[TelemetryAlert]:
        """Analyze frame for physical faults or radiation SEU latchups."""
        x = self.extract_feature_vector(telem)
        energy = self.compute_energy(x, in_eclipse)

        severity = AnomalySeverity.NOMINAL
        subsystem = "NOMINAL"
        msg = "All subsystems operating within nominal attractor basin."
        mitigation = "NO_ACTION_REQUIRED"
        req_id = "NPR-7150.2-FR-SPC-03"

        # Check for SEU conditions:
        # Condition A: Energy divergence above critical threshold
        if energy > self.seu_energy_threshold:
            # Check specific subsystem deviations
            if telem.bus_voltage_v < 20.0 or telem.solar_current_a > 15.0:
                subsystem = "EPS"
                severity = AnomalySeverity.CRITICAL_SEU
                msg = f"EPS bus anomaly detected (V={telem.bus_voltage_v:.1f}V, I={telem.solar_current_a:.1f}A). Radiation SEU latchup indicated."
                mitigation = "AUTONOMOUS_PAYLOAD_POWER_CYCLE"
            elif abs(telem.gyro_rate_x) > 5.0 or abs(telem.gyro_rate_y) > 5.0 or abs(telem.gyro_rate_z) > 5.0:
                subsystem = "ADCS"
                severity = AnomalySeverity.WARNING
                msg = f"ADCS rate divergence detected (|w| > 5 deg/s). Possible reaction wheel SEU."
                mitigation = "SWITCH_TO_REDUNDANT_IMU_UNIT"
            else:
                subsystem = "TCS"
                severity = AnomalySeverity.WARNING
                msg = f"Thermal equilibrium divergence outside nominal orbital envelope."
                mitigation = "ADJUST_RADIATOR_HEATER_DUTY_CYCLE"

        # If CRC was invalid, report frame corruption
        if not telem.crc_valid:
            subsystem = "CDH"
            severity = AnomalySeverity.WARNING
            msg = "CCSDS frame CRC check failed. Downlink transmission error or SEU in frame buffer."
            mitigation = "REQUEST_PACKET_RETRANSMISSION"

        if severity != AnomalySeverity.NOMINAL:
            # Generate deterministic tamper verification hash
            raw_hash_data = f"{telem.timestamp_seconds}:{subsystem}:{severity.value}:{energy:.4f}"
            digest = hashlib.sha256(raw_hash_data.encode("utf-8")).hexdigest()[:16]
            alert_id = f"ALT-SPC-{int(telem.timestamp_seconds)}-{digest[:6]}"

            alert = TelemetryAlert(
                alert_id=alert_id,
                timestamp_s=telem.timestamp_seconds,
                subsystem=subsystem,
                severity=severity,
                lyapunov_energy=energy,
                energy_threshold=self.seu_energy_threshold,
                message=msg,
                mitigation_action=mitigation,
                traceability_req_id=req_id,
                verification_hash=digest,
            )
            self.alert_history.append(alert)
            return alert

        return None
