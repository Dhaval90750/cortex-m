"""CORTEX-M Universal Platform Gateway.

Provides a unified programmatic facade connecting all 8 specialized platforms
and the developmental learning / synaptic grafting engine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Ensure repo root is available for platform imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Core learning & Beast Neural Substrate
from cortex_m.cortex_v2 import CortexV2
from cortex_m.learning.consolidation import NocturnalSleepConsolidator, SleepCycleReport
from cortex_m.learning.grafting import (
    GraftingResult,
    SynapticKnowledgePackager,
    SynapticPackageManifest,
)

# Platforms
from platforms.personal.src.interlock import ZeroEgressInterlock
from platforms.personal.src.substrate import PersonalSubstrateCoordinator
from platforms.personal.src.tenant_paging import PersonalTenantPagingEngine
from platforms.personal.src.vault import EncryptedPersonalVault

from platforms.ide.src.ast_parser import ASTSymbolTableParser
from platforms.ide.src.diff_engine import ASTDiffEngine
from platforms.ide.src.sandbox import SubprocessSandboxRunner
from platforms.ide.src.swarm import CortexCodeSwarm

from platforms.fin.src.anomaly_scorer import FinancialAnomalyScorer, FraudScoringReport
from platforms.fin.src.event_replay import HighThroughputReplayEngine
from platforms.fin.src.transaction_encoder import FinancialTransaction, TransactionFeatureEncoder

from platforms.robotics.src.kinematics import Manipulator6DOF
from platforms.robotics.src.path_planner import AttractorPathPlanner
from platforms.robotics.src.safety_interlock import DeterministicSafetyInterlock, SafetyStatusReport

from platforms.mesh.src.consensus import BFTConsensusEngine, ConsensusPhase, ConsensusProposal, ConsensusVote
from platforms.mesh.src.node import MeshNode, NodeStatus
from platforms.mesh.src.swarm_router import CortexSwarmRouter, SwarmTaskAllocation

from platforms.health.src.bci_decoder import BCIDecodingResult, CognitiveState, ContinuousBCIDecoder
from platforms.health.src.signal_filter import BiosignalFilter, FilteredSignal
from platforms.health.src.wavelet_analysis import NeuralBandpower, WaveletFrequencyDecomposer

from platforms.space.src.ccsds_telemetry import CCSDSTelemetryProcessor, DecommutatedTelemetry
from platforms.space.src.orbit_propagator import OrbitalElements, OrbitPropagator
from platforms.space.src.safety_supervisor import AnomalySeverity as SpaceSeverity, SpaceSafetySupervisor

from platforms.os.src.affinity_cluster import AffinityClusterReport, CognitiveAffinityGraph
from platforms.os.src.file_watcher import FileDocument, NonPrivilegedFileWatcher
from platforms.os.src.omnibar_search import OmnibarQueryResponse, SpotlightOmnibarEngine
from platforms.os.src.system_monitor import SystemResourceMonitor


class EcosystemGateway:
    """Universal facade orchestrating all 8 platform engines and core learning."""

    _instance: Optional[EcosystemGateway] = None

    def __init__(self):
        # Platform singletons
        self.personal_substrate = PersonalSubstrateCoordinator()
        self.ide_parser = ASTSymbolTableParser()
        self.ide_diff = ASTDiffEngine()
        self.ide_sandbox = SubprocessSandboxRunner(default_timeout_seconds=5.0)
        self.fin_scorer = FinancialAnomalyScorer(dim=32)
        self.fin_encoder = TransactionFeatureEncoder(output_dim=32)
        self.robotics_arm = Manipulator6DOF()
        self.robotics_interlock = DeterministicSafetyInterlock(manipulator=self.robotics_arm)
        self.robotics_planner = AttractorPathPlanner(manipulator=self.robotics_arm, interlock=self.robotics_interlock)
        self.mesh_consensus = BFTConsensusEngine(node_id="node_gateway", total_nodes=4)
        self.mesh_router = CortexSwarmRouter(consensus_engine=self.mesh_consensus)
        self.health_filter = BiosignalFilter(sampling_rate_hz=250.0)
        self.health_decomposer = WaveletFrequencyDecomposer(sampling_rate_hz=250.0)
        self.health_decoder = ContinuousBCIDecoder(beta=8.0)
        self.space_ccsds = CCSDSTelemetryProcessor()
        self.space_elements = OrbitalElements(
            semi_major_axis_km=6878.137,
            eccentricity=0.001,
            inclination_deg=51.6,
            raan_deg=45.0,
            arg_perigee_deg=0.0,
            mean_anomaly_deg=0.0,
            epoch_timestamp_s=1700000000.0,
        )
        self.space_propagator = OrbitPropagator(self.space_elements)
        self.space_supervisor = SpaceSafetySupervisor(beta=6.0, seu_energy_threshold=-0.75)
        self.os_watcher = NonPrivilegedFileWatcher(debounce_seconds=0.5)
        self.os_omnibar = SpotlightOmnibarEngine(beta=8.0, hopfield_iterations=3)
        self.os_graph = CognitiveAffinityGraph(affinity_threshold=0.45, sigma_affinity=1.0)
        self.os_monitor = SystemResourceMonitor()
        self.learning_consolidator = NocturnalSleepConsolidator()
        self.learning_packager = SynapticKnowledgePackager(dim=32)
        self.cortex = CortexV2(dim=64)

    @classmethod
    def get_instance(cls) -> EcosystemGateway:
        if cls._instance is None:
            cls._instance = EcosystemGateway()
        return cls._instance

    def get_ecosystem_overview(self) -> Dict[str, Any]:
        """Return comprehensive status and readiness metadata across all platforms."""
        return {
            "version": "2.2.0-COMMERCIAL",
            "status": "OPERATIONAL",
            "platforms": {
                "personal": {
                    "identity": "CORTEX-Personal",
                    "status": "READY",
                    "features": ["zero_egress_interlock", "encrypted_vault", "tenant_paging_2kb", "tabula_rasa_hebbian"],
                },
                "ide": {
                    "identity": "CORTEX-IDE",
                    "status": "READY",
                    "features": ["ast_symbol_parser", "diff_engine", "subprocess_sandbox_5s", "code_swarm"],
                },
                "fin": {
                    "identity": "CORTEX-Fin",
                    "status": "READY",
                    "features": ["transaction_encoder_32d", "hopfield_energy_attribution", "stream_replay_5000tps"],
                },
                "robotics": {
                    "identity": "CORTEX-Robotics",
                    "status": "READY",
                    "features": ["6dof_kinematics", "iso_10218_interlock", "quintic_attractor_planner"],
                },
                "mesh": {
                    "identity": "CORTEX-Mesh",
                    "status": "READY",
                    "features": ["pbft_bft_consensus", "mesh_node_health", "causal_sparse_router"],
                },
                "health": {
                    "identity": "CORTEX-Health",
                    "status": "READY",
                    "disclaimer": "NON-CLINICAL RESEARCH ONLY",
                    "features": ["biosignal_filter_60hz_notch", "wavelet_bandpower", "continuous_bci_decoder"],
                },
                "space": {
                    "identity": "CORTEX-Space",
                    "status": "READY",
                    "compliance": "NASA NPR 7150.2 Class C",
                    "features": ["ccsds_sync_asm_crc16", "j2_orbit_propagator", "seu_radiation_supervisor"],
                },
                "os": {
                    "identity": "CORTEX-OS",
                    "status": "READY",
                    "mode": "non_privileged_userspace",
                    "features": ["debounced_watcher_500ms", "affinity_clustering", "spotlight_omnibar_sub1ms", "memory_ceiling_2_5mb"],
                },
                "learning": {
                    "identity": "CORTEX-Developmental-Learning",
                    "status": "READY",
                    "features": ["nocturnal_sleep_consolidation", "synaptic_knowledge_packager", "orthogonal_grafting_0_collateral"],
                },
            },
        }

    def execute_personal(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CORTEX-Personal workflow."""
        tid = params.get("tenant_id", "default_user")
        action_norm = action.lower().strip()
        if action_norm in ("learn", "teach"):
            text = params.get("text", "")
            mem_id = self.personal_substrate.teach(text, source_context=f"tenant:{tid}")
            return {"success": True, "tenant_id": tid, "learned": text, "memory_id": mem_id}
        elif action_norm in ("recall", "query"):
            query_str = params.get("query", params.get("cue", ""))
            res = self.personal_substrate.query(query_str)
            return {
                "success": True,
                "tenant_id": tid,
                "query": query_str,
                "status": res.status,
                "confidence": res.confidence,
                "recalled_text": res.recalled_text,
                "energy": res.energy,
                "latency_ms": res.latency_ms,
            }
        elif action_norm in ("interlock", "check_interlock"):
            host = params.get("host", "127.0.0.1")
            port = int(params.get("port", 8888))
            try:
                ZeroEgressInterlock.assert_loopback_only(host, port)
                return {"success": True, "egress_permitted": True, "host": host, "port": port}
            except Exception as e:
                return {"success": False, "egress_permitted": False, "error": str(e)}
        elif action_norm in ("vault", "vault_status"):
            return {
                "success": True,
                "active_tenant": tid,
                "num_memories": len(self.personal_substrate._memory_text_store),
                "dim": self.personal_substrate.dim,
                "beta": self.personal_substrate.beta,
                "interlock_active": self.personal_substrate.interlock.strict_mode,
            }
        elif action_norm in ("unlearn", "forget"):
            target = params.get("memory_id") or params.get("text", "")
            success = self.personal_substrate.unlearn(target)
            return {"success": success, "target": target}
        elif action_norm in ("sleep", "consolidate"):
            report = self.personal_substrate.sleep_consolidate()
            return {
                "success": True,
                "replayed_attractors": report.replayed_attractors,
                "pruned_synapses": report.pruned_synapses,
                "retention_accuracy": report.retention_accuracy,
            }
        else:
            raise ValueError(f"Unknown personal action: {action}")

    def execute_ide(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CORTEX-IDE workflow."""
        action_norm = action.lower().strip()
        if action_norm in ("audit", "parse", "parse_ast"):
            code = params.get("code", "")
            symbols = self.ide_parser.parse_source(code)
            all_calls: List[str] = []
            for f in symbols.functions.values():
                all_calls.extend(f.calls)
            for c in symbols.classes.values():
                all_calls.extend(c.calls)
            return {
                "success": symbols.syntax_valid,
                "classes": list(symbols.classes.keys()),
                "functions": list(symbols.functions.keys()),
                "imports": symbols.imports,
                "calls": sorted(list(set(all_calls))),
                "security_flaws": symbols.security_issues,
                "is_clean": len(symbols.security_issues) == 0,
                "error_message": symbols.error_message,
            }
        elif action_norm in ("diff", "compute_diff"):
            orig = params.get("original", params.get("original_code", ""))
            mod = params.get("modified", params.get("modified_code", ""))
            filename = params.get("filename", "file.py")
            diff_res = self.ide_diff.compute_diff(orig, mod, filename)
            return {
                "success": True,
                "unified_diff": diff_res.unified_diff,
                "syntax_valid": diff_res.syntax_valid,
                "line_changes": diff_res.line_changes,
                "lines_added": diff_res.lines_added,
                "lines_removed": diff_res.lines_removed,
            }
        elif action_norm in ("sandbox", "execute"):
            cmd = params.get("command", [sys.executable, "-c", "print('sandbox ok')"])
            if isinstance(cmd, str):
                cmd = [cmd]
            res = self.ide_sandbox.run_command(cmd)
            return {
                "success": res.success,
                "exit_code": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "timed_out": res.timed_out,
                "duration_ms": res.duration_ms,
            }
        else:
            raise ValueError(f"Unknown ide action: {action}")

    def execute_fin(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CORTEX-Fin workflow."""
        action_norm = action.lower().strip()
        if action_norm in ("score", "evaluate"):
            tx = FinancialTransaction(
                tx_id=params.get("transaction_id", "tx_001"),
                account_id=params.get("account_id", "acc_1001"),
                amount_usd=float(params.get("amount", 125.0)),
                mcc=int(params.get("mcc", 5411)),
                geo_distance_km=float(params.get("distance_km", 5.0)),
                time_delta_seconds=float(params.get("time_delta_seconds", 3600.0)),
            )
            res = self.fin_scorer.score_transaction(tx)
            return {
                "success": True,
                "transaction_id": tx.tx_id,
                "lyapunov_energy": res.lyapunov_energy,
                "calibrated_risk": res.risk_score,
                "decision": res.decision,
                "feature_attributions": res.feature_attributions,
                "scoring_latency_us": res.scoring_latency_us,
            }
        elif action_norm in ("replay", "benchmark"):
            count = int(params.get("count", 100))
            replay_engine = HighThroughputReplayEngine(scorer=self.fin_scorer, encoder=self.fin_encoder)
            benchmark = replay_engine.benchmark_stream(count)
            return {
                "success": True,
                "total_transactions": benchmark["total_transactions"],
                "throughput_tps": benchmark["throughput_tps"],
                "p50_latency_ms": benchmark["p50_latency_ms"],
                "p95_latency_ms": benchmark["p95_latency_ms"],
                "p99_latency_ms": benchmark["p99_latency_ms"],
            }
        else:
            raise ValueError(f"Unknown fin action: {action}")

    def execute_robotics(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CORTEX-Robotics workflow."""
        action_norm = action.lower().strip()
        if action_norm in ("kinematics", "fk"):
            joints = params.get("joints", params.get("joint_angles", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            fk = self.robotics_arm.forward_kinematics(joints)
            is_valid = self.robotics_arm.validate_joint_limits(joints)
            return {
                "success": True,
                "joint_angles_rad": joints,
                "end_effector_position": list(fk.position_xyz),
                "joints_valid": is_valid,
            }
        elif action_norm in ("plan", "plan_path"):
            start = params.get("start", params.get("current_joint_angles", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            goal = params.get("goal", [0.5, -0.3, 0.8, 0.0, 0.2, 0.0])
            path = self.robotics_planner.plan_trajectory(start, goal)
            return {
                "success": path.is_safety_certified,
                "num_waypoints": len(path.waypoints_rad),
                "duration_seconds": path.total_duration_s,
                "max_tcp_speed_m_s": path.max_tcp_speed_m_s,
                "is_safety_certified": path.is_safety_certified,
                "waypoints": path.waypoints_rad[::max(1, len(path.waypoints_rad) // 5)],
            }
        elif action_norm in ("safety", "safety_check"):
            dist = float(params.get("human_distance_m", 1.5))
            allowed_speed = self.robotics_interlock.calculate_allowable_tcp_speed(dist)
            current_j = params.get("current_joints", [0.0] * 6)
            target_j = params.get("target_joints", [0.0005] * 6)
            dt_s = float(params.get("dt_seconds", 0.05))
            report = self.robotics_interlock.inspect_step(
                current_joints=current_j,
                target_joints=target_j,
                dt_seconds=dt_s,
                human_distance_m=dist,
            )
            return {
                "success": True,
                "is_safe": report.is_safe,
                "human_distance_m": dist,
                "allowed_tcp_speed_m_s": allowed_speed,
                "motion_allowed": report.is_safe and not report.e_stop_active,
                "estop_active": report.e_stop_active,
                "violations": report.violations,
            }
        elif action_norm in ("ik", "inverse_kinematics"):
            target_xyz = params.get("target_xyz", [0.4, 0.2, 0.3])
            seed = params.get("seed_joints", [0.0] * 6)
            converged, joint_angles, error_m = self.robotics_arm.inverse_kinematics(target_xyz, initial_guess=seed)
            return {
                "success": converged,
                "target_position": list(target_xyz),
                "joint_angles_rad": joint_angles,
                "position_error_m": error_m,
            }
        elif action_norm in ("manipulability", "manipulability_index"):
            joints = params.get("joints", [0.0] * 6)
            w = self.robotics_arm.manipulability_index(joints)
            return {
                "success": True,
                "joint_angles_rad": joints,
                "manipulability_index": w,
            }
        else:
            raise ValueError(f"Unknown robotics action: {action}")

    def execute_mesh(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CORTEX-Mesh workflow."""
        action_norm = action.lower().strip()
        if action_norm in ("consensus", "consensus_round"):
            proposal = ConsensusProposal(
                proposal_id=params.get("proposal_id", "prop_001"),
                view_number=0,
                sequence_number=int(params.get("round_id", 1)),
                proposer_node_id="node_gateway",
                payload={"data": params.get("proposal", "BLOCK_HEADER_001")},
            )
            proposal.compute_digest()
            vote = self.mesh_consensus.receive_pre_prepare(proposal)
            return {
                "success": vote is not None,
                "proposal_id": proposal.proposal_id,
                "digest": proposal.digest,
                "phase": vote.phase.value if vote else "REJECTED",
                "voter": vote.voter_node_id if vote else None,
            }
        elif action_norm in ("route", "route_task"):
            task_id = params.get("task_id", "task_001")
            target_xy = tuple(params.get("target_xy", (10.0, 15.0)))
            if not self.mesh_router.nodes:
                self.mesh_router.register_node(MeshNode("rover_alpha", (5.0, 5.0), 95.0))
                self.mesh_router.register_node(MeshNode("rover_beta", (20.0, 25.0), 80.0))
            alloc = self.mesh_router.allocate_task(task_id, target_xy)
            return {
                "success": alloc.assigned_node_id is not None,
                "task_id": alloc.task_id,
                "assigned_node": alloc.assigned_node_id,
                "distance_km": alloc.distance_km,
                "status": alloc.status,
            }
        else:
            raise ValueError(f"Unknown mesh action: {action}")

    def execute_health(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CORTEX-Health workflow."""
        action_norm = action.lower().strip()
        if action_norm in ("filter", "filter_signal"):
            raw = params.get("signal", params.get("samples", [50.0 + 15.0 * np.sin(0.1 * i) for i in range(250)]))
            filtered = self.health_filter.filter_signal(np.array(raw, dtype=np.float64))
            return {
                "success": True,
                "sampling_rate_hz": filtered.sampling_rate_hz,
                "duration_seconds": filtered.duration_seconds,
                "artifacts_clipped_count": filtered.artifacts_clipped_count,
                "mean_uv": float(np.mean(filtered.clean_microvolts)),
                "max_uv": float(np.max(filtered.clean_microvolts)),
                "min_uv": float(np.min(filtered.clean_microvolts)),
            }
        elif action_norm in ("decompose", "wavelet_bandpower"):
            samples = params.get("samples")
            if samples is not None:
                sig = np.array(samples, dtype=np.float64)
                freq = 10.0
            else:
                freq = float(params.get("frequency_hz", 10.0))
                t = np.linspace(0, 1.0, 250, endpoint=False)
                sig = 20.0 * np.sin(2.0 * np.pi * freq * t)
            bp = self.health_decomposer.decompose(sig)
            return {
                "success": True,
                "frequency_hz": freq,
                "delta_power": bp.delta_power,
                "theta_power": bp.theta_power,
                "alpha_power": bp.alpha_power,
                "beta_power": bp.beta_power,
                "gamma_power": bp.gamma_power,
                "focus_ratio": bp.focus_ratio,
                "calmness_ratio": bp.calmness_ratio,
            }
        elif action_norm in ("decode", "decode_bci"):
            sig = params.get("signal", params.get("samples"))
            if sig is None:
                # Default 10 Hz alpha rhythm
                t = np.linspace(0, 1.0, 250, endpoint=False)
                sig = (25.0 * np.sin(2.0 * np.pi * 10.0 * t)).tolist()
            res = self.health_decoder.decode_window(np.array(sig, dtype=np.float64))
            return {
                "success": True,
                "predicted_state": res.predicted_state.value,
                "confidence": res.confidence,
                "lyapunov_energy": res.lyapunov_energy,
                "non_clinical_disclaimer": res.non_clinical_disclaimer,
            }
        else:
            raise ValueError(f"Unknown health action: {action}")

    def execute_space(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CORTEX-Space workflow."""
        action_norm = action.lower().strip()
        if action_norm in ("propagate", "propagate_orbit"):
            delta_t_s = float(params.get("delta_t_s", params.get("propagation_time_seconds", 600.0)))
            pt = self.space_propagator.propagate_state(delta_t_s)
            return {
                "success": True,
                "timestamp_s": pt.timestamp_s,
                "latitude_deg": pt.latitude_deg,
                "longitude_deg": pt.longitude_deg,
                "altitude_km": pt.altitude_km,
                "velocity_km_s": pt.velocity_km_s,
                "in_eclipse": pt.in_eclipse,
                "station_contacts": pt.station_contacts,
                "orbital_period_s": self.space_propagator.period_s,
            }
        elif action_norm in ("sync_ccsds", "sync"):
            raw_bytes = params.get("raw_bytes")
            if raw_bytes is not None:
                packet = bytes(raw_bytes)
            else:
                packet = self.space_ccsds.build_telemetry_packet(
                    apid=int(params.get("apid", 100)),
                    seq_count=int(params.get("seq_count", 1)),
                    timestamp_s=float(params.get("timestamp_s", 1700000000.0)),
                    bus_voltage=float(params.get("bus_voltage", 28.0)),
                    solar_current=float(params.get("solar_current", 6.5)),
                    battery_temp=float(params.get("battery_temp", 22.0)),
                    gyro_x=float(params.get("gyro_x", 0.0)),
                    gyro_y=float(params.get("gyro_y", 0.0)),
                    gyro_z=float(params.get("gyro_z", 0.0)),
                    heater_on=bool(params.get("heater_on", False)),
                    radiator_temp=float(params.get("radiator_temp", 15.0)),
                )
            telem_list, _ = self.space_ccsds.process_stream(packet)
            if telem_list:
                telem = telem_list[0]
                return {
                    "success": True,
                    "apid": telem.apid,
                    "seq_count": telem.sequence_count,
                    "bus_voltage_v": telem.bus_voltage_v,
                    "solar_current_a": telem.solar_current_a,
                    "crc_valid": telem.crc_valid,
                    "packet_bytes_length": len(packet),
                }
            return {"success": False, "error": "Frame sync failed"}
        elif action_norm in ("seu_check", "supervise_telemetry", "supervise"):
            telem = DecommutatedTelemetry(
                apid=int(params.get("apid", 100)),
                sequence_count=int(params.get("seq_count", 1)),
                timestamp_seconds=float(params.get("timestamp_s", 1700000000.0)),
                bus_voltage_v=float(params.get("bus_voltage", 28.0)),
                solar_current_a=float(params.get("solar_current", 6.5)),
                battery_temp_c=float(params.get("battery_temp", 22.0)),
                gyro_rate_x=float(params.get("gyro_x", 0.0)),
                gyro_rate_y=float(params.get("gyro_y", 0.0)),
                gyro_rate_z=float(params.get("gyro_z", 0.0)),
                heater_on=bool(params.get("heater_on", False)),
                radiator_temp_c=float(params.get("radiator_temp", 15.0)),
                crc_valid=True,
            )
            in_eclipse = bool(params.get("in_eclipse", False))
            alert = self.space_supervisor.evaluate_telemetry(telem, in_eclipse=in_eclipse)
            return {
                "success": True,
                "is_anomalous": alert is not None,
                "alert": {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity.value,
                    "subsystem": alert.subsystem,
                    "message": alert.message,
                    "mitigation_action": alert.mitigation_action,
                    "traceability_req_id": alert.traceability_req_id,
                } if alert else None,
            }
        elif action_norm in ("predict_passes", "passes"):
            dur = float(params.get("duration_s", 86400.0))
            step = float(params.get("step_s", 300.0))
            passes = self.space_propagator.predict_station_passes(duration_s=dur, step_s=step)
            return {
                "success": True,
                "duration_seconds": dur,
                "step_seconds": step,
                "num_passes": len(passes),
                "passes": passes[:10],
            }
        elif action_norm in ("stream_continuity", "continuity"):
            packets = params.get("packets", [])
            metrics = self.space_ccsds.analyze_stream_continuity(packets)
            return {
                "success": True,
                "metrics": metrics,
            }
        else:
            raise ValueError(f"Unknown space action: {action}")

    def execute_os(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CORTEX-OS workflow."""
        action_norm = action.lower().strip()
        if action_norm in ("omnibar", "search"):
            query = params.get("query", "satellite telemetry")
            candidates = []
            docs_data = params.get("documents", [
                {"id": "doc1", "path": "space.py", "text": "satellite telemetry orbit j2"},
                {"id": "doc2", "path": "health.py", "text": "eeg biosignal notch filter"},
            ])
            for d in docs_data:
                vec = self.os_watcher.extract_gist_vector(d["text"], d["path"])
                candidates.append(FileDocument(
                    file_id=d["id"],
                    path=d["path"],
                    extension=".py",
                    file_size_bytes=len(d["text"]),
                    content_preview=d["text"],
                    vector_64d=vec,
                    last_modified_timestamp=1700000000.0,
                ))
            resp = self.os_omnibar.query(query, candidates, top_k=int(params.get("top_k", 3)))
            return {
                "success": True,
                "query": resp.query,
                "evaluation_latency_ms": resp.evaluation_latency_ms,
                "results": [
                    {
                        "rank": r.rank,
                        "file_id": r.file_id,
                        "file_path": r.file_path,
                        "resonance_score": r.resonance_score,
                    }
                    for r in resp.results
                ],
            }
        elif action_norm in ("monitor", "resources"):
            snapshot = self.os_monitor.inspect_resources([])
            return {
                "success": True,
                "status_summary": snapshot.status_summary,
                "is_non_privileged": snapshot.is_non_privileged,
                "memory_ceiling_exceeded": snapshot.memory_ceiling_exceeded,
                "estimated_memory_kb": snapshot.estimated_total_memory_kb,
            }
        else:
            raise ValueError(f"Unknown os action: {action}")

    def execute_learning(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CORTEX Developmental Learning & Synaptic Grafting."""
        action_norm = action.lower().strip()
        if action_norm in ("sleep", "consolidate"):
            dim = int(params.get("dim", 64))
            weights = torch.randn(dim, dim)
            weights = 0.5 * (weights + weights.t())
            weights.fill_diagonal_(0.0)
            prototypes = torch.randn(4, dim)
            w_cons, report = self.learning_consolidator.consolidate(weights, prototypes)
            return {
                "success": True,
                "replayed_attractors": report.replayed_attractors,
                "pruned_synapses": report.pruned_synapses,
                "initial_sparsity": report.initial_sparsity,
                "final_sparsity": report.final_sparsity,
                "retention_accuracy": report.retention_accuracy,
                "cycle_duration_ms": report.cycle_duration_ms,
            }
        elif action_norm in ("graft_filter", "graft"):
            dim = int(params.get("dim", 32))
            packager = SynapticKnowledgePackager(dim=dim)
            p_rec = torch.randn(2, dim)
            p_donor = torch.randn(2, dim)
            manifest, delta_W = packager.extract_package("test_domain", "donor_01", p_donor)
            w_rec = torch.zeros(dim, dim)
            w_new, res = packager.graft_package(w_rec, p_rec, manifest, delta_W, p_donor)
            return {
                "success": res.graft_success,
                "collateral_damage_percentage": res.collateral_damage_rate * 100.0,
                "zero_damage_verified": res.collateral_damage_rate == 0.0,
                "orthogonality_loss": res.orthogonality_loss,
            }
        elif action_norm in ("compress_synapses", "compress"):
            target_rank = int(params.get("target_rank", 16))
            res = self.cortex.compress_synapses(target_rank=target_rank)
            return {"success": True, **res}
        elif action_norm in ("counterfactual_audit", "audit"):
            cue = params.get("cue", "attractor memory")
            num_pert = int(params.get("num_perturbations", 5))
            eps = float(params.get("epsilon", 0.08))
            res = self.cortex.counterfactual_audit(cue, num_perturbations=num_pert, perturbation_epsilon=eps)
            return {"success": True, **res}
        elif action_norm in ("adaptive_recall", "recall"):
            cue = params.get("cue", "attractor memory")
            res = self.cortex.adaptive_recall(cue)
            return {
                "success": True,
                "diagnostics": res.get("diagnostics", {}),
            }
        elif action_norm in ("batch_recall", "batch"):
            cues = params.get("cues", ["sample1", "sample2"])
            res = self.cortex.batch_recall(cues)
            return {
                "success": True,
                "batch_size": len(cues),
                "results": res,
            }
        elif action_norm in ("energy_landscape", "landscape"):
            cue = params.get("cue", "sample")
            res = self.cortex.map_energy_landscape(cue)
            return {"success": True, **res}
        elif action_norm in ("reasoning_chain", "chain"):
            cue = params.get("cue", "sample")
            hops = int(params.get("max_hops", 3))
            res = self.cortex.associative_reasoning_chain(cue, max_hops=hops)
            return {"success": True, **res}
        elif action_norm in ("fuse_memories", "fuse"):
            m_ids = params.get("memory_ids", ["m1", "m2"])
            fused_id = params.get("fused_id", "fused_prototype")
            res = self.cortex.fuse_memories(m_ids, fused_id)
            return {"success": True, **res}
        else:
            raise ValueError(f"Unknown learning action: {action}")

