"""Global Universal CLI Entry Points for CORTEX-M Ecosystem.

Provides unified command-line interface entry points for all 8 product platforms,
developmental learning engines, and Studio services:
- cortex studio            -> Launch Cortex Studio Web App
- cortex code              -> Autonomous Multi-Agent Coding Swarm
- cortex personal          -> CORTEX-Personal Vault & Substrate
- cortex ide               -> CORTEX-IDE AST Audit, Diff & Sandbox
- cortex fin               -> CORTEX-Fin Lyapunov Anomaly Scorer & Replay
- cortex robotics          -> CORTEX-Robotics 6-DOF Kinematics, Safety Interlock & Path Planner
- cortex mesh              -> CORTEX-Mesh PBFT Consensus & Swarm Router
- cortex health            -> CORTEX-Health Biosignal Filter, Wavelet & BCI Decoder
- cortex space             -> CORTEX-Space CCSDS Synchronizer, Orbit Propagator & SEU Supervisor
- cortex os                -> CORTEX-OS Spotlight Omnibar & Resource Monitor
- cortex learning          -> Developmental Sleep Consolidation & SVD Synaptic Grafting
- cortex status / overview -> Full Ecosystem Health & Readiness Overview
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure src is always in sys.path
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from cortex_m.gateway import EcosystemGateway


def cortex_studio_main():
    """Launch Cortex Studio Web Application Server."""
    from cortex_m.server.app import start_cortex_studio
    
    parser = argparse.ArgumentParser(prog="cortex studio", description="Cortex Studio Web Application Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind server (default: 8888)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args(sys.argv[1:])

    start_cortex_studio(host=args.host, port=args.port, open_browser=not args.no_browser)


def cortex_code_main():
    """Autonomous Multi-Agent Coding Swarm."""
    from cortex_m.agents.task_completer import MultiAgentTaskCompleter

    parser = argparse.ArgumentParser(prog="cortex code", description="Autonomous Multi-Agent Coding Swarm")
    parser.add_argument("prompt", nargs="*", help="Goal or task description for the coding swarm")
    parser.add_argument("--auto-approve", action="store_true", help="Automatically approve safe actions")
    args = parser.parse_args(sys.argv[1:])

    prompt = " ".join(args.prompt).strip() or "Inspect active workspace and report status."
    print(f"\n[CORTEX-IDE Swarm] Task: {prompt}")
    completer = MultiAgentTaskCompleter()
    report = completer.complete_task(prompt)
    print(f"Status: {report.status}")
    print(f"Summary: {report.summary}")


def cortex_substrate_main():
    """Biological Synaptic Substrate CLI."""
    gateway = EcosystemGateway.get_instance()
    parser = argparse.ArgumentParser(prog="cortex substrate", description="Biological Synaptic Memory CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Substrate command")

    learn_p = subparsers.add_parser("learn", help="Imprint text into synaptic memory")
    learn_p.add_argument("text", help="Text to learn")
    learn_p.add_argument("--tenant", default="default_user", help="Tenant ID")

    recall_p = subparsers.add_parser("recall", help="Recall text from synaptic memory")
    recall_p.add_argument("query", help="Query to recall")
    recall_p.add_argument("--tenant", default="default_user", help="Tenant ID")

    sleep_p = subparsers.add_parser("sleep", help="Run nocturnal sleep consolidation cycle")

    args = parser.parse_args(sys.argv[1:])

    if args.subcommand == "learn":
        res = gateway.execute_personal("learn", {"text": args.text, "tenant_id": args.tenant})
        print(f"[Substrate Learn] Tenant: {args.tenant} -> Learned: '{args.text}'")
    elif args.subcommand == "recall":
        res = gateway.execute_personal("recall", {"query": args.query, "tenant_id": args.tenant})
        print(f"[Substrate Recall] Status: {res['status']} | Confidence: {res['confidence']:.2f}")
        print(f"Result: {res.get('recalled_text', 'No memory found.')}")
    elif args.subcommand == "sleep":
        res = gateway.execute_learning("sleep", {})
        replayed = res.get("replayed_attractors", res.get("slow_wave_replay_cycles", 0))
        pruned = res.get("pruned_synapses", res.get("synaptic_weights_pruned", 0))
        retention = res.get("retention_accuracy", 1.0) * 100.0
        print(f"[Substrate Sleep] Replayed attractors: {replayed} | Pruned synapses: {pruned} | Retention: {retention:.1f}%")
    else:
        parser.print_help()


def cortex_lm_main():
    """Interactive Terminal Chat & Synaptic Exploration Interface for CortexLM."""
    from pathlib import Path
    chat_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "cortex_lm_chat.py"
    if chat_script.exists():
        import runpy
        runpy.run_path(str(chat_script), run_name="__main__")
    else:
        from cortex_m import CortexLM
        lm = CortexLM(dim=64, gist_dim=32, imprint_foundation=True)
        print("\n[CortexLM] Interactive session started. Type 'exit' to quit.\n")
        while True:
            try:
                inp = input("cortex> ").strip()
                if not inp or inp.lower() in ("exit", "quit"):
                    break
                out = lm.chat(inp)
                print(f"CortexLM: {out.text}\n")
            except (KeyboardInterrupt, EOFError):
                break


def cortex_platform_cli():
    """Universal platform subcommands router."""
    gateway = EcosystemGateway.get_instance()
    platform = sys.argv[1].lower()
    args_rest = sys.argv[2:]

    if platform in ("status", "overview", "info"):
        overview = gateway.get_ecosystem_overview()
        print(f"\n========================================================================")
        print(f" CORTEX-M ECOSYSTEM STATUS: {overview['status']} (Version {overview['version']})")
        print(f"========================================================================")
        for name, plat in overview["platforms"].items():
            features = ", ".join(plat.get("features", []))
            print(f" * {plat['identity']:<28} [{plat['status']}] - Features: {features}")
        return

    if platform == "personal":
        parser = argparse.ArgumentParser(prog="cortex personal")
        sub = parser.add_subparsers(dest="action")
        p_learn = sub.add_parser("learn")
        p_learn.add_argument("text")
        p_learn.add_argument("--tenant", default="default_user")
        p_rec = sub.add_parser("recall")
        p_rec.add_argument("query")
        p_rec.add_argument("--tenant", default="default_user")
        p_lock = sub.add_parser("interlock")
        p_lock.add_argument("--host", default="127.0.0.1")
        p_lock.add_argument("--port", type=int, default=8888)
        p_po = sub.add_parser("page-out")
        p_po.add_argument("--tenant", default="default_user")
        p_po.add_argument("--path", default="tenant_slice.pt")
        p_pi = sub.add_parser("page-in")
        p_pi.add_argument("--path", default="tenant_slice.pt")
        sub.add_parser("footprint")
        parsed = parser.parse_args(args_rest)

        if parsed.action == "learn":
            r = gateway.execute_personal("learn", {"text": parsed.text, "tenant_id": parsed.tenant})
            print(f"[Personal] Imprinted into tenant '{parsed.tenant}' successfully.")
        elif parsed.action == "recall":
            r = gateway.execute_personal("recall", {"query": parsed.query, "tenant_id": parsed.tenant})
            print(f"[Personal] Status: {r['status']} | Confidence: {r['confidence']:.2f} | Energy: {r.get('energy', 0.0):.2f}")
            print(f"Recalled: {r['recalled_text']}")
        elif parsed.action == "interlock":
            r = gateway.execute_personal("interlock", {"host": parsed.host, "port": parsed.port})
            print(f"[Personal Interlock] Host: {parsed.host}:{parsed.port} -> Permitted: {r['egress_permitted']}")
        elif parsed.action == "page-out":
            r = gateway.execute_personal("page_out", {"tenant_id": parsed.tenant, "path": parsed.path})
            print(f"[Personal Paging] Tenant '{r['tenant_id']}' low-rank factor slice paged out to {r['path']}")
        elif parsed.action == "page-in":
            r = gateway.execute_personal("page_in", {"path": parsed.path})
            print(f"[Personal Paging] Loaded tenant factor slice '{r['tenant_id']}' from {r['path']}")
        elif parsed.action == "footprint":
            r = gateway.execute_personal("footprint", {})
            print(f"[Personal Footprint] Active tenants: {r.get('active_tenants', 0)} | Total Footprint: {r.get('total_bytes', 0)} bytes")
        else:
            parser.print_help()

    elif platform == "ide":
        parser = argparse.ArgumentParser(prog="cortex ide")
        sub = parser.add_subparsers(dest="action")
        p_aud = sub.add_parser("audit")
        p_aud.add_argument("file_or_code", help="Path to Python file or raw code")
        p_diff = sub.add_parser("diff")
        p_diff.add_argument("file1")
        p_diff.add_argument("file2")
        p_sand = sub.add_parser("sandbox")
        p_sand.add_argument("cmd", nargs=argparse.REMAINDER)
        p_pat = sub.add_parser("atomic-patch")
        p_pat.add_argument("bundle_json", help="JSON dictionary of filepath -> content")
        parsed, remaining = parser.parse_known_args(args_rest)
        if parsed.action == "sandbox" and remaining:
            parsed.cmd = (parsed.cmd or []) + remaining

        if parsed.action == "audit":
            p = Path(parsed.file_or_code)
            code = p.read_text(encoding="utf-8") if p.exists() else parsed.file_or_code
            r = gateway.execute_ide("audit", {"code": code})
            print(f"[IDE Audit] Clean: {r['is_clean']} | Classes: {r['classes']} | Functions: {r['functions']}")
            if r.get("functions_meta"):
                print(f"Cyclomatic Complexities: { {k: v['cyclomatic_complexity'] for k, v in r['functions_meta'].items()} }")
            if r["security_flaws"]:
                print(f"Security Flaws: {r['security_flaws']}")
        elif parsed.action == "diff":
            c1 = Path(parsed.file1).read_text(encoding="utf-8")
            c2 = Path(parsed.file2).read_text(encoding="utf-8")
            r = gateway.execute_ide("diff", {"original": c1, "modified": c2, "filename": parsed.file2})
            print(f"[IDE Diff] Syntax Valid: {r['syntax_valid']} | Line Changes: {r['line_changes']}")
            print(r["unified_diff"])
        elif parsed.action == "sandbox":
            r = gateway.execute_ide("sandbox", {"command": parsed.cmd})
            print(f"[IDE Sandbox] Exit: {r['exit_code']} | Timed out: {r['timed_out']} | Duration: {r['duration_ms']:.1f}ms")
            print(f"Output:\n{r['stdout']}")
        elif parsed.action == "atomic-patch":
            import json
            bundle = json.loads(parsed.bundle_json)
            r = gateway.execute_ide("atomic_patch", {"bundle": bundle})
            print(f"[IDE Atomic Patch] Success: {r.get('success')} | Committed: {r.get('committed_files')}")
        else:
            parser.print_help()

    elif platform == "fin":
        parser = argparse.ArgumentParser(prog="cortex fin")
        sub = parser.add_subparsers(dest="action")
        p_sc = sub.add_parser("score")
        p_sc.add_argument("--amount", type=float, default=125.0)
        p_sc.add_argument("--distance-km", type=float, default=5.0)
        p_sc.add_argument("--risk-score", type=float, default=0.1)
        p_rep = sub.add_parser("replay")
        p_rep.add_argument("--count", type=int, default=100)
        p_exp = sub.add_parser("explain")
        p_exp.add_argument("--amount", type=float, default=1250.0)
        p_exp.add_argument("--distance-km", type=float, default=450.0)
        p_exp.add_argument("--mcc", type=int, default=7995)
        sub.add_parser("velocity-anomalies")
        parsed = parser.parse_args(args_rest)

        if parsed.action == "score":
            r = gateway.execute_fin("score", {
                "amount": parsed.amount,
                "distance_km": parsed.distance_km,
                "risk_score": parsed.risk_score,
            })
            print(f"[Fin Anomaly] Decision: {r['decision']} | Energy: {r['lyapunov_energy']:.3f} | Calibrated Risk: {r['calibrated_risk']:.3f} | Latency: {r['scoring_latency_us']:.1f}us")
        elif parsed.action == "replay":
            r = gateway.execute_fin("replay", {"count": parsed.count})
            print(f"[Fin Replay] Stream throughput: {r['throughput_tps']:,.0f} tx/s | p50: {r['p50_latency_ms']:.3f}ms | p99: {r['p99_latency_ms']:.3f}ms")
        elif parsed.action == "explain":
            r = gateway.execute_fin("explain", {"amount": parsed.amount, "distance_km": parsed.distance_km, "mcc": parsed.mcc})
            print(f"[Fin Explainability] Decision: {r['decision']} | Risk: {r['risk_score']:.3f}")
            print(f"Adverse Action Codes: {r.get('adverse_action_codes')}")
            print(f"Top Risk Driver: {r.get('top_risk_driver')} (weight {r.get('top_driver_weight')})")
        elif parsed.action == "velocity-anomalies":
            r = gateway.execute_fin("velocity_anomalies", {})
            print(f"[Fin Velocity] Analyzed {r['analyzed_count']} txs | Detected {r['num_anomalies']} supersonic anomalies:")
            for anom in r["anomalies"]:
                print(f"  * Acc {anom['account_id']}: {anom['apparent_velocity_km_h']} km/h ({anom['distance_km']} km in {anom['time_interval_seconds']}s)")
        else:
            parser.print_help()

    elif platform == "robotics":
        parser = argparse.ArgumentParser(prog="cortex robotics")
        sub = parser.add_subparsers(dest="action")
        p_fk = sub.add_parser("kinematics")
        p_fk.add_argument("--joints", default="0,0,0,0,0,0")
        p_pl = sub.add_parser("plan")
        p_pl.add_argument("--start", default="0,0,0,0,0,0")
        p_pl.add_argument("--goal", default="0.5,-0.3,0.8,0,0.2,0")
        p_sf = sub.add_parser("safety")
        p_sf.add_argument("--dist", type=float, default=1.5)
        p_sf.add_argument("--speed", type=float, default=0.4)
        p_ik = sub.add_parser("ik")
        p_ik.add_argument("--target", default="0.4,0.2,0.3")
        p_man = sub.add_parser("manipulability")
        p_man.add_argument("--joints", default="0,0,0,0,0,0")
        parsed = parser.parse_args(args_rest)

        if parsed.action == "kinematics":
            j = [float(x.strip()) for x in parsed.joints.split(",")]
            r = gateway.execute_robotics("kinematics", {"joints": j})
            print(f"[Robotics 6-DOF] Valid: {r['joints_valid']} | EE Pos: {[round(x, 3) for x in r['end_effector_position']]}")
        elif parsed.action == "plan":
            s = [float(x.strip()) for x in parsed.start.split(",")]
            g = [float(x.strip()) for x in parsed.goal.split(",")]
            r = gateway.execute_robotics("plan", {"start": s, "goal": g})
            print(f"[Robotics Planner] C2 Quintic Trajectory Waypoints: {r['num_waypoints']} over {r['duration_seconds']}s")
        elif parsed.action == "safety":
            r = gateway.execute_robotics("safety", {"human_distance_m": parsed.dist, "cartesian_speed_m_s": parsed.speed})
            print(f"[Robotics Safety ISO 10218] Motion Allowed: {r['motion_allowed']} | Max TCP Speed: {r['allowed_tcp_speed_m_s']:.2f} m/s | E-Stop: {r['estop_active']}")
        elif parsed.action == "ik":
            t = [float(x.strip()) for x in parsed.target.split(",")]
            r = gateway.execute_robotics("ik", {"target_xyz": t})
            print(f"[Robotics IK] Converged: {r['success']} | Error: {r['position_error_m']:.4f}m | Joints: {[round(x, 3) for x in r['joint_angles_rad']]}")
        elif parsed.action == "manipulability":
            j = [float(x.strip()) for x in parsed.joints.split(",")]
            r = gateway.execute_robotics("manipulability", {"joints": j})
            print(f"[Robotics Manipulability] Yoshikawa Index w(q): {r['manipulability_index']:.4f}")
        else:
            parser.print_help()

    elif platform == "mesh":
        parser = argparse.ArgumentParser(prog="cortex mesh")
        sub = parser.add_subparsers(dest="action")
        p_cn = sub.add_parser("consensus")
        p_cn.add_argument("--proposal", default="BLOCK_STATE_001")
        p_rt = sub.add_parser("route")
        p_rt.add_argument("--task-id", default="task_001")
        p_vc = sub.add_parser("view-change")
        p_vc.add_argument("--new-view", type=int, default=1)
        p_cl = sub.add_parser("clock")
        p_cl.add_argument("--node-id", default="node_gateway")
        parsed = parser.parse_args(args_rest)

        if parsed.action == "consensus":
            r = gateway.execute_mesh("consensus", {"proposal": parsed.proposal})
            print(f"[Mesh PBFT Consensus] Pre-Prepare: {r['phase']} | Proposal ID: {r['proposal_id']} | Digest: {r['digest'][:16]}")
        elif parsed.action == "route":
            r = gateway.execute_mesh("route", {"task_id": parsed.task_id})
            print(f"[Mesh Swarm Router] Task '{r['task_id']}' -> Allocated Node: {r['assigned_node']}")
        elif parsed.action == "view-change":
            r = gateway.execute_mesh("view_change", {"new_view": parsed.new_view})
            print(f"[Mesh View-Change] Target View: {r['new_view']} | Phase: {r['vote_phase']} | Total Quorum Votes: {r['view_change_votes']}")
        elif parsed.action == "clock":
            r = gateway.execute_mesh("vector_clock", {"node_id": parsed.node_id})
            print(f"[Mesh Vector Clock] Node: {r['node_id']} -> Timestamp: {r['logical_timestamp']} | Clocks: {r['vector_clock']}")
        else:
            parser.print_help()

    elif platform == "health":
        parser = argparse.ArgumentParser(prog="cortex health")
        sub = parser.add_subparsers(dest="action")
        sub.add_parser("filter")
        p_dc = sub.add_parser("decompose")
        p_dc.add_argument("--freq", type=float, default=10.0)
        sub.add_parser("decode")
        sub.add_parser("snr")
        sub.add_parser("leads")
        sub.add_parser("psd")
        p_st = sub.add_parser("stream")
        p_st.add_argument("--channel", default="ch_01")
        parsed = parser.parse_args(args_rest)

        if parsed.action == "filter":
            r = gateway.execute_health("filter", {})
            print(f"[Health Signal Filter] Detrended & 60Hz Notch: Mean {r['mean_uv']:.2f} uV, Amplitude [{r['min_uv']:.1f}, {r['max_uv']:.1f}] uV")
        elif parsed.action == "decompose":
            r = gateway.execute_health("decompose", {"frequency_hz": parsed.freq})
            print(f"[Health Wavelet] Freq {r['frequency_hz']} Hz -> Alpha: {r['alpha_power']:.3f} | Beta: {r['beta_power']:.3f} | Focus Ratio: {r['focus_ratio']:.3f}")
        elif parsed.action == "decode":
            r = gateway.execute_health("decode", {})
            print(f"[Health BCI] State: {r['predicted_state']} | Confidence: {r['confidence']:.2f} | Energy: {r['lyapunov_energy']:.3f}")
            print(f"Notice: {r['non_clinical_disclaimer']}")
        elif parsed.action == "snr":
            r = gateway.execute_health("snr", {})
            print(f"[Health SNR] EEG Signal-to-Noise Ratio: {r['snr_db']} dB | Verified: {r['physiological_signal_verified']}")
        elif parsed.action == "leads":
            r = gateway.execute_health("electrode_check", {})
            print(f"[Health Electrode Impedance] All Leads Good: {r.get('all_leads_contact_good')} | Channels: {r.get('channels')}")
        elif parsed.action == "psd":
            r = gateway.execute_health("welch_psd", {})
            print(f"[Health Welch PSD] Peak Frequency: {r['peak_frequency_hz']} Hz | Peak Power: {r['peak_power']:.2f} uV^2/Hz")
        elif parsed.action == "stream":
            r = gateway.execute_health("decode_stream", {"channel_id": parsed.channel})
            print(f"[Health Stream Decoder] Channel {r['channel_id']} -> State: {r['predicted_state']} (Confidence: {r['confidence']:.2f})")
        else:
            parser.print_help()

    elif platform == "space":
        parser = argparse.ArgumentParser(prog="cortex space")
        sub = parser.add_subparsers(dest="action")
        p_pr = sub.add_parser("propagate")
        p_pr.add_argument("--delta-t", type=float, default=600.0)
        p_sy = sub.add_parser("sync-ccsds")
        p_su = sub.add_parser("seu-check")
        p_su.add_argument("--voltage", type=float, default=28.0)
        p_su.add_argument("--current", type=float, default=6.5)
        p_pa = sub.add_parser("passes")
        p_pa.add_argument("--duration-h", type=float, default=24.0)
        parsed = parser.parse_args(args_rest)

        if parsed.action == "propagate":
            r = gateway.execute_space("propagate", {"delta_t_s": parsed.delta_t})
            print(f"[Space J2 Orbit] Alt: {r['altitude_km']:.1f} km | Lat: {r['latitude_deg']:.2f} | Lon: {r['longitude_deg']:.2f} | Speed: {r['velocity_km_s']:.2f} km/s | Eclipse: {r['in_eclipse']}")
        elif parsed.action == "sync-ccsds":
            r = gateway.execute_space("sync_ccsds", {})
            print(f"[Space CCSDS Sync] Frame ASM 0x1ACFFC1D Locked | APID: {r['apid']} | CRC-16 Valid: {r['crc_valid']}")
        elif parsed.action == "seu-check" or parsed.action == "seu_check":
            r = gateway.execute_space("seu_check", {"bus_voltage": parsed.voltage, "solar_current": parsed.current})
            print(f"[Space SEU Supervisor] Anomalous: {r['is_anomalous']} | Status: {'NOMINAL' if not r['is_anomalous'] else r['alert']['severity']}")
        elif parsed.action == "passes":
            r = gateway.execute_space("predict_passes", {"duration_s": parsed.duration_h * 3600.0})
            print(f"[Space Station Passes] Predicted {r['num_passes']} passes over {parsed.duration_h} hours:")
            for p in r["passes"][:5]:
                print(f"  * {p['station_name']} at {p['timestamp_s']}s (AOS: {p['elevation_deg']:.1f} deg)")
        else:
            parser.print_help()

    elif platform == "os":
        parser = argparse.ArgumentParser(prog="cortex os")
        sub = parser.add_subparsers(dest="action")
        p_om = sub.add_parser("omnibar")
        p_om.add_argument("query")
        sub.add_parser("monitor")
        sub.add_parser("cluster")
        p_cl = sub.add_parser("classify")
        p_cl.add_argument("text")
        parsed = parser.parse_args(args_rest)

        if parsed.action == "omnibar":
            r = gateway.execute_os("omnibar", {"query": parsed.query})
            print(f"[OS Spotlight Omnibar] Query: '{r['query']}' evaluated in {r['evaluation_latency_ms']:.2f}ms:")
            for res in r["results"]:
                print(f"  #{res['rank']} {res['file_path']} (Resonance: {res['resonance_score']:.3f})")
        elif parsed.action == "monitor":
            r = gateway.execute_os("monitor", {})
            print(f"[OS Monitor] {r['status_summary']} | Unprivileged: {r['is_non_privileged']}")
        elif parsed.action == "cluster":
            r = gateway.execute_os("cluster", {})
            print(f"[OS Affinity Graph] Extracted {r['num_clusters']} clusters (Modularity Q: {r['newman_girvan_modularity']:.4f}, Density: {r['graph_density']:.3f}):")
            for c in r["clusters"]:
                print(f"  * {c['cluster_name']}: {len(c['document_paths'])} docs (Cohesion: {c['cohesion_score']:.3f})")
        elif parsed.action == "classify":
            r = gateway.execute_os("predict_cluster", {"text": parsed.text})
            print(f"[OS Classifier] Text '{parsed.text}' -> Best Cluster: {r['best_cluster_id']} (Resonance: {r['resonance_score']:.3f})")
        else:
            parser.print_help()

    elif platform == "learning":
        parser = argparse.ArgumentParser(prog="cortex learning")
        sub = parser.add_subparsers(dest="action")
        sub.add_parser("sleep")
        sub.add_parser("graft")
        p_cmp = sub.add_parser("compress")
        p_cmp.add_argument("--rank", type=int, default=16)
        p_ls = sub.add_parser("landscape")
        p_ls.add_argument("--cue", default="memory_anchor")
        p_ch = sub.add_parser("chain")
        p_ch.add_argument("--cue", default="concept_anchor")
        p_ch.add_argument("--hops", type=int, default=3)
        parsed = parser.parse_args(args_rest)

        if parsed.action == "sleep":
            r = gateway.execute_learning("sleep", {})
            print(f"[Learning Sleep] Replayed attractors: {r['replayed_attractors']} | Retention: {r['retention_accuracy'] * 100:.1f}% | Pruned: {r['pruned_synapses']}")
        elif parsed.action == "graft":
            r = gateway.execute_learning("graft_filter", {})
            print(f"[Learning Grafting] Zero collateral damage: {r['zero_damage_verified']} (Damage: {r['collateral_damage_percentage']}%)")
        elif parsed.action == "compress":
            r = gateway.execute_learning("compress_synapses", {"target_rank": parsed.rank})
            print(f"[Learning Compression] Synaptic Low-Rank SVD: Rank {r.get('target_rank')} | Ratio: {r.get('compression_ratio_pct', 0):.1f}%")
        elif parsed.action == "landscape":
            r = gateway.execute_learning("energy_landscape", {"cue": parsed.cue})
            print(f"[Learning Landscape] Energy Barrier: {r.get('energy_barrier_height', 0):.4f} | Attractor Margin: {r.get('attractor_margin', 0):.4f}")
        elif parsed.action == "chain":
            r = gateway.execute_learning("reasoning_chain", {"cue": parsed.cue, "max_hops": parsed.hops})
            print(f"[Learning Chain] Hops: {r.get('num_hops', 0)} | Trajectory: {[h.get('recalled_memory_id') for h in r.get('chain', [])]}")
        else:
            parser.print_help()

    else:
        print(f"Error: Unknown platform or command '{platform}'. Run 'cortex --help' for usage.")


def cortex_unified_main():
    """Universal smart dispatcher for the `cortex` CLI."""
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd in ("--help", "-h", "help"):
            print("""
========================================================================
 CORTEX-M: Universal Neural Memory, Multi-Agent Swarm & Platform Suite
========================================================================

General Commands:
  cortex studio              Launch Cortex Studio Web Application
  cortex code [prompt]       Autonomous Multi-Agent Coding Swarm
  cortex substrate           Biological Synaptic Memory CLI
  cortex status              Ecosystem Status & Platform Readiness
  cortex --help              Show this help message

Product Platforms:
  cortex personal            CORTEX-Personal Vault & Egress Interlock
  cortex ide                 CORTEX-IDE AST Audit, Diff & Sandbox
  cortex fin                 CORTEX-Fin Anomaly Scorer & Replay
  cortex robotics            CORTEX-Robotics 6-DOF Kinematics & Safety Interlock
  cortex mesh                CORTEX-Mesh PBFT Consensus & Swarm Router
  cortex health              CORTEX-Health Biosignal Filter & BCI Decoder
  cortex space               CORTEX-Space CCSDS Synchronizer & Orbit Propagator
  cortex os                  CORTEX-OS Spotlight Omnibar & Resource Monitor
  cortex learning            Developmental Sleep Consolidation & SVD Grafting
""")
            return
        elif cmd == "studio":
            sys.argv.pop(1)
            cortex_studio_main()
            return
        elif cmd in ("code", "cortex-code", "cortexcode"):
            sys.argv.pop(1)
            cortex_code_main()
            return
        elif cmd in ("substrate", "memory"):
            sys.argv.pop(1)
            cortex_substrate_main()
            return
        elif cmd in ("lm", "chat", "cortexlm", "cortex-lm"):
            sys.argv.pop(1)
            cortex_lm_main()
            return
        elif cmd in (
            "status", "overview", "info",
            "personal", "ide", "fin", "robotics",
            "mesh", "health", "space", "os", "learning",
        ):
            cortex_platform_cli()
            return

    # Default to overview if no args
    cortex_platform_cli()


if __name__ == "__main__":
    cortex_unified_main()
