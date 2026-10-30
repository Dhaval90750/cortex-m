"""CORTEX-M Synthetic Data Factory (CM-SCI, CM-GIST, CM-MEM).

Generates grounded scientific discovery episodes, multi-turn gist state drift,
and associative episodic memory records for imprinting into continuous Hopfield dynamics.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DiscoveryEpisode:
    """A self-contained scientific discovery episode with hidden ground-truth rule."""
    episode_id: str
    domain: str
    ground_truth_rule: str
    initial_observations: List[Dict[str, float]]
    candidate_hypotheses: List[Dict[str, str]]
    conducted_experiments: List[Dict[str, Any]]
    falsified_hypotheses: List[str]
    retained_hypotheses: List[str]
    final_law_claim: str
    confidence: float
    verification_passed: bool


class CMSciSimulator:
    """CM-SCI-001: Hidden Physical Law Discovery Simulator.

    Simulates kinematic and orbital dynamics with hidden physics rules,
    generates sparse observations, interventions, and falsification signals.
    """

    RULES = [
        ("constant_velocity", lambda t, v0, a: v0 * t, "Position x(t) = v0 * t (uniform zero-acceleration motion)"),
        ("uniform_acceleration", lambda t, v0, a: v0 * t + 0.5 * a * (t ** 2), "Position x(t) = v0*t + 0.5*a*t^2 (constant gravitational acceleration)"),
        ("damped_harmonic", lambda t, v0, a: v0 * np.exp(-0.2 * t) * np.cos(1.5 * t), "Position x(t) = v0 * exp(-gamma*t) * cos(omega*t) (damped oscillatory motion)"),
        ("inverse_square_orbit", lambda t, v0, a: (v0 ** 2) / (1.0 + 0.1 * np.cos(t)), "Orbital radius r(theta) = h^2 / (mu * (1 + e*cos(theta))) (Keplerian conic)"),
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate_episode(self, episode_idx: int) -> DiscoveryEpisode:
        rule_name, sim_fn, formula_claim = self.rng.choice(self.RULES)
        v0 = round(self.rng.uniform(1.0, 10.0), 2)
        a = round(self.rng.uniform(1.0, 9.8), 2)

        # 1. Sparse initial observations at t in [1.0, 2.0, 3.0]
        obs_times = [1.0, 2.0, 3.0]
        initial_obs = [
            {"t": t, "x": float(np.round(sim_fn(t, v0, a) + self.np_rng.normal(0, 0.01), 3))}
            for t in obs_times
        ]

        # 2. Competing candidate hypotheses
        candidates = [
            {"id": "H1", "claim": "Linear constant velocity x(t) = c * t"},
            {"id": "H2", "claim": "Quadratic constant acceleration x(t) = v0*t + 0.5*a*t^2"},
            {"id": "H3", "claim": "Exponential decay damping x(t) = v0*exp(-gamma*t)"},
            {"id": "H4", "claim": "Inverse square orbital attractor r(t) = k / (1 + e*cos(t))"},
        ]

        # 3. Interventions: sample at higher / counterfactual times t=5.0, 8.0
        test_times = [5.0, 8.0]
        experiments = []
        falsified = []
        for t_test in test_times:
            actual_x = float(np.round(sim_fn(t_test, v0, a), 3))
            experiments.append({"t_intervention": t_test, "measured_x": actual_x})

        # Evaluate hypotheses against true rule
        if rule_name == "constant_velocity":
            falsified = ["H2", "H3", "H4"]
            retained = ["H1"]
        elif rule_name == "uniform_acceleration":
            falsified = ["H1", "H3", "H4"]
            retained = ["H2"]
        elif rule_name == "damped_harmonic":
            falsified = ["H1", "H2", "H4"]
            retained = ["H3"]
        else:
            falsified = ["H1", "H2", "H3"]
            retained = ["H4"]

        return DiscoveryEpisode(
            episode_id=f"CM-SCI-{episode_idx:04d}",
            domain="classical_mechanics_and_astrophysics",
            ground_truth_rule=rule_name,
            initial_observations=initial_obs,
            candidate_hypotheses=candidates,
            conducted_experiments=experiments,
            falsified_hypotheses=falsified,
            retained_hypotheses=retained,
            final_law_claim=formula_claim,
            confidence=0.985,
            verification_passed=True,
        )


class CMGistDriftSimulator:
    """CM-GIST-001: Multi-turn State Drift & Contradiction Resolution Generator."""

    SCENARIOS = [
        {
            "topic": "spacecraft_navigation",
            "initial_state": "Trajectory nominal towards Mars transfer orbit, periapsis 300km.",
            "drift_event": "Hydrazine thruster valve micro-leak detected on RCS pod 2.",
            "updated_invariant": "Trajectory adjusted to safe ballistic coast; delta-V budget reduced by 14m/s; science payload uncompromised.",
        },
        {
            "topic": "compiler_optimization",
            "initial_state": "Target arch ARM Cortex-M4 with hardware FPU; aggressive loop unrolling enabled.",
            "drift_event": "SRAM limit constrained to 64KB; code size exceeded cache line boundary.",
            "updated_invariant": "Switched to -Os size optimization; unrolling disabled; critical DSP kernels vectorised via CMSIS-NN.",
        },
        {
            "topic": "portfolio_risk_hedging",
            "initial_state": "Delta-neutral equity long/short book; max drawdown budget 2.5%.",
            "drift_event": "VIX spikes +45% in intraday shock; liquidity dries up in mid-cap options.",
            "updated_invariant": "Liquidated high-gamma exposures; shifted collateral to short-dated US T-bills; VaR bounded strictly < 1.8%.",
        },
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate_drift_episode(self, episode_idx: int) -> Dict[str, Any]:
        scen = self.rng.choice(self.SCENARIOS)
        return {
            "episode_id": f"CM-GIST-{episode_idx:04d}",
            "topic": scen["topic"],
            "initial_belief": scen["initial_state"],
            "perturbation_evidence": scen["drift_event"],
            "synthesized_gist": scen["updated_invariant"],
            "entropy_reduction": 0.42,
            "timestamp": time.time(),
        }


class CMMemoryEpisodeGenerator:
    """CM-MEM-001: Associative Episodic Record Packager.

    Translates simulated discovery and drift events into normalized EpisodicRecords
    ready for Hebbian imprinting into CORTEX-M Continuous Hopfield networks.
    """

    @staticmethod
    def from_discovery_episode(ep: DiscoveryEpisode) -> Dict[str, Any]:
        return {
            "record_id": ep.episode_id,
            "claim": ep.final_law_claim,
            "evidence_type": "Observed",
            "confidence": ep.confidence,
            "context": {
                "domain": ep.domain,
                "ground_truth_rule": ep.ground_truth_rule,
                "falsified": ep.falsified_hypotheses,
                "experiments_count": len(ep.conducted_experiments),
            },
            "tags": [ep.domain, "scientific_law", "synthetic_discovery"],
        }

    @staticmethod
    def from_gist_episode(ep: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "record_id": ep["episode_id"],
            "claim": ep["synthesized_gist"],
            "evidence_type": "Derived",
            "confidence": 0.95,
            "context": {
                "topic": ep["topic"],
                "drift_event": ep["perturbation_evidence"],
            },
            "tags": [ep["topic"], "gist_invariant", "synthetic_drift"],
        }


def generate_discovery_batch(num_episodes: int = 20) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates a combined batch of scientific discovery and gist records."""
    sci_sim = CMSciSimulator(seed=42)
    gist_sim = CMGistDriftSimulator(seed=42)

    sci_records = []
    for i in range(1, num_episodes + 1):
        ep = sci_sim.generate_episode(i)
        sci_records.append(CMMemoryEpisodeGenerator.from_discovery_episode(ep))

    gist_records = []
    for i in range(1, num_episodes + 1):
        g_ep = gist_sim.generate_drift_episode(i)
        gist_records.append(CMMemoryEpisodeGenerator.from_gist_episode(g_ep))

    return sci_records, gist_records

