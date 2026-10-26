"""Continuous Hopfield BCI Decoder & Cognitive State Attribution for CORTEX-Health.

Decodes preprocessed EEG windows into discrete cognitive / motor imagery states:
1. REST / BASELINE
2. MOTOR_LEFT_HAND
3. MOTOR_RIGHT_HAND
4. HIGH_FOCUS / ATTENTIVE
5. DROWSY / FATIGUED

Enforces strict non-clinical research disclaimer on all outputs.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .signal_filter import BiosignalFilter, FilteredSignal
from .wavelet_analysis import NeuralBandpower, WaveletFrequencyDecomposer


class CognitiveState(str, enum.Enum):
    REST = "REST"
    MOTOR_LEFT_HAND = "MOTOR_LEFT_HAND"
    MOTOR_RIGHT_HAND = "MOTOR_RIGHT_HAND"
    HIGH_FOCUS = "HIGH_FOCUS"
    DROWSY = "DROWSY"


@dataclass
class BCIDecodingResult:
    """BCI classification outcome with non-clinical disclaimer."""
    predicted_state: CognitiveState
    confidence: float  # [0.0, 1.0]
    lyapunov_energy: float
    bandpower: NeuralBandpower
    decoding_latency_ms: float
    non_clinical_disclaimer: str = (
        "NON-CLINICAL RESEARCH ONLY: This output is strictly engineered for laboratory BCI "
        "and HCI research. Explicitly prohibited for diagnostic, surgical, or clinical use."
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContinuousBCIDecoder:
    """Classifies cognitive states using continuous Hopfield attractor energy basins."""

    FEATURE_DIM = 5  # [Delta, Theta, Alpha, Beta, Gamma] normalized relative powers

    def __init__(self, beta: float = 8.0, device: str = "cpu"):
        self.beta = beta
        self.device = device
        self.filter = BiosignalFilter()
        self.decomposer = WaveletFrequencyDecomposer()

        # Prototype attractor vectors for the 5 states in 5-D spectral space
        # [Delta, Theta, Alpha, Beta, Gamma]
        self._attractor_map = {
            CognitiveState.REST: torch.tensor([0.15, 0.20, 0.45, 0.15, 0.05], dtype=torch.float32),
            CognitiveState.MOTOR_LEFT_HAND: torch.tensor([0.10, 0.15, 0.20, 0.40, 0.15], dtype=torch.float32),
            CognitiveState.MOTOR_RIGHT_HAND: torch.tensor([0.10, 0.15, 0.20, 0.35, 0.20], dtype=torch.float32),
            CognitiveState.HIGH_FOCUS: torch.tensor([0.05, 0.10, 0.15, 0.50, 0.20], dtype=torch.float32),
            CognitiveState.DROWSY: torch.tensor([0.45, 0.35, 0.10, 0.05, 0.05], dtype=torch.float32),
        }
        # Normalize prototypes to unit hypersphere
        self._states = list(self._attractor_map.keys())
        stacked = torch.stack([F.normalize(self._attractor_map[s], p=2, dim=0) for s in self._states])
        self.prototypes = stacked.to(device)  # (5, 5)
        self._ema_probabilities: Optional[torch.Tensor] = None

    def reset_stream(self) -> None:
        """Reset historical temporal EMA smoothing buffer."""
        self._ema_probabilities = None

    def decode_window(self, raw_eeg_channel_1d: np.ndarray) -> BCIDecodingResult:
        """Filter, decompose, and decode a 1-second EEG window into a cognitive state."""
        start_time = time.perf_counter()

        # 1. Filter
        filtered = self.filter.filter_signal(raw_eeg_channel_1d)
        clean_channel = filtered.clean_microvolts[0]

        # 2. Decompose into spectral bandpowers
        bp = self.decomposer.decompose(clean_channel)
        total = max(1e-6, bp.total_power)
        feat_vec = torch.tensor(
            [
                bp.delta_power / total,
                bp.theta_power / total,
                bp.alpha_power / total,
                bp.beta_power / total,
                bp.gamma_power / total,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        normalized_cue = F.normalize(feat_vec, p=2, dim=0)

        # 3. Continuous Hopfield attractor projection: scores = M * x
        scores = torch.mv(self.prototypes, normalized_cue)
        probs = F.softmax(self.beta * scores, dim=0)
        best_prob, best_idx = probs.max(dim=0)

        # Compute continuous Hopfield Lyapunov energy: E = - (1/beta)*logsumexp(beta * scores)
        lse = torch.logsumexp(self.beta * scores, dim=0) / self.beta
        quadratic = 0.5 * torch.dot(normalized_cue, normalized_cue)
        energy = float((quadratic - lse).item())

        predicted_state = self._states[int(best_idx.item())]
        confidence = float(best_prob.item())
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return BCIDecodingResult(
            predicted_state=predicted_state,
            confidence=round(confidence, 4),
            lyapunov_energy=round(energy, 4),
            bandpower=bp,
            decoding_latency_ms=round(duration_ms, 3),
        )

    def decode_stream_window(
        self,
        raw_eeg_channel_1d: np.ndarray,
        ema_alpha: float = 0.35,
    ) -> BCIDecodingResult:
        """Decode streaming EEG window with Exponential Moving Average temporal stabilization.
        
        Prevents high-frequency classification jitter across sequential sliding windows:
            P_t = alpha * P_inst + (1 - alpha) * P_{t-1}
        """
        inst_res = self.decode_window(raw_eeg_channel_1d)

        # Convert instantaneous prediction back to state probability vector
        inst_probs = torch.zeros(len(self._states), device=self.device)
        for i, s in enumerate(self._states):
            if s == inst_res.predicted_state:
                inst_probs[i] = inst_res.confidence
            else:
                inst_probs[i] = (1.0 - inst_res.confidence) / max(1, len(self._states) - 1)

        if self._ema_probabilities is None:
            self._ema_probabilities = inst_probs
        else:
            self._ema_probabilities = ema_alpha * inst_probs + (1.0 - ema_alpha) * self._ema_probabilities

        smoothed_prob, best_idx = self._ema_probabilities.max(dim=0)
        smoothed_state = self._states[int(best_idx.item())]

        inst_res.predicted_state = smoothed_state
        inst_res.confidence = round(float(smoothed_prob.item()), 4)
        inst_res.metadata["smoothed_via_ema"] = True
        inst_res.metadata["ema_alpha"] = ema_alpha
        return inst_res
