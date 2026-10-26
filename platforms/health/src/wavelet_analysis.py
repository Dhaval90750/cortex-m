"""Wavelet & Spectral Bandpower Decomposer for CORTEX-Health.

Decomposes neural signals into standard clinical frequency rhythms:
- Delta (0.5 - 4.0 Hz): Deep non-REM sleep & unconscious state.
- Theta (4.0 - 8.0 Hz): Drowsiness, meditation & memory encoding.
- Alpha (8.0 - 13.0 Hz): Relaxed wakefulness & posterior sensory gating.
- Beta (13.0 - 30.0 Hz): Active focus, cognitive alertness & motor execution.
- Gamma (30.0 - 100.0 Hz): High-level feature binding & cognitive integration.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


class FrequencyBand(str, enum.Enum):
    DELTA = "DELTA"
    THETA = "THETA"
    ALPHA = "ALPHA"
    BETA = "BETA"
    GAMMA = "GAMMA"


@dataclass
class NeuralBandpower:
    """Bandpower decomposition metrics across an EEG window."""
    delta_power: float
    theta_power: float
    alpha_power: float
    beta_power: float
    gamma_power: float
    total_power: float
    focus_ratio: float  # Beta / (Theta + Alpha)
    calmness_ratio: float  # Alpha / Beta


class WaveletFrequencyDecomposer:
    """Computes spectral bandpower estimates via discrete Fourier and filterbanks."""

    BANDS = {
        FrequencyBand.DELTA: (0.5, 4.0),
        FrequencyBand.THETA: (4.0, 8.0),
        FrequencyBand.ALPHA: (8.0, 13.0),
        FrequencyBand.BETA: (13.0, 30.0),
        FrequencyBand.GAMMA: (30.0, 100.0),
    }

    def __init__(self, sampling_rate_hz: float = 250.0):
        self.sampling_rate_hz = sampling_rate_hz

    def decompose(self, signal_1d: np.ndarray) -> NeuralBandpower:
        """Decompose a single-channel 1D EEG signal into neural bandpower metrics."""
        x = np.array(signal_1d, dtype=np.float64)
        N = len(x)
        if N < 4:
            return NeuralBandpower(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Apply Hanning window to prevent spectral leakage
        window = np.hanning(N)
        fft_vals = np.fft.rfft(x * window)
        fft_freqs = np.fft.rfftfreq(N, d=1.0 / self.sampling_rate_hz)

        # Power Spectral Density (PSD)
        psd = (np.abs(fft_vals) ** 2) / N

        powers = {}
        for band, (f_low, f_high) in self.BANDS.items():
            idx = np.where((fft_freqs >= f_low) & (fft_freqs < f_high))[0]
            if len(idx) > 0:
                powers[band] = float(np.sum(psd[idx]))
            else:
                powers[band] = 0.0

        p_delta = powers[FrequencyBand.DELTA]
        p_theta = powers[FrequencyBand.THETA]
        p_alpha = powers[FrequencyBand.ALPHA]
        p_beta = powers[FrequencyBand.BETA]
        p_gamma = powers[FrequencyBand.GAMMA]
        total_p = sum([p_delta, p_theta, p_alpha, p_beta, p_gamma])

        # Focus / Attentional Index: Beta / (Theta + Alpha)
        focus = p_beta / max(1e-6, (p_theta + p_alpha))
        calmness = p_alpha / max(1e-6, p_beta)

        return NeuralBandpower(
            delta_power=round(p_delta, 4),
            theta_power=round(p_theta, 4),
            alpha_power=round(p_alpha, 4),
            beta_power=round(p_beta, 4),
            gamma_power=round(p_gamma, 4),
            total_power=round(total_p, 4),
            focus_ratio=round(focus, 4),
            calmness_ratio=round(calmness, 4),
        )

    def welch_psd(
        self,
        signal_1d: np.ndarray,
        window_size: int = 128,
        overlap: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute smoothed Power Spectral Density using Welch's averaged periodogram.
        
        Splits signal into 50% overlapping Hanning-windowed segments to reduce spectral variance.
        """
        x = np.array(signal_1d, dtype=np.float64).flatten()
        N = len(x)
        if N < window_size:
            window_size = N
            overlap = 0

        step = max(1, window_size - overlap)
        window = np.hanning(window_size)
        win_energy = np.sum(window ** 2)

        segment_psds = []
        for start in range(0, N - window_size + 1, step):
            segment = x[start : start + window_size] * window
            fft_seg = np.fft.rfft(segment)
            seg_psd = (np.abs(fft_seg) ** 2) / (self.sampling_rate_hz * win_energy)
            segment_psds.append(seg_psd)

        if not segment_psds:
            fft_seg = np.fft.rfft(x * np.hanning(N))
            freqs = np.fft.rfftfreq(N, d=1.0 / self.sampling_rate_hz)
            psd = (np.abs(fft_seg) ** 2) / max(1.0, N)
            return freqs, psd

        avg_psd = np.mean(segment_psds, axis=0)
        freqs = np.fft.rfftfreq(window_size, d=1.0 / self.sampling_rate_hz)
        return freqs, avg_psd

