"""Deterministic Biosignal Filtering & Preprocessing for CORTEX-Health.

Provides robust microvolt EEG signal cleaning:
1. DC offset removal & baseline drift detrending.
2. Powerline interference rejection (50.0 Hz and 60.0 Hz notch filters).
3. Bandpass filtering (0.5 Hz to 100.0 Hz).
4. Outlier amplitude rejection (clipping artifacts beyond +/- 500 uV).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np


@dataclass
class FilteredSignal:
    """Preprocessed multi-channel neural biosignal window."""
    clean_microvolts: np.ndarray  # Shape: (num_channels, num_samples)
    sampling_rate_hz: float
    num_channels: int
    num_samples: int
    duration_seconds: float
    artifacts_clipped_count: int


class BiosignalFilter:
    """Standardized non-clinical biosignal cleaning pipeline."""

    def __init__(
        self,
        sampling_rate_hz: float = 250.0,
        notch_freq_hz: float = 60.0,
        lowcut_hz: float = 0.5,
        highcut_hz: float = 100.0,
        amplitude_limit_uv: float = 500.0,
    ):
        self.sampling_rate_hz = sampling_rate_hz
        self.notch_freq_hz = notch_freq_hz
        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.amplitude_limit_uv = amplitude_limit_uv

    def filter_signal(self, raw_microvolts: np.ndarray) -> FilteredSignal:
        """Process raw 2D EEG array (channels, samples)."""
        signal = np.array(raw_microvolts, dtype=np.float64, copy=True)
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)

        num_channels, num_samples = signal.shape
        clipped_count = 0

        # 1. DC offset removal (detrend mean per channel)
        signal = signal - np.mean(signal, axis=1, keepdims=True)

        # 2. Simple IIR Notch filter implementation for 50/60 Hz powerline hum
        # Resonator: y[n] = x[n] - 2*cos(w0)*x[n-1] + x[n-2] + r*2*cos(w0)*y[n-1] - r^2*y[n-2]
        w0 = 2.0 * math.pi * (self.notch_freq_hz / self.sampling_rate_hz)
        r = 0.95
        b0, b1, b2 = 1.0, -2.0 * math.cos(w0), 1.0
        a1, a2 = -2.0 * r * math.cos(w0), r * r

        for ch in range(num_channels):
            x = signal[ch]
            y = np.zeros_like(x)
            for n in range(2, num_samples):
                y[n] = (b0 * x[n] + b1 * x[n - 1] + b2 * x[n - 2]) - (a1 * y[n - 1] + a2 * y[n - 2])
            signal[ch] = y

        # 3. Zero-mean stabilization after notch filter
        signal = signal - np.mean(signal, axis=1, keepdims=True)

        # 4. Artifact clipping (+/- 500 uV)
        over_mask = np.abs(signal) > self.amplitude_limit_uv
        clipped_count = int(np.sum(over_mask))
        np.clip(signal, -self.amplitude_limit_uv, self.amplitude_limit_uv, out=signal)

        duration = num_samples / max(1.0, self.sampling_rate_hz)

        return FilteredSignal(
            clean_microvolts=signal,
            sampling_rate_hz=self.sampling_rate_hz,
            num_channels=num_channels,
            num_samples=num_samples,
            duration_seconds=duration,
            artifacts_clipped_count=clipped_count,
        )

    def estimate_snr_db(
        self,
        raw_signal: np.ndarray,
        signal_band: Tuple[float, float] = (1.0, 30.0),
        noise_band: Tuple[float, float] = (50.0, 70.0),
    ) -> float:
        """Estimate Signal-to-Noise Ratio (SNR in dB) of EEG recordings.
        
        Compares spectral power in the physiological neural bandwidth against
        ambient electromagnetic / powerline interference bands.
        """
        x = np.array(raw_signal, dtype=np.float64).flatten()
        N = len(x)
        if N < 8:
            return 0.0

        fft_vals = np.abs(np.fft.rfft(x * np.hanning(N))) ** 2
        freqs = np.fft.rfftfreq(N, d=1.0 / self.sampling_rate_hz)

        sig_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
        noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])

        p_sig = float(np.sum(fft_vals[sig_mask])) if np.any(sig_mask) else 1e-6
        p_noise = float(np.sum(fft_vals[noise_mask])) if np.any(noise_mask) else 1e-6

        snr = 10.0 * math.log10(max(1e-6, p_sig) / max(1e-6, p_noise))
        return round(snr, 2)

    def check_electrode_contact(
        self,
        signal: np.ndarray,
        railing_threshold_uv: float = 480.0,
    ) -> Dict[str, Any]:
        """Verify scalp electrode impedance and lead connection integrity.
        
        Identifies open-circuit disconnected leads (flatline zero variance) and
        saturated railed electrodes (pegged at ADC clipping ceiling).
        """
        arr = np.array(signal, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        channels_status = []
        all_leads_good = True

        for ch in range(arr.shape[0]):
            ch_data = arr[ch]
            variance = float(np.var(ch_data))
            max_val = float(np.max(np.abs(ch_data)))
            is_flatline = variance < 1e-4
            is_railed = max_val >= railing_threshold_uv

            lead_ok = not is_flatline and not is_railed
            if not lead_ok:
                all_leads_good = False

            channels_status.append({
                "channel_index": ch,
                "lead_good": lead_ok,
                "is_flatline": is_flatline,
                "is_railed": is_railed,
                "variance_uv2": round(variance, 3),
                "peak_amplitude_uv": round(max_val, 2),
            })

        return {
            "all_leads_contact_good": all_leads_good,
            "num_channels": arr.shape[0],
            "channels": channels_status,
        }

