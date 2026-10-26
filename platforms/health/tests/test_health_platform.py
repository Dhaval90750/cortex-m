"""Unit tests for CORTEX-Health / BCI platform."""

import math
import unittest
import numpy as np
import torch

from platforms.health.src.bci_decoder import BCIDecodingResult, CognitiveState, ContinuousBCIDecoder
from platforms.health.src.signal_filter import BiosignalFilter, FilteredSignal
from platforms.health.src.wavelet_analysis import FrequencyBand, NeuralBandpower, WaveletFrequencyDecomposer


class TestCortexHealthPlatform(unittest.TestCase):
    def setUp(self):
        self.sampling_rate = 250.0
        self.filter = BiosignalFilter(sampling_rate_hz=self.sampling_rate, notch_freq_hz=60.0)
        self.decomposer = WaveletFrequencyDecomposer(sampling_rate_hz=self.sampling_rate)
        self.decoder = ContinuousBCIDecoder(beta=8.0)

    def test_biosignal_filtering_dc_offset_and_clipping(self):
        t = np.linspace(0, 1.0, int(self.sampling_rate), endpoint=False)
        # Synthetic signal with 50.0 uV DC offset, 10 Hz wave, and a 800 uV spike artifact
        raw = 50.0 + 15.0 * np.sin(2 * np.pi * 10.0 * t)
        raw[50] = 850.0  # Excessive spike

        filtered = self.filter.filter_signal(raw)
        self.assertIsInstance(filtered, FilteredSignal)
        # DC offset removed (within 2 uV residual due to asymmetric single-sample spike clipping)
        self.assertAlmostEqual(float(np.mean(filtered.clean_microvolts)), 0.0, delta=2.0)
        # Spike clipped to limit (500 uV)
        self.assertLessEqual(float(np.max(filtered.clean_microvolts)), 500.0)
        self.assertGreater(filtered.artifacts_clipped_count, 0)

    def test_wavelet_frequency_decomposition_alpha_and_beta(self):
        t = np.linspace(0, 2.0, int(self.sampling_rate * 2), endpoint=False)

        # 10 Hz pure sine wave (Alpha band: 8 - 13 Hz)
        alpha_wave = 20.0 * np.sin(2 * np.pi * 10.0 * t)
        bp_alpha = self.decomposer.decompose(alpha_wave)
        self.assertGreater(bp_alpha.alpha_power, bp_alpha.beta_power)
        self.assertGreater(bp_alpha.alpha_power, bp_alpha.delta_power)

        # 22 Hz pure sine wave (Beta band: 13 - 30 Hz)
        beta_wave = 20.0 * np.sin(2 * np.pi * 22.0 * t)
        bp_beta = self.decomposer.decompose(beta_wave)
        self.assertGreater(bp_beta.beta_power, bp_beta.alpha_power)
        self.assertGreater(bp_beta.focus_ratio, bp_alpha.focus_ratio)

    def test_continuous_bci_decoding_and_disclaimer(self):
        t = np.linspace(0, 1.0, int(self.sampling_rate), endpoint=False)
        # Alpha dominant relaxed rhythm
        alpha_signal = 25.0 * np.sin(2 * np.pi * 10.0 * t) + np.random.normal(0, 2.0, len(t))

        res = self.decoder.decode_window(alpha_signal)
        self.assertIsInstance(res, BCIDecodingResult)
        self.assertEqual(res.predicted_state, CognitiveState.REST)
        self.assertGreaterEqual(res.confidence, 0.5)

        # Verify non-clinical disclaimer is always enforced
        self.assertIn("NON-CLINICAL RESEARCH ONLY", res.non_clinical_disclaimer)
        self.assertIn("Explicitly prohibited", res.non_clinical_disclaimer)

    def test_snr_and_electrode_impedance_contact(self):
        t = np.linspace(0, 1.0, int(self.sampling_rate), endpoint=False)
        clean = 15.0 * np.sin(2 * np.pi * 10.0 * t)
        snr = self.filter.estimate_snr_db(clean)
        self.assertGreater(snr, 10.0)

        # Check electrode contact test (good lead vs flatline vs railed)
        multi_ch = np.array([
            clean,
            np.zeros_like(clean),  # Flatline disconnected
            np.full_like(clean, 495.0),  # Saturated railed
        ])
        status = self.filter.check_electrode_contact(multi_ch)
        self.assertFalse(status["all_leads_contact_good"])
        self.assertTrue(status["channels"][0]["lead_good"])
        self.assertTrue(status["channels"][1]["is_flatline"])
        self.assertTrue(status["channels"][2]["is_railed"])

    def test_welch_psd_periodogram(self):
        t = np.linspace(0, 2.0, int(self.sampling_rate * 2), endpoint=False)
        sig = 20.0 * np.sin(2 * np.pi * 12.0 * t) + np.random.normal(0, 1.0, len(t))
        freqs, psd = self.decomposer.welch_psd(sig, window_size=128, overlap=64)
        self.assertEqual(len(freqs), len(psd))
        self.assertGreater(len(freqs), 10)
        # Peak frequency should be near 12 Hz
        peak_freq = freqs[np.argmax(psd)]
        self.assertAlmostEqual(peak_freq, 12.0, delta=2.5)

    def test_stream_window_decoding_with_ema(self):
        self.decoder.reset_stream()
        t = np.linspace(0, 1.0, int(self.sampling_rate), endpoint=False)
        alpha_signal = 25.0 * np.sin(2 * np.pi * 10.0 * t)

        # Run 3 consecutive windows
        r1 = self.decoder.decode_stream_window(alpha_signal, ema_alpha=0.4)
        r2 = self.decoder.decode_stream_window(alpha_signal, ema_alpha=0.4)
        r3 = self.decoder.decode_stream_window(alpha_signal, ema_alpha=0.4)

        self.assertEqual(r3.predicted_state, CognitiveState.REST)
        self.assertTrue(r3.metadata["smoothed_via_ema"])
        self.assertGreater(r3.confidence, 0.5)


if __name__ == "__main__":
    unittest.main()

