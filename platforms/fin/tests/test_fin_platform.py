"""Unit tests for CORTEX-Fin platform."""

import unittest
import torch

from platforms.fin.src.anomaly_scorer import FinancialAnomalyScorer, FraudScoringReport
from platforms.fin.src.event_replay import HighThroughputReplayEngine, ReplayBatchResult
from platforms.fin.src.transaction_encoder import FinancialTransaction, TransactionFeatureEncoder


class TestCortexFinPlatform(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.encoder = TransactionFeatureEncoder(output_dim=32, seed=42)
        self.scorer = FinancialAnomalyScorer(dim=32)

    def test_transaction_feature_encoding(self):
        tx = FinancialTransaction(
            tx_id="tx_001",
            account_id="acc_123",
            amount_usd=45.50,
            mcc=5411,
            geo_distance_km=2.5,
            time_delta_seconds=3600.0,
            channel="POS_CHIP",
            is_international=False,
            velocity_1h_count=1,
        )
        vec = self.encoder.encode_transaction(tx)
        self.assertEqual(vec.shape, (32,))
        # Unit norm check
        norm = torch.norm(vec).item()
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_anomaly_scoring_and_attribution(self):
        # 1. Create a corpus of ordinary local grocery purchases
        normal_txs = [
            FinancialTransaction(
                tx_id=f"tx_norm_{i}",
                account_id="acc_main",
                amount_usd=25.0 + (i % 20),
                mcc=5411,
                geo_distance_km=1.0 + (i % 3),
                time_delta_seconds=7200.0,
                channel="POS_CHIP",
                is_international=False,
                velocity_1h_count=1,
            )
            for i in range(15)
        ]
        self.scorer.learn_normal_behavior(normal_txs)

        # 2. Score a normal transaction (should PASS with negative energy)
        normal_probe = FinancialTransaction(
            tx_id="tx_probe_norm",
            account_id="acc_main",
            amount_usd=30.0,
            mcc=5411,
            geo_distance_km=1.5,
            time_delta_seconds=7200.0,
            channel="POS_CHIP",
            is_international=False,
            velocity_1h_count=1,
        )
        norm_report = self.scorer.score_transaction(normal_probe)
        self.assertEqual(norm_report.decision, "PASS")
        self.assertLess(norm_report.lyapunov_energy, 0.0)
        self.assertTrue(norm_report.shadow_mode)

        # 3. Score an anomalous international high-value transaction
        fraud_tx = FinancialTransaction(
            tx_id="tx_fraud_alert",
            account_id="acc_main",
            amount_usd=98500.0,
            mcc=7995,  # High risk gambling/crypto
            geo_distance_km=12500.0,  # 12,500 km away
            time_delta_seconds=120.0,  # 2 mins later
            channel="CARD_NOT_PRESENT",
            is_international=True,
            velocity_1h_count=24,
        )
        fraud_report = self.scorer.score_transaction(fraud_tx)
        self.assertIn(fraud_report.decision, ["REVIEW", "ESCALATE"])
        self.assertGreater(fraud_report.risk_score, norm_report.risk_score)

        # 4. Verify explainable feature attributions are present
        attributions = fraud_report.feature_attributions
        self.assertIn("log_amount", attributions)
        self.assertIn("log_geo_distance", attributions)
        self.assertIn("is_international", attributions)

    def test_high_throughput_event_replay(self):
        engine = HighThroughputReplayEngine(self.scorer)
        batch = [
            FinancialTransaction(
                tx_id=f"replay_tx_{i}",
                account_id=f"acc_{i % 5}",
                amount_usd=10.0 + i,
                mcc=5411,
                geo_distance_km=2.0,
                time_delta_seconds=1800.0,
                channel="POS_CHIP",
            )
            for i in range(100)
        ]

        result = engine.replay_stream(batch)
        self.assertIsInstance(result, ReplayBatchResult)
        self.assertEqual(result.total_transactions, 100)
        self.assertGreater(result.throughput_tx_per_sec, 500.0)
        self.assertGreater(result.mean_latency_us, 0.0)
        self.assertGreater(result.p95_latency_us, 0.0)

    def test_fcra_gdpr_explain_decision(self):
        tx = FinancialTransaction(
            tx_id="tx_high_risk_burst",
            account_id="acc_user",
            amount_usd=250000.0,
            mcc=7995,
            geo_distance_km=9000.0,
            time_delta_seconds=30.0,
            channel="CARD_NOT_PRESENT",
            is_international=True,
            velocity_1h_count=18,
        )
        explanation = self.scorer.explain_decision(tx)
        self.assertEqual(explanation["tx_id"], "tx_high_risk_burst")
        self.assertTrue(explanation["compliance"]["fcra_compliant"])
        self.assertTrue(explanation["compliance"]["gdpr_article_22_explainable"])
        self.assertGreater(len(explanation["adverse_action_codes"]), 0)
        self.assertIsNotNone(explanation["top_risk_driver"])

    def test_impossible_leap_velocity(self):
        engine = HighThroughputReplayEngine(self.scorer)
        txs = [
            FinancialTransaction(
                tx_id="tx_nyc",
                account_id="acc_traveler",
                amount_usd=50.0,
                mcc=5411,
                geo_distance_km=0.0,
                time_delta_seconds=0.0,
            ),
            FinancialTransaction(
                tx_id="tx_london",
                account_id="acc_traveler",
                amount_usd=1200.0,
                mcc=5311,
                geo_distance_km=5585.0,  # NYC to London
                time_delta_seconds=600.0,  # 10 minutes later (apparent ~33,510 km/h)
            ),
        ]
        anomalies = engine.detect_velocity_anomalies(txs, max_plausible_km_h=900.0)
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["account_id"], "acc_traveler")
        self.assertTrue(anomalies[0]["physical_impossibility"])
        self.assertGreater(anomalies[0]["apparent_velocity_km_h"], 900.0)


if __name__ == "__main__":
    unittest.main()

