"""Financial Transaction Representation & Dense Tensor Encoder for CORTEX-Fin.

Encodes high-throughput transaction payloads into normalized hypersphere feature vectors
suitable for continuous Hopfield anomaly scoring.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F


@dataclass
class FinancialTransaction:
    """Standardized institutional transaction event."""
    tx_id: str
    account_id: str
    amount_usd: float
    mcc: int  # Merchant Category Code (e.g. 5411 = Grocery, 5732 = Electronics, 7995 = Gambling)
    geo_distance_km: float  # Distance from previous transaction location
    time_delta_seconds: float  # Time elapsed since prior account event
    channel: str = "CARD_NOT_PRESENT"  # 'POS_CHIP', 'CARD_NOT_PRESENT', 'ATM', 'WIRE'
    is_international: bool = False
    velocity_1h_count: int = 1
    currency: str = "USD"


class TransactionFeatureEncoder:
    """Projects tabular financial events into a continuous (output_dim) feature tensor."""

    CHANNEL_MAP = {
        "POS_CHIP": 0,
        "CONTACTLESS": 1,
        "CARD_NOT_PRESENT": 2,
        "ATM": 3,
        "WIRE": 4,
    }

    FEATURE_NAMES = [
        "log_amount",
        "mcc_normalized",
        "log_geo_distance",
        "log_time_delta",
        "channel_code",
        "is_international",
        "velocity_1h",
    ]

    def __init__(self, output_dim: int = 32, seed: int = 42):
        self.output_dim = output_dim
        num_raw_features = len(self.FEATURE_NAMES)

        # Orthogonal projection matrix to expand 7-D raw features into output_dim
        torch.manual_seed(seed)
        proj = torch.empty(num_raw_features, output_dim)
        torch.nn.init.orthogonal_(proj)
        self.projection = proj

    def encode_transaction(self, tx: FinancialTransaction) -> torch.Tensor:
        """Encode a single transaction into a unit-normalized (output_dim) dense vector."""
        log_amt = math.log1p(max(0.0, tx.amount_usd)) / 10.0
        mcc_norm = (tx.mcc % 10000) / 10000.0
        log_geo = math.log1p(max(0.0, tx.geo_distance_km)) / 10.0
        log_time = math.log1p(max(0.0, tx.time_delta_seconds)) / 15.0
        chan_val = self.CHANNEL_MAP.get(tx.channel, 2) / 5.0
        intl_val = 1.0 if tx.is_international else 0.0
        velo_val = min(tx.velocity_1h_count, 50) / 50.0

        raw_vec = torch.tensor(
            [log_amt, mcc_norm, log_geo, log_time, chan_val, intl_val, velo_val],
            dtype=torch.float32,
        )
        expanded = torch.matmul(raw_vec, self.projection)
        return F.normalize(expanded, p=2, dim=-1)

    def encode_batch(self, transactions: Sequence[FinancialTransaction]) -> torch.Tensor:
        """Vectorized encoding of a batch of transaction events."""
        vecs = [self.encode_transaction(tx) for tx in transactions]
        return torch.stack(vecs, dim=0)
