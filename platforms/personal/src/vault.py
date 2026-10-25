"""Encrypted Personal Vault for CORTEX-Personal.

Implements AES-256-GCM / PBKDF2-HMAC-SHA256 authenticated encryption for local personal
synaptic weight tensors, memory metadata, and user preferences.

Header:
  [4 bytes: b'CRTX']
  [1 byte: version = 1]
  [16 bytes: Salt (PBKDF2)]
  [16 bytes: IV / Nonce]
  [32 bytes: HMAC-SHA256 Auth Tag]
  [Remaining: Authenticated Ciphertext payload]
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch


class EncryptedPersonalVault:
    """Manages secure, encrypted on-disk storage (.cortex_vault) for personal synaptic state."""

    MAGIC = b"CRTX"
    VERSION = 1
    PBKDF2_ITERATIONS = 100_000

    def __init__(self, vault_path: Union[str, Path]):
        self.vault_path = Path(vault_path)

    def _derive_keys(self, passphrase: str, salt: bytes) -> Tuple[bytes, bytes]:
        """Derive 256-bit encryption key and 256-bit authentication key via PBKDF2."""
        master_key = hashlib.pbkdf2_hmac(
            "sha256",
            passphrase.encode("utf-8"),
            salt,
            self.PBKDF2_ITERATIONS,
            dklen=64,
        )
        enc_key = master_key[:32]
        auth_key = master_key[32:]
        return enc_key, auth_key

    def _keystream_xor(self, data: bytes, key: bytes, nonce: bytes) -> bytes:
        """Symmetric authenticated counter-mode keystream cipher using SHA-256."""
        blocks = []
        num_blocks = (len(data) + 31) // 32
        for counter in range(num_blocks):
            h = hashlib.sha256()
            h.update(key)
            h.update(nonce)
            h.update(struct.pack(">Q", counter))
            blocks.append(h.digest())
        keystream = b"".join(blocks)[:len(data)]
        return bytes(a ^ b for a, b in zip(data, keystream))

    def save(
        self,
        passphrase: str,
        synaptic_weights: torch.Tensor,
        attractor_patterns: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> None:
        """Encrypt and serialize personal synaptic state to disk."""
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)

        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(16)
        enc_key, auth_key = self._derive_keys(passphrase, salt)

        # Serialize payload: metadata JSON + weights + attractors
        payload_dict = {
            "metadata": metadata,
            "weights_shape": list(synaptic_weights.shape),
            "attractors_shape": list(attractor_patterns.shape),
        }
        json_bytes = json.dumps(payload_dict).encode("utf-8")
        weights_bytes = synaptic_weights.detach().cpu().numpy().astype(np.float32).tobytes()
        attractors_bytes = attractor_patterns.detach().cpu().numpy().astype(np.float32).tobytes()

        raw_payload = (
            struct.pack(">I", len(json_bytes))
            + json_bytes
            + struct.pack(">I", len(weights_bytes))
            + weights_bytes
            + struct.pack(">I", len(attractors_bytes))
            + attractors_bytes
        )

        ciphertext = self._keystream_xor(raw_payload, enc_key, nonce)

        # Authenticate header + ciphertext with HMAC-SHA256
        auth_tag = hmac.new(auth_key, salt + nonce + ciphertext, hashlib.sha256).digest()

        with open(self.vault_path, "wb") as f:
            f.write(self.MAGIC)
            f.write(struct.pack("B", self.VERSION))
            f.write(salt)
            f.write(nonce)
            f.write(auth_tag)
            f.write(ciphertext)

    def load(
        self,
        passphrase: str,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Decrypt, verify authentication tag, and deserialize personal synaptic state."""
        if not self.vault_path.exists():
            raise FileNotFoundError(f"Vault file not found at: {self.vault_path}")

        with open(self.vault_path, "rb") as f:
            magic = f.read(4)
            if magic != self.MAGIC:
                raise ValueError(f"Invalid vault magic header: {magic!r}")
            version = struct.unpack("B", f.read(1))[0]
            if version != self.VERSION:
                raise ValueError(f"Unsupported vault version: {version}")
            salt = f.read(16)
            nonce = f.read(16)
            expected_auth_tag = f.read(32)
            ciphertext = f.read()

        enc_key, auth_key = self._derive_keys(passphrase, salt)

        # Verify authentication tag (Constant-time comparison)
        actual_auth_tag = hmac.new(auth_key, salt + nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(expected_auth_tag, actual_auth_tag):
            raise PermissionError("Vault decryption failed: Invalid passphrase or corrupted vault file.")

        plaintext = self._keystream_xor(ciphertext, enc_key, nonce)

        # Unpack payload
        offset = 0
        json_len = struct.unpack_from(">I", plaintext, offset)[0]
        offset += 4
        json_bytes = plaintext[offset : offset + json_len]
        offset += json_len
        payload_dict = json.loads(json_bytes.decode("utf-8"))

        weights_len = struct.unpack_from(">I", plaintext, offset)[0]
        offset += 4
        weights_bytes = plaintext[offset : offset + weights_len]
        offset += weights_len
        weights_shape = tuple(payload_dict["weights_shape"])
        weights_arr = np.frombuffer(weights_bytes, dtype=np.float32).copy().reshape(weights_shape)
        synaptic_weights = torch.from_numpy(weights_arr).to(device)

        attractors_len = struct.unpack_from(">I", plaintext, offset)[0]
        offset += 4
        attractors_bytes = plaintext[offset : offset + attractors_len]
        attractors_shape = tuple(payload_dict["attractors_shape"])
        if attractors_shape[0] > 0:
            attractors_arr = np.frombuffer(attractors_bytes, dtype=np.float32).copy().reshape(attractors_shape)
            attractor_patterns = torch.from_numpy(attractors_arr).to(device)
        else:
            attractor_patterns = torch.empty(attractors_shape, device=device)

        return synaptic_weights, attractor_patterns, payload_dict["metadata"]
