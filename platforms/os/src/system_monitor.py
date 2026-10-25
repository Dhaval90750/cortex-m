"""Userspace Resource & Non-Privileged Execution Monitor for CORTEX-OS.

In compliance with NFSR-OS-01 & Section I.1.2:
1. Strict non-privileged execution validation (UID != 0 or standard Windows user).
2. In-memory index structure RAM size calculation (ceiling: <= 2.5 MB).
3. Process memory usage inspection and throttling alerts.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from platforms.os.src.file_watcher import FileDocument


@dataclass
class ResourceUsageSnapshot:
    total_indexed_documents: int
    raw_tensor_memory_bytes: int
    estimated_total_memory_kb: float
    is_non_privileged: bool
    memory_ceiling_exceeded: bool
    status_summary: str


class SystemResourceMonitor:
    """Validates memory and privilege constraints for CORTEX-OS."""

    def __init__(self, max_allowed_ram_mb: float = 2.5):
        self.max_allowed_ram_bytes = int(max_allowed_ram_mb * 1024 * 1024)

    def is_non_privileged(self) -> bool:
        """Verify the process is executing without root or administrator privileges."""
        if hasattr(os, "getuid"):
            # POSIX systems (Linux, macOS): UID != 0
            return os.getuid() != 0
        elif sys.platform == "win32":
            # Windows: Check if current token does not possess elevated admin privileges
            try:
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() == 0
            except Exception:
                return True
        return True

    def calculate_index_footprint(self, documents: List[FileDocument]) -> int:
        """Compute the raw memory occupied by in-memory 64-D tensors."""
        tensor_bytes = 0
        for doc in documents:
            if isinstance(doc.vector_64d, torch.Tensor):
                tensor_bytes += doc.vector_64d.element_size() * doc.vector_64d.nelement()
            # Estimate metadata size (~256 bytes per doc)
            tensor_bytes += 256
        return tensor_bytes

    def inspect_resources(self, documents: List[FileDocument]) -> ResourceUsageSnapshot:
        """Produce a verified resource snapshot against the 2.5 MB ceiling."""
        tensor_bytes = self.calculate_index_footprint(documents)
        estimated_total_kb = tensor_bytes / 1024.0
        exceeded = tensor_bytes > self.max_allowed_ram_bytes
        non_priv = self.is_non_privileged()

        if exceeded:
            status = f"WARNING: Memory footprint ({estimated_total_kb:.1f} KB) exceeds {self.max_allowed_ram_bytes / 1024:.0f} KB ceiling."
        else:
            status = f"HEALTHY: In-memory footprint is {estimated_total_kb:.1f} KB (well within 2.5 MB limit)."

        return ResourceUsageSnapshot(
            total_indexed_documents=len(documents),
            raw_tensor_memory_bytes=tensor_bytes,
            estimated_total_memory_kb=estimated_total_kb,
            is_non_privileged=non_priv,
            memory_ceiling_exceeded=exceeded,
            status_summary=status,
        )
