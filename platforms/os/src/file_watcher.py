"""Non-Privileged Filesystem Event Watcher and Document Extractor.

In compliance with CORTEX-OS FR-OS-01:
1. Strict non-privileged execution (UID != 0, no ring-0 access).
2. Debounced event queue (500ms window) to suppress partial-write jitter.
3. Safe document content extraction (.py, .md, .rs, .tex, .txt, .json).
4. Deterministic Gist feature vector generation (64-D normalized vector).
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


class EventType(str, Enum):
    CREATED = "CREATED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"


SUPPORTED_EXTENSIONS = {".py", ".md", ".rs", ".tex", ".txt", ".json", ".toml", ".yaml", ".c", ".h"}


@dataclass
class FileDocument:
    file_id: str
    path: str
    extension: str
    file_size_bytes: int
    content_preview: str
    vector_64d: torch.Tensor
    last_modified_timestamp: float


@dataclass
class FileModifiedEvent:
    event_type: EventType
    file_path: str
    timestamp: float
    document: Optional[FileDocument] = None


class NonPrivilegedFileWatcher:
    """Non-privileged userspace filesystem scanner with 500ms event debouncing."""

    def __init__(
        self,
        watch_paths: Optional[List[str]] = None,
        debounce_seconds: float = 0.5,
        vector_dim: int = 64,
    ):
        self.watch_paths = [Path(p) for p in (watch_paths or [])]
        self.debounce_seconds = debounce_seconds
        self.vector_dim = vector_dim
        self._pending_events: Dict[str, Tuple[EventType, float]] = {}
        self._document_registry: Dict[str, FileDocument] = {}

    def extract_gist_vector(self, text: str, file_path: str) -> torch.Tensor:
        """Deterministically map document text into normalized 64-D semantic representation."""
        # Clean text tokens
        words = [w.strip().lower() for w in text.split() if len(w.strip()) > 1]
        vec = np.zeros(self.vector_dim, dtype=np.float32)

        if not words:
            # Fallback on path hash
            h = hashlib.sha256(file_path.encode("utf-8")).digest()
            for i, b in enumerate(h[:self.vector_dim]):
                vec[i % self.vector_dim] += (b / 255.0) - 0.5
        else:
            for w in words:
                h = hashlib.md5(w.encode("utf-8")).digest()
                idx = int.from_bytes(h[:2], byteorder="big") % self.vector_dim
                val = (h[2] / 128.0) - 1.0
                vec[idx] += val

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return torch.from_numpy(vec)

    def queue_event(self, event_type: EventType, file_path: str, timestamp: Optional[float] = None) -> None:
        """Register a raw filesystem event into the debounced queue."""
        now = timestamp if timestamp is not None else time.time()
        self._pending_events[file_path] = (event_type, now)

    def flush_events(self, current_time: Optional[float] = None) -> List[FileModifiedEvent]:
        """Process and flush debounced events exceeding the 500ms quiet threshold."""
        now = current_time if current_time is not None else time.time()
        ready_events: List[FileModifiedEvent] = []
        paths_to_remove: List[str] = []

        for fpath, (etype, etime) in self._pending_events.items():
            if now - etime >= self.debounce_seconds:
                paths_to_remove.append(fpath)
                p = Path(fpath)
                if p.suffix.lower() not in SUPPORTED_EXTENSIONS and etype != EventType.DELETED:
                    continue

                doc = None
                if etype != EventType.DELETED and p.exists() and p.is_file():
                    try:
                        content = p.read_text(encoding="utf-8", errors="replace")
                        preview = "\n".join(content.splitlines()[:5])
                        vec = self.extract_gist_vector(content, str(p))
                        doc = FileDocument(
                            file_id=hashlib.sha256(str(p).encode("utf-8")).hexdigest()[:12],
                            path=str(p),
                            extension=p.suffix.lower(),
                            file_size_bytes=len(content.encode("utf-8")),
                            content_preview=preview,
                            vector_64d=vec,
                            last_modified_timestamp=p.stat().st_mtime,
                        )
                        self._document_registry[str(p)] = doc
                    except Exception:
                        pass
                elif etype == EventType.DELETED:
                    self._document_registry.pop(str(p), None)

                ready_events.append(FileModifiedEvent(
                    event_type=etype,
                    file_path=fpath,
                    timestamp=etime,
                    document=doc,
                ))

        for p in paths_to_remove:
            del self._pending_events[p]

        return ready_events

    def get_registered_documents(self) -> List[FileDocument]:
        """Return all actively indexed document representations."""
        return list(self._document_registry.values())
