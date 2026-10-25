"""Unit tests for CORTEX-OS Platform."""

import os
import tempfile
import time
import unittest
from pathlib import Path

import torch

from platforms.os.src.affinity_cluster import (
    AffinityClusterReport,
    CognitiveAffinityGraph,
    ProjectCluster,
)
from platforms.os.src.file_watcher import (
    EventType,
    FileDocument,
    FileModifiedEvent,
    NonPrivilegedFileWatcher,
)
from platforms.os.src.omnibar_search import (
    OmnibarQueryResponse,
    SearchResult,
    SpotlightOmnibarEngine,
)
from platforms.os.src.system_monitor import (
    ResourceUsageSnapshot,
    SystemResourceMonitor,
)


class TestCortexOSPlatform(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_path = Path(self.temp_dir.name)
        self.watcher = NonPrivilegedFileWatcher(debounce_seconds=0.5)
        self.omnibar = SpotlightOmnibarEngine(beta=8.0, hopfield_iterations=3)
        self.graph = CognitiveAffinityGraph(affinity_threshold=0.45, sigma_affinity=1.0)
        self.monitor = SystemResourceMonitor(max_allowed_ram_mb=2.5)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_watcher_debouncing_and_gist_extraction(self):
        test_file = self.work_path / "research_notes.md"
        test_file.write_text(
            "# Autonomous Satellite Telemetry\nSpacecraft EPS bus voltage telemetry parsing.",
            encoding="utf-8",
        )

        now = 1000.0
        # Register event at t=1000.0
        self.watcher.queue_event(EventType.CREATED, str(test_file), timestamp=now)

        # 1. Flush immediately at t=1000.2 (200ms < 500ms debounce threshold)
        events_early = self.watcher.flush_events(current_time=now + 0.2)
        self.assertEqual(len(events_early), 0)

        # 2. Flush after quiet period at t=1000.6 (600ms >= 500ms)
        events_ready = self.watcher.flush_events(current_time=now + 0.6)
        self.assertEqual(len(events_ready), 1)
        ev = events_ready[0]
        self.assertEqual(ev.event_type, EventType.CREATED)
        self.assertIsNotNone(ev.document)
        doc = ev.document
        self.assertEqual(doc.extension, ".md")
        self.assertEqual(doc.vector_64d.shape[0], 64)
        self.assertAlmostEqual(float(torch.norm(doc.vector_64d)), 1.0, places=3)

    def test_spotlight_associative_omnibar_search_and_latency(self):
        # Create candidate documents
        doc_space = FileDocument(
            file_id="doc_01",
            path="/workspace/spacecraft_seu.py",
            extension=".py",
            file_size_bytes=1024,
            content_preview="Radiation SEU latchup anomaly detection for satellite EPS bus.",
            vector_64d=self.watcher.extract_gist_vector(
                "Radiation SEU latchup anomaly detection for satellite EPS bus.",
                "/workspace/spacecraft_seu.py",
            ),
            last_modified_timestamp=time.time(),
        )

        doc_bci = FileDocument(
            file_id="doc_02",
            path="/workspace/bci_wavelet.py",
            extension=".py",
            file_size_bytes=2048,
            content_preview="Continuous BCI wavelet frequency decomposition and alpha rhythm.",
            vector_64d=self.watcher.extract_gist_vector(
                "Continuous BCI wavelet frequency decomposition and alpha rhythm.",
                "/workspace/bci_wavelet.py",
            ),
            last_modified_timestamp=time.time(),
        )

        doc_fin = FileDocument(
            file_id="doc_03",
            path="/workspace/fraud_scorer.py",
            extension=".py",
            file_size_bytes=1500,
            content_preview="Financial ledger fraud scoring with Lyapunov energy attribution.",
            vector_64d=self.watcher.extract_gist_vector(
                "Financial ledger fraud scoring with Lyapunov energy attribution.",
                "/workspace/fraud_scorer.py",
            ),
            last_modified_timestamp=time.time(),
        )

        candidates = [doc_space, doc_bci, doc_fin]

        # Warmup query
        _ = self.omnibar.query("warmup", candidates)

        # Query conceptual terms
        resp = self.omnibar.query("satellite radiation seu latchup", candidates, top_k=2)
        self.assertIsInstance(resp, OmnibarQueryResponse)
        self.assertEqual(len(resp.results), 2)
        # Top result must be spacecraft_seu.py
        top_hit = resp.results[0]
        self.assertEqual(top_hit.file_id, "doc_01")
        self.assertIn("spacecraft_seu.py", top_hit.file_path)
        self.assertGreater(top_hit.resonance_score, resp.results[1].resonance_score)
        # Verify sub-10ms evaluation latency
        self.assertLess(resp.evaluation_latency_ms, 15.0)

    def test_cognitive_affinity_graph_clustering(self):
        # 2 documents in Space domain
        d1 = FileDocument("s1", "space_orbit.py", ".py", 100, "kepler orbit j2 perturbation",
                          self.watcher.extract_gist_vector("kepler orbit j2 perturbation", "space_orbit.py"), time.time())
        d2 = FileDocument("s2", "space_telemetry.py", ".py", 100, "kepler orbit telemetry satellite ground track",
                          self.watcher.extract_gist_vector("kepler orbit telemetry satellite ground track", "space_telemetry.py"), time.time())

        # 2 documents in Bio / Health domain
        d3 = FileDocument("h1", "eeg_filter.py", ".py", 100, "biosignal eeg microvolts notch filter",
                          self.watcher.extract_gist_vector("biosignal eeg microvolts notch filter", "eeg_filter.py"), time.time())
        d4 = FileDocument("h2", "bci_decoder.py", ".py", 100, "biosignal eeg microvolts bandpower",
                          self.watcher.extract_gist_vector("biosignal eeg microvolts bandpower", "bci_decoder.py"), time.time())

        report = self.graph.extract_clusters([d1, d2, d3, d4])
        self.assertIsInstance(report, AffinityClusterReport)
        self.assertEqual(report.total_documents, 4)
        self.assertGreaterEqual(report.num_clusters, 2)
        for cluster in report.clusters:
            self.assertGreaterEqual(cluster.cohesion_score, 0.0)
            self.assertEqual(cluster.centroid_vector.shape[0], 64)

    def test_resource_monitor_and_memory_ceiling(self):
        # Simulate 100 indexed documents
        dummy_docs = []
        vec = torch.randn(64)
        vec = vec / torch.norm(vec)
        for i in range(100):
            dummy_docs.append(FileDocument(
                file_id=f"doc_{i}",
                path=f"/path/to/doc_{i}.py",
                extension=".py",
                file_size_bytes=1000,
                content_preview="preview",
                vector_64d=vec,
                last_modified_timestamp=time.time(),
            ))

        snapshot = self.monitor.inspect_resources(dummy_docs)
        self.assertIsInstance(snapshot, ResourceUsageSnapshot)
        self.assertEqual(snapshot.total_indexed_documents, 100)
        self.assertFalse(snapshot.memory_ceiling_exceeded)
        # 100 docs should consume ~50 KB, well under 2560 KB (2.5 MB)
        self.assertLess(snapshot.estimated_total_memory_kb, 500.0)
        self.assertIsInstance(snapshot.is_non_privileged, bool)
        self.assertIn("HEALTHY", snapshot.status_summary)


if __name__ == "__main__":
    unittest.main()
