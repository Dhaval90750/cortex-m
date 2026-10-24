"""Database Mastery & Synaptic Schema Ingestion Suite for Cortex-M.

Provides direct bridging between relational/document/vector databases
and Cortex-M's continuous synaptic memory substrate.
"""

from __future__ import annotations

from .db_trainer import DatabaseKnowledgeTrainer, DatabaseSchemaMetadata

__all__ = [
    "DatabaseKnowledgeTrainer",
    "DatabaseSchemaMetadata",
]
