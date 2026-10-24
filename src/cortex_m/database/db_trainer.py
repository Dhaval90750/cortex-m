"""Database Knowledge Trainer & Synaptic Schema Ingestion Engine.

Enables CortexLM to learn real-world database schemas, SQL DDL definitions,
table relationships, indexes, and queries directly into continuous synaptic weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from cortex_m.model.cortex_lm import CortexLM


@dataclass
class ColumnSpec:
    name: str
    data_type: str
    is_primary_key: bool = False
    is_nullable: bool = True
    references: Optional[str] = None


@dataclass
class TableSpec:
    name: str
    columns: List[ColumnSpec] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)
    raw_ddl: str = ""


@dataclass
class DatabaseSchemaMetadata:
    database_name: str
    tables: Dict[str, TableSpec] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)


class DatabaseKnowledgeTrainer:
    """Ingests database schemas and SQL knowledge into CortexLM's synaptic memory."""

    def __init__(self, lm: CortexLM):
        self.lm = lm

    def parse_sql_ddl(self, ddl_text: str, database_name: str = "production_db") -> DatabaseSchemaMetadata:
        """Parse raw SQL CREATE TABLE statements into structured table specifications."""
        schema = DatabaseSchemaMetadata(database_name=database_name)

        # Regex match CREATE TABLE statements
        table_matches = re.finditer(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_]+)\s*\((.*?)\);",
            ddl_text,
            re.IGNORECASE | re.DOTALL,
        )

        for match in table_matches:
            tbl_name = match.group(1).strip().lower()
            body = match.group(2).strip()

            table_spec = TableSpec(name=tbl_name, raw_ddl=match.group(0))
            lines = [line.strip().rstrip(",") for line in body.split("\n") if line.strip()]

            for line in lines:
                # Primary key constraint line
                if re.match(r"^PRIMARY\s+KEY\s*\((.*?)\)", line, re.IGNORECASE):
                    pk_match = re.match(r"^PRIMARY\s+KEY\s*\((.*?)\)", line, re.IGNORECASE)
                    if pk_match:
                        pk_col = pk_match.group(1).replace('"', "").replace("`", "").strip().lower()
                        for col in table_spec.columns:
                            if col.name == pk_col:
                                col.is_primary_key = True
                    continue

                # Foreign key constraint line
                if re.match(r"^FOREIGN\s+KEY\s*\((.*?)\)\s+REFERENCES\s+([a-zA-Z0-9_]+)\s*\((.*?)\)", line, re.IGNORECASE):
                    fk_m = re.match(r"^FOREIGN\s+KEY\s*\((.*?)\)\s+REFERENCES\s+([a-zA-Z0-9_]+)\s*\((.*?)\)", line, re.IGNORECASE)
                    if fk_m:
                        fk_col = fk_m.group(1).strip().lower()
                        ref_tbl = fk_m.group(2).strip().lower()
                        ref_col = fk_m.group(3).strip().lower()
                        rel_desc = f"{tbl_name}.{fk_col} references {ref_tbl}.{ref_col}"
                        schema.relationships.append(rel_desc)
                        for col in table_spec.columns:
                            if col.name == fk_col:
                                col.references = f"{ref_tbl}.{ref_col}"
                    continue

                # Standard column line: col_name TYPE constraints
                parts = line.split()
                if len(parts) >= 2:
                    c_name = parts[0].replace('"', "").replace("`", "").strip().lower()
                    c_type = parts[1].upper()
                    is_pk = "PRIMARY KEY" in line.upper()
                    is_null = "NOT NULL" not in line.upper() and not is_pk

                    col_spec = ColumnSpec(
                        name=c_name,
                        data_type=c_type,
                        is_primary_key=is_pk,
                        is_nullable=is_null,
                    )
                    table_spec.columns.append(col_spec)

            schema.tables[tbl_name] = table_spec

        return schema

    def ingest_schema(
        self,
        schema_or_ddl: str | DatabaseSchemaMetadata,
        database_name: str = "app_db",
    ) -> Dict[str, Any]:
        """Convert database schema into semantic knowledge and imprint directly into synapses."""
        if isinstance(schema_or_ddl, str):
            schema = self.parse_sql_ddl(schema_or_ddl, database_name=database_name)
        else:
            schema = schema_or_ddl

        imprinted_tables: List[str] = []
        imprinted_relations: List[str] = []

        # 1. Imprint Table Definitions
        for tbl_name, table in schema.tables.items():
            cols_summary = ", ".join([f"{c.name} ({c.data_type})" for c in table.columns])
            pks = [c.name for c in table.columns if c.is_primary_key]
            pk_str = f"with primary key '{pks[0]}'" if pks else "without primary key"

            table_knowledge = (
                f"Database table {tbl_name} in schema {schema.database_name} {pk_str} "
                f"contains columns: {cols_summary}. Table {tbl_name} columns in {schema.database_name}."
            )
            mem_id = f"db_table_{schema.database_name}_{tbl_name}"
            self.lm.learn(
                table_knowledge,
                memory_id=mem_id,
                source=f"db_schema_{schema.database_name}",
            )
            imprinted_tables.append(tbl_name)

        # 2. Imprint Relational Foreign Key Graph
        for rel in schema.relationships:
            rel_knowledge = f"Database relational integrity constraint: table {rel} in schema {schema.database_name}."
            rel_id = f"db_rel_{schema.database_name}_{re.sub(r'[^a-zA-Z0-9_]', '_', rel)}"
            self.lm.learn(
                rel_knowledge,
                memory_id=rel_id,
                source=f"db_schema_{schema.database_name}",
            )
            imprinted_relations.append(rel)

        return {
            "status": "schema_imprinted_in_synapses",
            "database_name": schema.database_name,
            "total_tables_ingested": len(imprinted_tables),
            "tables": imprinted_tables,
            "relationships": imprinted_relations,
        }
