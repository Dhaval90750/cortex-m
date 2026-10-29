"""Validation Script for CORTEX-M Master Data Registry.

Enforces 12-field schema compliance, URL protocols, and role/license governance.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tests_e2e.reference_contracts import MasterDataRegistryValidator


def validate_registry_file(file_path: Path) -> bool:
    if not file_path.exists():
        print(f"[FAIL] Registry file not found: {file_path}")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[INFO] Validating {len(rows)} entries in {file_path}...")
    errors: list[str] = []
    for idx, row in enumerate(rows, start=1):
        is_valid, row_errors = MasterDataRegistryValidator.validate_row(row)
        if not is_valid:
            errors.append(f"Row {idx} (ID: {row.get('dataset_id', 'unknown')}): {', '.join(row_errors)}")

    if errors:
        print(f"[FAIL] Found {len(errors)} validation error(s):")
        for err in errors[:10]:
            print(f"  - {err}")
        return False

    print(f"[SUCCESS] All {len(rows)} entries passed 12-field schema validation.")
    return True


if __name__ == "__main__":
    registry_path = Path(__file__).parent.parent / "data" / "registry" / "datasets.csv"
    success = validate_registry_file(registry_path)
    sys.exit(0 if success else 1)
