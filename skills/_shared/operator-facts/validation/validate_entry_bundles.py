from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

VALIDATION_DIR = Path(__file__).resolve().parent
OPERATOR_FACTS_ROOT = VALIDATION_DIR.parent
SCRIPT_DIR = OPERATOR_FACTS_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import OPERATOR_FACTS_ROOT as ROOT_FROM_COMMON
from validate_ms_facts import load_json, validate_instance


SCHEMA_DIR = ROOT_FROM_COMMON / "schemas"
DEFAULT_BUNDLE_ROOT = ROOT_FROM_COMMON / "bundles" / "entries"


def iter_bundle_paths(bundle_root: Path) -> List[Path]:
    return sorted(path for path in bundle_root.rglob("*.json") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate refactored MindSpore entry bundles.")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--schema", type=Path, default=SCHEMA_DIR / "ms_entry_bundle.schema.json")
    args = parser.parse_args()

    schema = load_json(args.schema)
    bundle_paths = iter_bundle_paths(args.bundle_root)
    errors: List[str] = []

    for path in bundle_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        errors.extend(validate_instance(data, schema, path.name))

    if errors:
        print(f"validate_entry_bundles: FAILED ({len(errors)} issues)")
        for error in errors:
            print(f"- {error}")
        return 1

    print("validate_entry_bundles: OK")
    print(f"bundle_root={args.bundle_root}")
    print(f"bundle_count={len(bundle_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
