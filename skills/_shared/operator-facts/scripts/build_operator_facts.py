from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import DEFAULT_MS_ROOT, DEFAULT_OUT_DIR, OPERATOR_FACTS_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = OPERATOR_FACTS_ROOT / "validation"
DEFAULT_BUNDLE_ROOT = OPERATOR_FACTS_ROOT / "bundles" / "entries"


def run(script: Path, *extra_args: str) -> None:
    subprocess.run([sys.executable, str(script), *extra_args], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build refactored MindSpore operator-facts tables, validate them, and generate entry bundles."
    )
    parser.add_argument("--ms-root", type=Path, default=DEFAULT_MS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    run(SCRIPT_DIR / "build_ms_facts.py", "--ms-root", str(args.ms_root), "--out-dir", str(args.out_dir))
    if not args.skip_validate:
        run(VALIDATION_DIR / "validate_ms_facts.py", "--data-dir", str(args.out_dir))

    run(
        SCRIPT_DIR / "build_entry_bundles.py",
        "--entry-identity",
        str(args.out_dir / "ms_entry_identity.jsonl"),
        "--unit-identity",
        str(args.out_dir / "ms_unit_identity.jsonl"),
        "--entry-unit-edges",
        str(args.out_dir / "ms_entry_unit_edges.jsonl"),
        "--unit-graph-edges",
        str(args.out_dir / "ms_unit_graph_edges.jsonl"),
        "--bundle-root",
        str(args.bundle_root),
    )
    if not args.skip_validate:
        run(VALIDATION_DIR / "validate_entry_bundles.py", "--bundle-root", str(args.bundle_root))

    print("build_operator_facts: OK")
    print(f"ms_root={args.ms_root}")
    print(f"out_dir={args.out_dir}")
    print(f"bundle_root={args.bundle_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
