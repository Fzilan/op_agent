from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(script_name: str, *extra_args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / script_name), *extra_args], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build phase-1 operator-facts indexes and bundles.")
    parser.add_argument("--ms-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    args = parser.parse_args()

    if sys.version_info < (3, 9):
        raise SystemExit("build_phase1.py requires Python 3.9+")

    shared_args = []
    if args.ms_root is not None:
        shared_args.extend(["--ms-root", str(args.ms_root)])
    if args.out_dir is not None:
        api_identity_jsonl = args.out_dir / "api_identity.jsonl"
        api_identity_csv = args.out_dir / "api_identity.csv"
        ms_coverage_jsonl = args.out_dir / "ms_coverage.jsonl"
        ms_coverage_csv = args.out_dir / "ms_coverage.csv"
        op_bundles_jsonl = args.out_dir / "op_bundles.jsonl"
    else:
        api_identity_jsonl = None
        api_identity_csv = None
        ms_coverage_jsonl = None
        ms_coverage_csv = None
        op_bundles_jsonl = None

    api_identity_args = list(shared_args)
    if api_identity_jsonl is not None and api_identity_csv is not None:
        api_identity_args.extend(["--out-jsonl", str(api_identity_jsonl), "--out-csv", str(api_identity_csv)])

    ms_coverage_args = list(shared_args)
    if ms_coverage_jsonl is not None and ms_coverage_csv is not None:
        ms_coverage_args.extend(["--out-jsonl", str(ms_coverage_jsonl), "--out-csv", str(ms_coverage_csv)])

    bundle_args = []
    if api_identity_jsonl is not None and ms_coverage_jsonl is not None and op_bundles_jsonl is not None:
        bundle_args.extend(
            [
                "--api-identity",
                str(api_identity_jsonl),
                "--ms-coverage",
                str(ms_coverage_jsonl),
                "--out-jsonl",
                str(op_bundles_jsonl),
            ]
        )
    if args.bundle_root is not None:
        bundle_args.extend(["--bundle-root", str(args.bundle_root)])

    run("build_api_identity.py", *api_identity_args)
    run("build_ms_coverage.py", *ms_coverage_args)
    run("build_bundles.py", *bundle_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
