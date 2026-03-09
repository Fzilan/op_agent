#!/usr/bin/env python3
"""Extract ACLNN gap candidates from chains.json for skill/LLM step."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract ACLNN gap candidates from chains.json")
    p.add_argument("--chains-json", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = json.loads(args.chains_json.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for r in rows:
        op = str(r.get("operator", ""))
        c = (r.get("aclnn_completeness") or {})
        gaps = c.get("gap_candidates", []) or []
        if not gaps:
            continue
        out.append(
            {
                "operator": op,
                "front_signatures": c.get("front_signatures", []),
                "observed_aclnn_apis": c.get("observed_apis", []),
                "gap_candidates": gaps,
            }
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"results": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(args.out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

