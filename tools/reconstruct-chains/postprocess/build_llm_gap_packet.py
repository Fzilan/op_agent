#!/usr/bin/env python3
"""Build a compact LLM review packet from chains.json + gap candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LLM packet for ACLNN gap review")
    p.add_argument("--chains-json", type=Path, required=True)
    p.add_argument("--gap-json", type=Path, required=True, help="output from extract_gap_candidates.py")
    p.add_argument("--out-json", type=Path, required=True)
    return p.parse_args()


def short_path(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "aclnn_api": p.get("aclnn_api"),
        "dispatch_note": p.get("dispatch_note"),
        "chain": [str(x.get("name", "")) for x in (p.get("chain") or [])],
        "path_conditions": p.get("path_conditions", [])[:6],
        "endpoint": p.get("endpoint", {}),
    }


def main() -> int:
    args = parse_args()
    chains = json.loads(args.chains_json.read_text(encoding="utf-8"))
    gaps_obj = json.loads(args.gap_json.read_text(encoding="utf-8"))
    gap_rows = gaps_obj if isinstance(gaps_obj, list) else gaps_obj.get("results", [])
    gap_by_op = {str(x.get("operator", "")): x for x in gap_rows if isinstance(x, dict)}

    out_rows: list[dict[str, Any]] = []
    for r in chains:
        op = str(r.get("operator", ""))
        if not op or op not in gap_by_op:
            continue
        c = (r.get("aclnn_completeness") or {})
        g = gap_by_op[op]
        out_rows.append(
            {
                "operator": op,
                "front_signatures": r.get("front_signatures", []),
                "dispatch_summary": r.get("dispatch_summary", []),
                "observed_aclnn_apis": c.get("observed_apis", []),
                "observed_paths": [short_path(p) for p in (r.get("paths") or [])],
                "gap_candidates": g.get("gap_candidates", []),
                "review_focus": {
                    "semantic_set": "front_signatures + overload hints in path_conditions",
                    "condition_set": "path_conditions around ACLNN call edges",
                    "aclnn_mapping": "observed_aclnn_apis + observed_paths",
                    "directness": "dispatch_note/dispatch_summary strict_direct/helper/logic_preprocessed",
                },
            }
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"results": out_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(args.out_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

