#!/usr/bin/env python3
"""Apply LLM gap judgement results back to chains/key_chains outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply ACLNN gap LLM judgement backfill")
    p.add_argument("--chains-json", type=Path, required=True)
    p.add_argument("--llm-results-json", type=Path, required=True)
    p.add_argument("--out-chains-json", type=Path, default=None)
    p.add_argument("--key-json", type=Path, default=None, help="optional key_chains.json to update")
    p.add_argument("--out-key-json", type=Path, default=None)
    return p.parse_args()


def load_llm_results(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    items = obj if isinstance(obj, list) else obj.get("results", [])
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        op = str(it.get("operator", "")).strip()
        api = str(it.get("aclnn_api", "")).strip()
        if not op or not api:
            continue
        out[(op, api)] = {
            "likely_related": bool(it.get("likely_related", False)),
            "confidence": (None if it.get("confidence", None) is None else float(it.get("confidence", 0.0) or 0.0)),
            "reason": str(it.get("reason", "")),
        }
    return out


def main() -> int:
    args = parse_args()
    rows = json.loads(args.chains_json.read_text(encoding="utf-8"))
    llm_map = load_llm_results(args.llm_results_json)

    for r in rows:
        op = str(r.get("operator", ""))
        c = (r.get("aclnn_completeness") or {})
        gaps = c.get("gap_candidates", []) or []
        observed = set(c.get("observed_apis", []) or [])
        suspected: list[str] = []
        new_gaps: list[dict[str, Any]] = []
        for g in gaps:
            api = str((g or {}).get("aclnn_api", ""))
            j = llm_map.get((op, api)) or {}
            likely = bool(j.get("likely_related", False))
            if likely:
                suspected.append(api)
            rec = dict(g)
            rec["llm_judgement"] = {
                "likely_related": likely,
                "confidence": j.get("confidence", None),
                "reason": j.get("reason", ""),
            }
            new_gaps.append(rec)
        c["gap_candidates"] = new_gaps
        c["suspected_missing_apis"] = sorted(set(suspected))
        c["final_api_catalog"] = sorted(observed | set(suspected))
        c["judge_source"] = "skill_backfill"
        r["aclnn_completeness"] = c

    out_chains = args.out_chains_json or args.chains_json
    out_chains.parent.mkdir(parents=True, exist_ok=True)
    out_chains.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.key_json:
        key_rows = json.loads(args.key_json.read_text(encoding="utf-8"))
        by_op = {str(r.get("operator", "")): r for r in rows}
        for k in key_rows:
            op = str(k.get("operator", ""))
            src = by_op.get(op) or {}
            c = src.get("aclnn_completeness", {}) or {}
            k["aclnn_mapping_catalog"] = c.get("final_api_catalog", [])
            k["aclnn_gap_suspects"] = c.get("suspected_missing_apis", [])
        out_key = args.out_key_json or args.key_json
        out_key.parent.mkdir(parents=True, exist_ok=True)
        out_key.write_text(json.dumps(key_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(str(out_chains))
        print(str(out_key))
    else:
        print(str(out_chains))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

