from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from common import DEFAULT_OUT_DIR, OPERATOR_FACTS_ROOT


DEFAULT_BUNDLE_ROOT = OPERATOR_FACTS_ROOT / "bundles" / "entries"


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def coverage_view(unit_row: dict) -> dict:
    return {
        "aclnn": list(unit_row.get("aclnn", [])),
        "infer": bool(unit_row.get("infer", False)),
        "kbk": bool(unit_row.get("kbk", False)),
        "pyboost": bool(unit_row.get("pyboost", False)),
        "bprop": bool(unit_row.get("bprop", False)),
        "bprop_units": list(unit_row.get("bprop_units", [])),
    }


def entry_view(entry_row: dict) -> dict:
    return {
        "public_api": entry_row["public_api"],
        "public_surface": entry_row["public_surface"],
        "public_name": entry_row["public_name"],
        "entry_type": entry_row["entry_type"],
        "source_type": entry_row["source_type"],
        "source_path": entry_row["source_path"],
    }


def branch_item(edge_row: dict, unit_row: dict) -> dict:
    return {
        "route_type": edge_row["edge_type"],
        "resolver_type": edge_row["resolver_type"],
        "target_symbol": edge_row["target_symbol"],
        "dispatch_order": edge_row["dispatch_order"],
        "unit_id": unit_row["unit_id"],
        "unit_name": unit_row["unit_name"],
        "op": unit_row["op"],
        "primitive": unit_row["primitive"],
        "yaml_path": unit_row["yaml_path"],
        "coverage": coverage_view(unit_row),
    }


def component_item(graph_row: dict, unit_by_id: Dict[str, dict]) -> dict:
    base = {
        "component_type": graph_row["child_ref_type"],
        "condition": graph_row["condition"],
        "via_symbol": graph_row["via_symbol"],
    }
    child_ref_type = graph_row["child_ref_type"]
    if child_ref_type == "public_api":
        base["public_api"] = graph_row["child_ref"]
        return base
    if child_ref_type == "primitive_symbol":
        base["primitive_symbol"] = graph_row["child_ref"]
        return base

    unit_row = unit_by_id[graph_row["child_ref"]]
    base.update(
        {
            "unit_id": unit_row["unit_id"],
            "unit_name": unit_row["unit_name"],
            "unit_type": unit_row["unit_type"],
        }
    )
    if unit_row["unit_type"] == "branch":
        base.update(
            {
                "op": unit_row["op"],
                "primitive": unit_row["primitive"],
                "yaml_path": unit_row["yaml_path"],
                "coverage": coverage_view(unit_row),
            }
        )
        return base

    if unit_row["unit_type"] == "composite":
        base["impl_path"] = unit_row["impl_path"]
        return base

    return base


def build_bundle(
    entry_row: dict,
    edge_rows: List[dict],
    unit_by_id: Dict[str, dict],
    graph_by_parent: Dict[str, List[dict]],
    generated_at: str,
) -> dict:
    bundle = {
        "bundle_id": entry_row["public_api"],
        "bundle_type": "entry",
        "schema_version": "v2",
        "generated_at": generated_at,
        "entry": entry_view(entry_row),
    }

    sorted_edges = sorted(edge_rows, key=lambda item: (item["dispatch_order"], item["edge_id"]))
    if not sorted_edges:
        return bundle

    composite_route = next(
        (row for row in sorted_edges if unit_by_id.get(row["unit_id"], {}).get("unit_type") == "composite"),
        None,
    )
    if composite_route is not None:
        unit_row = unit_by_id.get(composite_route["unit_id"], {})
        graph_rows = graph_by_parent.get(composite_route["unit_id"], [])
        bundle["composite"] = {
            "resolver_type": composite_route["resolver_type"],
            "target_symbol": composite_route["target_symbol"],
            "impl_path": unit_row.get("impl_path", ""),
            "components": [
                component_item(graph_row, unit_by_id)
                for graph_row in sorted(graph_rows, key=lambda item: (item["call_order"], item["graph_edge_id"]))
            ],
        }
        return bundle

    branch_edges = [edge_row for edge_row in sorted_edges if unit_by_id.get(edge_row["unit_id"], {}).get("unit_type") == "branch"]
    if branch_edges:
        bundle["branches"] = [
            branch_item(edge_row, unit_by_id[edge_row["unit_id"]])
            for edge_row in branch_edges
            if edge_row["unit_id"] in unit_by_id
        ]
        return bundle
    return bundle


def bundle_output_path(bundle_root: Path, public_api: str) -> Path:
    return bundle_root / f"{public_api}.json"


def clear_bundle_files(bundle_root: Path) -> None:
    if not bundle_root.exists():
        return
    for path in bundle_root.rglob("*.json"):
        if path.is_file():
            path.unlink()


def write_bundle_files(bundle_root: Path, bundles: Iterable[dict]) -> None:
    clear_bundle_files(bundle_root)
    for bundle in bundles:
        out_path = bundle_output_path(bundle_root, bundle["bundle_id"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_rows(
    entry_rows: List[dict],
    unit_rows: List[dict],
    edge_rows: List[dict],
    graph_rows: List[dict],
    generated_at: str,
) -> List[dict]:
    unit_by_id = {row["unit_id"]: row for row in unit_rows}
    edges_by_entry: Dict[str, List[dict]] = {}
    for row in edge_rows:
        edges_by_entry.setdefault(row["entry_id"], []).append(row)
    graph_by_parent: Dict[str, List[dict]] = {}
    for row in graph_rows:
        graph_by_parent.setdefault(row["parent_unit_id"], []).append(row)

    bundles: List[dict] = []
    for entry_row in sorted(entry_rows, key=lambda item: item["entry_id"]):
        bundles.append(
            build_bundle(
                entry_row=entry_row,
                edge_rows=edges_by_entry.get(entry_row["entry_id"], []),
                unit_by_id=unit_by_id,
                graph_by_parent=graph_by_parent,
                generated_at=generated_at,
            )
        )
    return bundles


def main() -> int:
    parser = argparse.ArgumentParser(description="Build entry bundles from refactored MindSpore operator-facts tables.")
    parser.add_argument("--entry-identity", type=Path, default=DEFAULT_OUT_DIR / "ms_entry_identity.jsonl")
    parser.add_argument("--unit-identity", type=Path, default=DEFAULT_OUT_DIR / "ms_unit_identity.jsonl")
    parser.add_argument("--entry-unit-edges", type=Path, default=DEFAULT_OUT_DIR / "ms_entry_unit_edges.jsonl")
    parser.add_argument("--unit-graph-edges", type=Path, default=DEFAULT_OUT_DIR / "ms_unit_graph_edges.jsonl")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    args = parser.parse_args()

    entry_rows = read_jsonl(args.entry_identity)
    unit_rows = read_jsonl(args.unit_identity)
    edge_rows = read_jsonl(args.entry_unit_edges)
    graph_rows = read_jsonl(args.unit_graph_edges)
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    bundles = build_rows(entry_rows, unit_rows, edge_rows, graph_rows, generated_at)
    write_bundle_files(args.bundle_root, bundles)

    print(f"entry_bundle_rows={len(bundles)}")
    print(f"bundle_root={args.bundle_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
