from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from common import DEFAULT_MS_ROOT, DEFAULT_OUT_DIR, OpDefEntry, build_op_catalog, write_csv, write_jsonl
from unit_coverage_scan import (
    build_candidate_keys,
    load_aclnn_config,
    match_named_files,
    resolve_dispatch_kind,
    scan_bprop,
    scan_infer,
    scan_kernel_auto_generate,
    scan_kernel_customize,
    scan_pyboost_auto_generate,
    scan_pyboost_customize,
)


def branch_unit_id(entry: OpDefEntry) -> str:
    return f"branch::{entry.op_branch}::{entry.primitive}"


def branch_display_id(entry: OpDefEntry) -> str:
    return f"branch::{entry.primitive}"


def composite_unit_id(ms_root: Path, impl_path: Path, impl_symbol: str) -> str:
    short_path = str(impl_path.relative_to(ms_root / "python" / "mindspore"))
    return f"composite::{short_path}::{impl_symbol}"


def composite_display_id(unit_name: str) -> str:
    return f"composite::{unit_name}"


def parse_composite_unit_id(ms_root: Path, unit_id: str) -> Optional[Tuple[Path, str]]:
    if not unit_id.startswith("composite::"):
        return None
    parts = unit_id.split("::", 2)
    if len(parts) != 3:
        return None
    rel_path, symbol = parts[1], parts[2]
    return ms_root / "python" / "mindspore" / rel_path, symbol


def load_jsonl_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def json_cell(value) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_branch_rows(ms_root: Path) -> List[dict]:
    op_by_branch, _ = build_op_catalog(ms_root)
    aclnn_config = load_aclnn_config(ms_root)
    customize_kernel_map, customize_kernel_files = scan_kernel_customize(ms_root)
    _, auto_kernel_files = scan_kernel_auto_generate(ms_root)
    customize_pyboost_map, customize_pyboost_files = scan_pyboost_customize(ms_root)
    auto_pyboost_files = scan_pyboost_auto_generate(ms_root)
    infer_primitives, infer_files = scan_infer(ms_root)
    bprop_files = scan_bprop(ms_root)

    rows: List[dict] = []
    for entry in sorted(op_by_branch.values(), key=lambda item: (item.op, item.primitive)):
        candidate_keys = build_candidate_keys(entry)
        dispatch_type = resolve_dispatch_kind(entry)
        aclnn_values: Set[str] = set()

        if dispatch_type == "customize":
            for key in candidate_keys:
                aclnn_values.update(customize_kernel_map.get(key, set()))
                aclnn_values.update(customize_pyboost_map.get(key, set()))
            pyboost_evidence = match_named_files(candidate_keys, customize_pyboost_files)
            kbk_evidence = match_named_files(candidate_keys, customize_kernel_files)
        elif dispatch_type == "auto_generate":
            mapped = next((aclnn_config.get(key) for key in sorted(candidate_keys) if aclnn_config.get(key)), None)
            if mapped:
                aclnn_values.add(mapped)
            else:
                aclnn_values.add(f"aclnn{entry.class_name}")
            pyboost_evidence = match_named_files(candidate_keys, auto_pyboost_files)
            kbk_evidence = match_named_files(candidate_keys, auto_kernel_files)
        else:
            pyboost_evidence = []
            kbk_evidence = []

        infer_evidence = match_named_files(candidate_keys, infer_files)
        infer_present = entry.primitive in infer_primitives or bool(infer_evidence)
        bprop_evidence = match_named_files(candidate_keys, bprop_files)

        rows.append(
            {
                "unit_id": branch_unit_id(entry),
                "unit_type": "branch",
                "unit_name": entry.primitive,
                "display_id": branch_display_id(entry),
                "op": entry.op,
                "primitive": entry.primitive,
                "yaml_path": entry.op_yaml_path,
                "dispatch_enable": entry.dispatch_enable,
                "dispatch_type": dispatch_type,
                "dispatch_ascend": entry.dispatch_ascend,
                "aclnn": sorted(aclnn_values),
                "infer": infer_present,
                "pyboost": dispatch_type == "auto_generate" or bool(pyboost_evidence),
                "kbk": dispatch_type == "auto_generate" or bool(kbk_evidence),
                "bprop": bool(bprop_evidence),
                "bprop_units": [],
            }
        )
    return rows


def collect_composite_unit_ids(entry_edge_rows: List[dict], graph_rows: List[dict]) -> List[str]:
    unit_ids = {
        row["unit_id"]
        for row in entry_edge_rows
        if row.get("edge_type") == "composite" and str(row.get("unit_id", "")).startswith("composite::")
    }
    unit_ids.update(
        row["child_ref"]
        for row in graph_rows
        if row.get("child_ref_type") == "unit" and str(row.get("child_ref", "")).startswith("composite::")
    )
    return sorted(unit_ids)


def build_graph_index(rows: List[dict]) -> Dict[str, List[dict]]:
    graph_by_parent: Dict[str, List[dict]] = {}
    for row in rows:
        graph_by_parent.setdefault(row["parent_unit_id"], []).append(row)
    for edges in graph_by_parent.values():
        edges.sort(key=lambda item: (item["call_order"], item["graph_edge_id"]))
    return graph_by_parent


def build_entry_edge_index(rows: List[dict]) -> Dict[str, List[dict]]:
    entry_edges: Dict[str, List[dict]] = {}
    for row in rows:
        entry_edges.setdefault(row["entry_id"], []).append(row)
    for edges in entry_edges.values():
        edges.sort(key=lambda item: (item["dispatch_order"], item["edge_id"]))
    return entry_edges


def collect_reachable_leaf_unit_ids(
    unit_id: str,
    graph_by_parent: Dict[str, List[dict]],
    entry_edges_by_entry: Dict[str, List[dict]],
    visiting_units: Optional[Set[str]] = None,
    visiting_entries: Optional[Set[str]] = None,
) -> Set[str]:
    if not str(unit_id).startswith("composite::"):
        return {unit_id}

    if visiting_units is None:
        visiting_units = set()
    if visiting_entries is None:
        visiting_entries = set()
    if unit_id in visiting_units:
        return set()

    edges = graph_by_parent.get(unit_id, [])
    if not edges:
        return set()

    next_visiting_units = set(visiting_units)
    next_visiting_units.add(unit_id)
    leaf_ids: Set[str] = set()

    for edge in edges:
        child_ref_type = edge["child_ref_type"]
        child_ref = edge["child_ref"]
        if child_ref_type == "unit":
            leaf_ids.update(
                collect_reachable_leaf_unit_ids(
                    child_ref,
                    graph_by_parent,
                    entry_edges_by_entry,
                    visiting_units=next_visiting_units,
                    visiting_entries=visiting_entries,
                )
            )
            continue
        if child_ref_type != "public_api" or child_ref in visiting_entries:
            continue
        next_visiting_entries = set(visiting_entries)
        next_visiting_entries.add(child_ref)
        for routed_edge in entry_edges_by_entry.get(child_ref, []):
            routed_unit_id = routed_edge["unit_id"]
            leaf_ids.update(
                collect_reachable_leaf_unit_ids(
                    routed_unit_id,
                    graph_by_parent,
                    entry_edges_by_entry,
                    visiting_units=next_visiting_units,
                    visiting_entries=next_visiting_entries,
                )
            )
    return leaf_ids


def leaf_unit_summary(row: dict) -> dict:
    return {
        "unit_id": row["unit_id"],
        "unit_name": row["unit_name"],
        "unit_type": row["unit_type"],
        "aclnn": list(row.get("aclnn", [])),
        "infer": bool(row.get("infer", False)),
        "pyboost": bool(row.get("pyboost", False)),
        "kbk": bool(row.get("kbk", False)),
        "bprop": bool(row.get("bprop", False)),
        "bprop_units": list(row.get("bprop_units", [])),
    }


def build_composite_rows(
    ms_root: Path,
    branch_rows: List[dict],
    entry_edges_path: Path,
    graph_edges_path: Path,
) -> List[dict]:
    entry_edge_rows = load_jsonl_rows(entry_edges_path)
    graph_rows = load_jsonl_rows(graph_edges_path)
    graph_by_parent = build_graph_index(graph_rows)
    entry_edges_by_entry = build_entry_edge_index(entry_edge_rows)
    branch_rows_by_id = {row["unit_id"]: row for row in branch_rows}

    rows: List[dict] = []
    for unit_id in collect_composite_unit_ids(entry_edge_rows, graph_rows):
        parsed = parse_composite_unit_id(ms_root, unit_id)
        if parsed is None:
            continue
        impl_path, impl_symbol = parsed
        leaf_ids = sorted(
            leaf_id
            for leaf_id in collect_reachable_leaf_unit_ids(
                unit_id,
                graph_by_parent,
                entry_edges_by_entry,
            )
            if leaf_id in branch_rows_by_id
        )
        rows.append(
            {
                "unit_id": unit_id,
                "unit_type": "composite",
                "unit_name": impl_symbol,
                "display_id": composite_display_id(impl_symbol),
                "impl_path": str(impl_path.relative_to(ms_root.parent)),
                "impl_symbol": impl_symbol,
                "direct_aclnn": [],
                "direct_infer": False,
                "direct_pyboost": False,
                "direct_kbk": False,
                "direct_bprop": False,
                "leaf_units": [leaf_unit_summary(branch_rows_by_id[leaf_id]) for leaf_id in leaf_ids],
            }
        )
    return rows


def write_csv_rows(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    csv_rows = []
    for row in rows:
        csv_rows.append({field: json_cell(row.get(field, "")) for field in fieldnames})
    write_csv(path, csv_rows, fieldnames)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ms_unit_identity index for operator-facts.")
    parser.add_argument("--ms-root", type=Path, default=DEFAULT_MS_ROOT)
    parser.add_argument(
        "--entry-unit-edges",
        type=Path,
        default=DEFAULT_OUT_DIR / "ms_entry_unit_edges.jsonl",
    )
    parser.add_argument(
        "--unit-graph-edges",
        type=Path,
        default=DEFAULT_OUT_DIR / "ms_unit_graph_edges.jsonl",
    )
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_DIR / "ms_unit_identity.jsonl")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_DIR / "ms_unit_identity.csv")
    args = parser.parse_args()

    branch_rows = build_branch_rows(args.ms_root)
    composite_rows = build_composite_rows(
        args.ms_root,
        branch_rows=branch_rows,
        entry_edges_path=args.entry_unit_edges,
        graph_edges_path=args.unit_graph_edges,
    )
    rows = sorted(branch_rows + composite_rows, key=lambda item: (item["unit_type"], item["unit_id"]))
    write_jsonl(args.out_jsonl, rows)
    fieldnames = [
        "unit_id",
        "unit_type",
        "unit_name",
        "display_id",
        "op",
        "primitive",
        "yaml_path",
        "impl_path",
        "impl_symbol",
        "compat_source",
        "dispatch_enable",
        "dispatch_type",
        "dispatch_ascend",
        "aclnn",
        "infer",
        "pyboost",
        "kbk",
        "bprop",
        "bprop_units",
        "direct_aclnn",
        "direct_infer",
        "direct_pyboost",
        "direct_kbk",
        "direct_bprop",
        "leaf_units",
    ]
    write_csv_rows(args.out_csv, rows, fieldnames)
    print(f"ms_unit_identity_rows={len(rows)}")
    print(f"branch_units={len(branch_rows)}")
    print(f"composite_units={len(composite_rows)}")
    print(f"out_jsonl={args.out_jsonl}")
    print(f"out_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
