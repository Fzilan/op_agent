from __future__ import annotations

import argparse
from pathlib import Path

from build_ms_entry_identity import build_rows as build_entry_identity_rows
from build_ms_entry_unit_edges import build_rows as build_entry_unit_edge_rows
from build_ms_unit_graph_edges import build_rows as build_unit_graph_rows
from build_ms_unit_identity import build_branch_rows, build_composite_rows, write_csv_rows
from common import DEFAULT_MS_ROOT, DEFAULT_OUT_DIR, write_csv, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Build all refactored MindSpore operator-facts tables.")
    parser.add_argument("--ms-root", type=Path, default=DEFAULT_MS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entry_identity_jsonl = out_dir / "ms_entry_identity.jsonl"
    entry_identity_csv = out_dir / "ms_entry_identity.csv"
    entry_unit_edges_jsonl = out_dir / "ms_entry_unit_edges.jsonl"
    entry_unit_edges_csv = out_dir / "ms_entry_unit_edges.csv"
    unit_graph_edges_jsonl = out_dir / "ms_unit_graph_edges.jsonl"
    unit_graph_edges_csv = out_dir / "ms_unit_graph_edges.csv"
    unit_identity_jsonl = out_dir / "ms_unit_identity.jsonl"
    unit_identity_csv = out_dir / "ms_unit_identity.csv"

    entry_identity_rows = build_entry_identity_rows(args.ms_root)
    write_jsonl(entry_identity_jsonl, entry_identity_rows)
    write_csv(
        entry_identity_csv,
        entry_identity_rows,
        [
            "entry_id",
            "public_api",
            "public_surface",
            "public_name",
            "entry_type",
            "source_type",
            "source_path",
        ],
    )

    entry_unit_edge_rows = build_entry_unit_edge_rows(args.ms_root)
    write_jsonl(entry_unit_edges_jsonl, entry_unit_edge_rows)
    write_csv(
        entry_unit_edges_csv,
        entry_unit_edge_rows,
        [
            "edge_id",
            "entry_id",
            "unit_id",
            "edge_type",
            "dispatch_order",
            "resolver_type",
            "resolver_path",
            "match_condition",
            "target_symbol",
        ],
    )

    unit_graph_rows = build_unit_graph_rows(args.ms_root, entry_unit_edges_jsonl)
    write_jsonl(unit_graph_edges_jsonl, unit_graph_rows)
    write_csv(
        unit_graph_edges_csv,
        unit_graph_rows,
        [
            "graph_edge_id",
            "parent_unit_id",
            "child_ref_type",
            "child_ref",
            "call_order",
            "condition",
            "via_symbol",
            "via_path",
        ],
    )

    branch_rows = build_branch_rows(args.ms_root)
    composite_rows = build_composite_rows(
        args.ms_root,
        branch_rows=branch_rows,
        entry_edges_path=entry_unit_edges_jsonl,
        graph_edges_path=unit_graph_edges_jsonl,
    )
    unit_identity_rows = sorted(branch_rows + composite_rows, key=lambda item: (item["unit_type"], item["unit_id"]))
    write_jsonl(unit_identity_jsonl, unit_identity_rows)
    write_csv_rows(
        unit_identity_csv,
        unit_identity_rows,
        [
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
        ],
    )

    print(f"ms_entry_identity_rows={len(entry_identity_rows)}")
    print(f"ms_entry_unit_edges_rows={len(entry_unit_edge_rows)}")
    print(f"ms_unit_graph_edges_rows={len(unit_graph_rows)}")
    print(f"ms_unit_identity_rows={len(unit_identity_rows)}")
    print(f"branch_units={len(branch_rows)}")
    print(f"composite_units={len(composite_rows)}")
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
