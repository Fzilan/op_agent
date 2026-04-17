from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from build_ms_entry_identity import (
    active_branches,
    extract_tensor_registry_key,
    parse_tensor_registry_bindings,
    rel_source,
)
from common import (
    DEFAULT_MS_ROOT,
    DEFAULT_OUT_DIR,
    ApiBranchEntry,
    OpDefEntry,
    build_api_def_catalog,
    build_op_catalog,
    load_module_info,
    write_csv,
    write_jsonl,
)
from symbol_resolution import resolve_symbol_to_entries


def branch_unit_id(entry: OpDefEntry) -> str:
    return f"branch::{entry.op_branch}::{entry.primitive}"


def composite_unit_id(ms_root: Path, impl_path: Path, impl_symbol: str) -> str:
    short_path = str(impl_path.relative_to(ms_root / "python" / "mindspore"))
    return f"composite::{short_path}::{impl_symbol}"


def edge_id(entry_id: str, edge_type: str, unit_name: str, dispatch_order: int) -> str:
    return f"{entry_id}::{edge_type}::{unit_name}::{dispatch_order}"


def build_function_impl_index(ms_root: Path) -> Dict[str, List[Path]]:
    root = ms_root / "python" / "mindspore" / "ops" / "function"
    index: Dict[str, List[Path]] = {}
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                index.setdefault(node.name, []).append(path)
    return index


def resolve_wrapper_target_symbol(ms_root: Path, module_name: str, func_name: str) -> str:
    info = load_module_info(ms_root, module_name)
    if info is None:
        return func_name
    func = info.functions.get(func_name)
    if func is None:
        return func_name
    for node in ast.walk(func):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
    return func_name


def resolve_tensor_registry_target(
    ms_root: Path,
    registry_key: str,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    registry_bindings: Dict[str, Tuple[str, str]],
    function_impl_index: Dict[str, List[Path]],
    op_by_symbol: Dict[str, List[OpDefEntry]],
) -> Tuple[str, List[Tuple[str, str]], Optional[Tuple[Path, str]], str]:
    active = active_branches(api_def_catalog.get(registry_key, []))
    if active:
        return "branch", [(branch.op_branch, branch.py_method or registry_key) for branch in active], None, registry_key

    binding = registry_bindings.get(registry_key)
    if binding is None:
        return "unresolved", [], None, registry_key

    target_module, target_symbol = binding
    info = load_module_info(ms_root, target_module)
    if info is not None and target_symbol in info.functions and info.path.name != "__init__.py":
        return "composite", [], (info.path, target_symbol), target_symbol

    impl_paths = function_impl_index.get(target_symbol, [])
    if len(impl_paths) == 1:
        return "composite", [], (impl_paths[0], target_symbol), target_symbol

    resolved = resolve_symbol_to_entries(
        ms_root=ms_root,
        target_module=target_module,
        target_symbol=target_symbol,
        api_def_catalog=api_def_catalog,
        op_by_symbol=op_by_symbol,
    )
    if resolved:
        return (
            "branch",
            [(row["op_branch"], row.get("py_method") or target_symbol) for row in resolved if row.get("op_branch")],
            None,
            target_symbol,
        )
    if target_module == "mindspore.ops.functional":
        return "composite", [], None, target_symbol
    return "unresolved", [], None, target_symbol


def build_api_def_edges(ms_root: Path, api_def_catalog: Dict[str, List[ApiBranchEntry]], op_by_branch: Dict[str, OpDefEntry]) -> List[dict]:
    rows: List[dict] = []
    for api_name, branches in sorted(api_def_catalog.items()):
        active = active_branches(branches)
        if not active:
            continue
        interfaces: Set[str] = set()
        for branch in active:
            interfaces.update(item.strip() for item in branch.interface.split(",") if item.strip())
        branch_count = len(active)
        edge_type = "direct" if branch_count == 1 else "overload"
        active_sorted = sorted(active, key=lambda item: item.branch_order)

        for surface in sorted(interfaces):
            if surface == "function":
                entry_id = f"mindspore.ops.{api_name}"
                public_surface = "mindspore.ops"
            elif surface == "tensor":
                entry_id = f"mindspore.Tensor.{api_name}"
                public_surface = "mindspore.Tensor"
            else:
                continue
            for dispatch_order, branch in enumerate(active_sorted):
                unit = op_by_branch.get(branch.op_branch)
                if unit is None:
                    continue
                resolver_type = "py_method" if branch.py_method else "api_def"
                rows.append(
                    {
                        "edge_id": edge_id(entry_id, edge_type, unit.primitive, dispatch_order),
                        "entry_id": entry_id,
                        "unit_id": branch_unit_id(unit),
                        "edge_type": edge_type,
                        "dispatch_order": dispatch_order,
                        "resolver_type": resolver_type,
                        "resolver_path": branch.source_file,
                        "match_condition": "",
                        "target_symbol": branch.py_method or api_name,
                    }
                )
    return rows


def build_mint_edges(
    ms_root: Path,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    op_by_branch: Dict[str, OpDefEntry],
    op_by_symbol: Dict[str, List[OpDefEntry]],
) -> List[dict]:
    mint_info = load_module_info(ms_root, "mindspore.mint")
    if mint_info is None:
        return []
    source_path = rel_source(mint_info.path, ms_root)
    rows: List[dict] = []

    for public_name, binding in sorted(mint_info.from_imports.items()):
        if not binding.module.startswith("mindspore.ops"):
            continue
        resolved = resolve_symbol_to_entries(
            ms_root=ms_root,
            target_module=binding.module,
            target_symbol=binding.symbol or public_name,
            api_def_catalog=api_def_catalog,
            op_by_symbol=op_by_symbol,
        )
        branch_rows = sorted({row["op_branch"]: row for row in resolved if row.get("op_branch")}.values(), key=lambda item: item["op_branch"])
        if not branch_rows:
            continue
        edge_type = "direct" if len(branch_rows) == 1 else "overload"
        entry_id = f"mindspore.mint.{public_name}"
        for dispatch_order, row in enumerate(branch_rows):
            unit = op_by_branch.get(row["op_branch"])
            if unit is None:
                continue
            rows.append(
                {
                    "edge_id": edge_id(entry_id, edge_type, unit.primitive, dispatch_order),
                    "entry_id": entry_id,
                    "unit_id": branch_unit_id(unit),
                    "edge_type": edge_type,
                    "dispatch_order": dispatch_order,
                    "resolver_type": "alias",
                    "resolver_path": source_path,
                    "match_condition": "",
                    "target_symbol": binding.symbol or public_name,
                }
            )

    for public_name in sorted(mint_info.functions):
        if public_name.startswith("_"):
            continue
        resolved = resolve_symbol_to_entries(
            ms_root=ms_root,
            target_module="mindspore.mint",
            target_symbol=public_name,
            api_def_catalog=api_def_catalog,
            op_by_symbol=op_by_symbol,
        )
        branch_rows = sorted({row["op_branch"]: row for row in resolved if row.get("op_branch")}.values(), key=lambda item: item["op_branch"])
        if not branch_rows:
            continue
        edge_type = "direct" if len(branch_rows) == 1 else "overload"
        entry_id = f"mindspore.mint.{public_name}"
        target_symbol = resolve_wrapper_target_symbol(ms_root, "mindspore.mint", public_name)
        for dispatch_order, row in enumerate(branch_rows):
            unit = op_by_branch.get(row["op_branch"])
            if unit is None:
                continue
            rows.append(
                {
                    "edge_id": edge_id(entry_id, edge_type, unit.primitive, dispatch_order),
                    "entry_id": entry_id,
                    "unit_id": branch_unit_id(unit),
                    "edge_type": edge_type,
                    "dispatch_order": dispatch_order,
                    "resolver_type": "wrapper",
                    "resolver_path": source_path,
                    "match_condition": "",
                    "target_symbol": target_symbol,
                }
            )
    return rows


def build_ops_function_edges(
    ms_root: Path,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    op_by_branch: Dict[str, OpDefEntry],
    op_by_symbol: Dict[str, List[OpDefEntry]],
) -> List[dict]:
    info = load_module_info(ms_root, "mindspore.ops.function")
    if info is None:
        return []
    source_path = rel_source(info.path, ms_root)
    function_impl_index = build_function_impl_index(ms_root)
    rows: List[dict] = []

    for public_name, binding in sorted(info.from_imports.items()):
        if public_name.startswith("_"):
            continue
        target_symbol = binding.symbol or public_name
        resolved = resolve_symbol_to_entries(
            ms_root=ms_root,
            target_module=binding.module,
            target_symbol=target_symbol,
            api_def_catalog=api_def_catalog,
            op_by_symbol=op_by_symbol,
        )
        branch_rows = sorted({row["op_branch"]: row for row in resolved if row.get("op_branch")}.values(), key=lambda item: item["op_branch"])
        entry_id = f"mindspore.ops.{public_name}"
        if branch_rows:
            edge_type = "direct" if len(branch_rows) == 1 else "overload"
            for dispatch_order, row in enumerate(branch_rows):
                unit = op_by_branch.get(row["op_branch"])
                if unit is None:
                    continue
                rows.append(
                    {
                        "edge_id": edge_id(entry_id, edge_type, unit.primitive, dispatch_order),
                        "entry_id": entry_id,
                        "unit_id": branch_unit_id(unit),
                        "edge_type": edge_type,
                        "dispatch_order": dispatch_order,
                        "resolver_type": "alias",
                        "resolver_path": source_path,
                        "match_condition": "",
                        "target_symbol": target_symbol,
                    }
                )
            continue

        target_info = load_module_info(ms_root, binding.module)
        if target_info is not None and target_symbol in target_info.functions and target_info.path.name != "__init__.py":
            rows.append(
                {
                    "edge_id": edge_id(entry_id, "composite", target_symbol, 0),
                    "entry_id": entry_id,
                    "unit_id": composite_unit_id(ms_root, target_info.path, target_symbol),
                    "edge_type": "composite",
                    "dispatch_order": 0,
                    "resolver_type": "alias",
                    "resolver_path": source_path,
                    "match_condition": "",
                    "target_symbol": target_symbol,
                }
            )
            continue

        impl_paths = function_impl_index.get(target_symbol, [])
        if len(impl_paths) == 1:
            rows.append(
                {
                    "edge_id": edge_id(entry_id, "composite", target_symbol, 0),
                    "entry_id": entry_id,
                    "unit_id": composite_unit_id(ms_root, impl_paths[0], target_symbol),
                    "edge_type": "composite",
                    "dispatch_order": 0,
                    "resolver_type": "alias",
                    "resolver_path": source_path,
                    "match_condition": "",
                    "target_symbol": target_symbol,
                }
            )

    return rows


def build_tensor_class_edges(
    ms_root: Path,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    op_by_branch: Dict[str, OpDefEntry],
    op_by_symbol: Dict[str, List[OpDefEntry]],
    existing_entry_ids: Set[str],
) -> List[dict]:
    path = ms_root / "python" / "mindspore" / "common" / "tensor.py"
    functional_path = rel_source(ms_root / "python" / "mindspore" / "ops" / "functional.py", ms_root)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    tensor_class = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Tensor"), None)
    if tensor_class is None:
        return []

    registry_bindings = parse_tensor_registry_bindings(ms_root)
    function_impl_index = build_function_impl_index(ms_root)
    rows: List[dict] = []

    for node in tensor_class.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        method_name = node.name
        if method_name.startswith("_"):
            continue
        entry_id = f"mindspore.Tensor.{method_name}"
        if entry_id in existing_entry_ids:
            continue
        registry_key = extract_tensor_registry_key(node)
        if not registry_key:
            continue

        target_kind, branch_targets, composite_target, target_symbol = resolve_tensor_registry_target(
            ms_root=ms_root,
            registry_key=registry_key,
            api_def_catalog=api_def_catalog,
            registry_bindings=registry_bindings,
            function_impl_index=function_impl_index,
            op_by_symbol=op_by_symbol,
        )
        if target_kind == "branch":
            branch_targets_sorted = sorted(branch_targets, key=lambda item: item[0])
            edge_type = "direct" if len(branch_targets_sorted) == 1 else "overload"
            for dispatch_order, (op_branch, branch_symbol) in enumerate(branch_targets_sorted):
                unit = op_by_branch.get(op_branch)
                if unit is None:
                    continue
                rows.append(
                    {
                        "edge_id": edge_id(entry_id, edge_type, unit.primitive, dispatch_order),
                        "entry_id": entry_id,
                        "unit_id": branch_unit_id(unit),
                        "edge_type": edge_type,
                        "dispatch_order": dispatch_order,
                        "resolver_type": "tensor_registry",
                        "resolver_path": functional_path,
                        "match_condition": "",
                        "target_symbol": branch_symbol,
                    }
                )
        elif target_kind == "composite":
            impl_path, impl_symbol = composite_target if composite_target is not None else (
                function_impl_index.get(target_symbol, [None])[0],
                target_symbol,
            )
            if impl_path is None:
                continue
            rows.append(
                {
                    "edge_id": edge_id(entry_id, "composite", impl_symbol, 0),
                    "entry_id": entry_id,
                    "unit_id": composite_unit_id(ms_root, impl_path, impl_symbol),
                    "edge_type": "composite",
                    "dispatch_order": 0,
                    "resolver_type": "tensor_registry",
                    "resolver_path": functional_path,
                    "match_condition": "",
                    "target_symbol": target_symbol,
                }
            )
    return rows


def dedupe_rows(rows: Iterable[dict]) -> List[dict]:
    dedup: Dict[str, dict] = {}
    for row in rows:
        dedup[row["edge_id"]] = row
    return sorted(dedup.values(), key=lambda item: (item["entry_id"], item["dispatch_order"], item["unit_id"]))


def build_rows(ms_root: Path) -> List[dict]:
    op_by_branch, op_by_symbol = build_op_catalog(ms_root)
    api_def_catalog = build_api_def_catalog(ms_root)

    rows: List[dict] = []
    rows.extend(build_api_def_edges(ms_root, api_def_catalog, op_by_branch))
    rows.extend(build_ops_function_edges(ms_root, api_def_catalog, op_by_branch, op_by_symbol))
    rows.extend(build_mint_edges(ms_root, api_def_catalog, op_by_branch, op_by_symbol))
    existing_entry_ids = {row["entry_id"] for row in rows}
    rows.extend(build_tensor_class_edges(ms_root, api_def_catalog, op_by_branch, op_by_symbol, existing_entry_ids))
    return dedupe_rows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ms_entry_unit_edges index for operator-facts.")
    parser.add_argument("--ms-root", type=Path, default=DEFAULT_MS_ROOT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_DIR / "ms_entry_unit_edges.jsonl")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_DIR / "ms_entry_unit_edges.csv")
    args = parser.parse_args()

    rows = build_rows(args.ms_root)

    write_jsonl(args.out_jsonl, rows)
    fieldnames = [
        "edge_id",
        "entry_id",
        "unit_id",
        "edge_type",
        "dispatch_order",
        "resolver_type",
        "resolver_path",
        "match_condition",
        "target_symbol",
    ]
    write_csv(args.out_csv, rows, fieldnames)
    print(f"ms_entry_unit_edges_rows={len(rows)}")
    print(f"out_jsonl={args.out_jsonl}")
    print(f"out_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
