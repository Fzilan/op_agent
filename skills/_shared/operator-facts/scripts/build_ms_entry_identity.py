from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from common import (
    DEFAULT_MS_ROOT,
    DEFAULT_OUT_DIR,
    ApiBranchEntry,
    build_api_def_catalog,
    build_op_catalog,
    load_module_info,
    write_csv,
    write_jsonl,
)
from symbol_resolution import resolve_symbol_to_entries


SOURCE_PRIORITY = {
    "api_def": 0,
    "wrapper": 1,
    "mint_export": 2,
    "tensor_class": 3,
    "tensor_method": 4,
    "tensor_registry": 5,
    "unknown": 6,
}


def rel_source(path: Path, ms_root: Path) -> str:
    return str(path.relative_to(ms_root.parent))


def active_branches(branches: Iterable[ApiBranchEntry]) -> List[ApiBranchEntry]:
    return [branch for branch in branches if not branch.deprecated]


def classify_branch_count(count: int) -> str:
    if count <= 0:
        return "unresolved"
    if count == 1:
        return "single"
    return "overload"


def classify_resolved_rows(rows: List[dict]) -> str:
    branches = {row["op_branch"] for row in rows if row.get("op_branch")}
    return classify_branch_count(len(branches))


def add_entry(rows_by_id: Dict[str, dict], row: dict) -> None:
    current = rows_by_id.get(row["entry_id"])
    if current is None:
        rows_by_id[row["entry_id"]] = row
        return
    current_priority = SOURCE_PRIORITY.get(current["source_type"], 99)
    next_priority = SOURCE_PRIORITY.get(row["source_type"], 99)
    if next_priority < current_priority:
        rows_by_id[row["entry_id"]] = row


def build_api_def_entries(ms_root: Path, api_def_catalog: Dict[str, List[ApiBranchEntry]]) -> List[dict]:
    rows: List[dict] = []
    for api_name, branches in sorted(api_def_catalog.items()):
        active = active_branches(branches)
        if not active:
            continue
        interfaces: Set[str] = set()
        for branch in active:
            interfaces.update(item.strip() for item in branch.interface.split(",") if item.strip())
        entry_type = classify_branch_count(len({branch.op_branch for branch in active}))
        source_path = active[0].source_file
        if "function" in interfaces:
            rows.append(
                {
                    "entry_id": f"mindspore.ops.{api_name}",
                    "public_api": f"mindspore.ops.{api_name}",
                    "public_surface": "mindspore.ops",
                    "public_name": api_name,
                    "entry_type": entry_type,
                    "source_type": "api_def",
                    "source_path": source_path,
                }
            )
        if "tensor" in interfaces:
            rows.append(
                {
                    "entry_id": f"mindspore.Tensor.{api_name}",
                    "public_api": f"mindspore.Tensor.{api_name}",
                    "public_surface": "mindspore.Tensor",
                    "public_name": api_name,
                    "entry_type": entry_type,
                    "source_type": "api_def",
                    "source_path": source_path,
                }
            )
    return rows


def build_mint_entries(
    ms_root: Path,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    op_by_symbol: Dict[str, List],
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
        rows.append(
            {
                "entry_id": f"mindspore.mint.{public_name}",
                "public_api": f"mindspore.mint.{public_name}",
                "public_surface": "mindspore.mint",
                "public_name": public_name,
                "entry_type": classify_resolved_rows(resolved),
                "source_type": "mint_export",
                "source_path": source_path,
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
        rows.append(
            {
                "entry_id": f"mindspore.mint.{public_name}",
                "public_api": f"mindspore.mint.{public_name}",
                "public_surface": "mindspore.mint",
                "public_name": public_name,
                "entry_type": classify_resolved_rows(resolved),
                "source_type": "wrapper",
                "source_path": source_path,
            }
        )

    return rows


def classify_export_target(
    ms_root: Path,
    target_module: str,
    target_symbol: str,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    op_by_symbol: Dict[str, List],
) -> str:
    resolved = resolve_symbol_to_entries(
        ms_root=ms_root,
        target_module=target_module,
        target_symbol=target_symbol,
        api_def_catalog=api_def_catalog,
        op_by_symbol=op_by_symbol,
    )
    if resolved:
        return classify_resolved_rows(resolved)

    info = load_module_info(ms_root, target_module)
    if info is not None and target_symbol in info.functions and info.path.name != "__init__.py":
        return "composite"
    return "unresolved"


def build_ops_function_entries(
    ms_root: Path,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    op_by_symbol: Dict[str, List],
) -> List[dict]:
    info = load_module_info(ms_root, "mindspore.ops.function")
    if info is None:
        return []
    source_path = rel_source(info.path, ms_root)
    rows: List[dict] = []

    for public_name, binding in sorted(info.from_imports.items()):
        if public_name.startswith("_"):
            continue
        rows.append(
            {
                "entry_id": f"mindspore.ops.{public_name}",
                "public_api": f"mindspore.ops.{public_name}",
                "public_surface": "mindspore.ops",
                "public_name": public_name,
                "entry_type": classify_export_target(
                    ms_root=ms_root,
                    target_module=binding.module,
                    target_symbol=binding.symbol or public_name,
                    api_def_catalog=api_def_catalog,
                    op_by_symbol=op_by_symbol,
                ),
                "source_type": "wrapper",
                "source_path": source_path,
            }
        )

    for public_name in sorted(info.functions):
        if public_name.startswith("_"):
            continue
        rows.append(
            {
                "entry_id": f"mindspore.ops.{public_name}",
                "public_api": f"mindspore.ops.{public_name}",
                "public_surface": "mindspore.ops",
                "public_name": public_name,
                "entry_type": "composite",
                "source_type": "wrapper",
                "source_path": source_path,
            }
        )

    return rows


def parse_tensor_registry_bindings(ms_root: Path) -> Dict[str, Tuple[str, str]]:
    info = load_module_info(ms_root, "mindspore.ops.functional")
    if info is None:
        return {}
    try:
        tree = ast.parse(info.path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    bindings: Dict[str, Tuple[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not isinstance(call.func, ast.Name) or call.func.id != "setattr":
            continue
        if len(call.args) != 3:
            continue
        base, key_arg, target = call.args
        if not isinstance(base, ast.Name) or base.id != "tensor_operator_registry":
            continue
        if not isinstance(key_arg, ast.Constant) or not isinstance(key_arg.value, str):
            continue
        registry_key = key_arg.value
        if isinstance(target, ast.Name):
            if target.id in info.from_imports:
                binding = info.from_imports[target.id]
                bindings[registry_key] = (binding.module, binding.symbol or target.id)
            elif target.id in info.functions:
                bindings[registry_key] = (info.module_name, target.id)
            else:
                # functional.py re-exports many symbols through star imports.
                # Keep the local symbol so later classification can still treat it
                # as a composite-style registry target instead of dropping it.
                bindings[registry_key] = (info.module_name, target.id)
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            base_name = target.value.id
            if base_name in info.imports:
                bindings[registry_key] = (info.imports[base_name], target.attr)
            elif base_name in info.from_imports:
                binding = info.from_imports[base_name]
                bindings[registry_key] = (binding.module, target.attr)
    return bindings


def extract_tensor_registry_key(func: ast.FunctionDef) -> Optional[str]:
    for stmt in func.body:
        if not isinstance(stmt, ast.Return) or not isinstance(stmt.value, ast.Call):
            continue
        callee = stmt.value.func
        if not isinstance(callee, ast.Call):
            continue
        if not isinstance(callee.func, ast.Attribute):
            continue
        if callee.func.attr != "get":
            continue
        owner = callee.func.value
        if not isinstance(owner, ast.Name) or owner.id != "tensor_operator_registry":
            continue
        if not callee.args:
            continue
        key_arg = callee.args[0]
        if isinstance(key_arg, ast.Constant) and isinstance(key_arg.value, str):
            return key_arg.value
    return None


def classify_tensor_method(
    method_name: str,
    registry_key: str,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    registry_bindings: Dict[str, Tuple[str, str]],
    ms_root: Path,
    op_by_symbol: Dict[str, List],
) -> str:
    active = active_branches(api_def_catalog.get(registry_key, []))
    if active:
        return classify_branch_count(len({branch.op_branch for branch in active}))

    binding = registry_bindings.get(registry_key)
    if binding is None:
        return "unresolved"

    target_module, target_symbol = binding
    resolved = resolve_symbol_to_entries(
        ms_root=ms_root,
        target_module=target_module,
        target_symbol=target_symbol,
        api_def_catalog=api_def_catalog,
        op_by_symbol=op_by_symbol,
    )
    resolved_type = classify_resolved_rows(resolved)
    if resolved_type != "unresolved":
        return resolved_type

    info = load_module_info(ms_root, target_module)
    if info is not None and target_symbol in info.functions:
        return "composite"
    if target_module == "mindspore.ops.functional":
        return "composite"
    return "unresolved"


def build_tensor_class_entries(
    ms_root: Path,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    op_by_symbol: Dict[str, List],
    existing_entry_ids: Set[str],
) -> List[dict]:
    path = ms_root / "python" / "mindspore" / "common" / "tensor.py"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    tensor_class = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Tensor"), None)
    if tensor_class is None:
        return []

    registry_bindings = parse_tensor_registry_bindings(ms_root)
    source_path = rel_source(path, ms_root)
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
        rows.append(
            {
                "entry_id": entry_id,
                "public_api": entry_id,
                "public_surface": "mindspore.Tensor",
                "public_name": method_name,
                "entry_type": classify_tensor_method(
                    method_name=method_name,
                    registry_key=registry_key,
                    api_def_catalog=api_def_catalog,
                    registry_bindings=registry_bindings,
                    ms_root=ms_root,
                    op_by_symbol=op_by_symbol,
                ),
                "source_type": "tensor_class",
                "source_path": source_path,
            }
        )
    return rows


def build_rows(ms_root: Path) -> List[dict]:
    api_def_catalog = build_api_def_catalog(ms_root)
    _, op_by_symbol = build_op_catalog(ms_root)
    rows_by_id: Dict[str, dict] = {}

    for row in build_api_def_entries(ms_root, api_def_catalog):
        add_entry(rows_by_id, row)

    for row in build_ops_function_entries(ms_root, api_def_catalog, op_by_symbol):
        add_entry(rows_by_id, row)

    for row in build_mint_entries(ms_root, api_def_catalog, op_by_symbol):
        add_entry(rows_by_id, row)

    existing_entry_ids = set(rows_by_id)
    for row in build_tensor_class_entries(ms_root, api_def_catalog, op_by_symbol, existing_entry_ids):
        add_entry(rows_by_id, row)

    return sorted(rows_by_id.values(), key=lambda item: item["entry_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ms_entry_identity index for operator-facts.")
    parser.add_argument("--ms-root", type=Path, default=DEFAULT_MS_ROOT)
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_DIR / "ms_entry_identity.jsonl")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_DIR / "ms_entry_identity.csv")
    args = parser.parse_args()

    rows = build_rows(args.ms_root)
    write_jsonl(args.out_jsonl, rows)
    fieldnames = [
        "entry_id",
        "public_api",
        "public_surface",
        "public_name",
        "entry_type",
        "source_type",
        "source_path",
    ]
    write_csv(args.out_csv, rows, fieldnames)
    print(f"ms_entry_identity_rows={len(rows)}")
    print(f"out_jsonl={args.out_jsonl}")
    print(f"out_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
