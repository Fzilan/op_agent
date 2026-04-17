from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from common import DEFAULT_MS_ROOT, DEFAULT_OUT_DIR, build_api_def_catalog, build_op_catalog, load_module_info, write_csv, write_jsonl
from symbol_resolution import resolve_symbol_to_entries


@dataclass(frozen=True)
class GraphEdgeDraft:
    parent_unit_id: str
    child_ref_type: str
    child_ref: str
    condition: str
    via_symbol: str
    via_path: str


def branch_unit_id(op_branch: str, primitive: str) -> str:
    return f"branch::{op_branch}::{primitive}"


def composite_unit_id(ms_root: Path, impl_path: Path, impl_symbol: str) -> str:
    short_path = str(impl_path.relative_to(ms_root / "python" / "mindspore"))
    return f"composite::{short_path}::{impl_symbol}"


def graph_edge_id(parent_unit_id: str, child_ref_type: str, via_symbol: str, call_order: int) -> str:
    parent_name = parent_unit_id.rsplit("::", 1)[-1]
    return f"{parent_name}::{child_ref_type}::{via_symbol}::{call_order}"


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


def parse_composite_unit_id(ms_root: Path, unit_id: str) -> Optional[Tuple[Path, str]]:
    if not unit_id.startswith("composite::"):
        return None
    parts = unit_id.split("::", 2)
    if len(parts) != 3:
        return None
    rel_path, symbol = parts[1], parts[2]
    return ms_root / "python" / "mindspore" / rel_path, symbol


def module_name_from_impl_path(ms_root: Path, impl_path: Path) -> str:
    rel = impl_path.relative_to(ms_root / "python")
    return ".".join(rel.with_suffix("").parts)


def load_composite_unit_ids(path: Path) -> List[str]:
    if not path.exists():
        return []
    unit_ids: List[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("edge_type") == "composite" and row.get("unit_id"):
                unit_ids.append(row["unit_id"])
    return sorted(set(unit_ids))


def condition_text(expr: ast.expr) -> str:
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and len(expr.comparators) == 1:
        left = expr.left
        right = expr.comparators[0]
        if isinstance(left, ast.Name) and isinstance(right, ast.Constant):
            if isinstance(expr.ops[0], ast.Is):
                return f"{left.id} is {right.value!r}" if right.value is not None else f"{left.id} is None"
            if isinstance(expr.ops[0], ast.IsNot):
                return f"{left.id} is not {right.value!r}" if right.value is not None else f"{left.id} is not None"
    if hasattr(ast, "unparse"):
        return ast.unparse(expr)
    return "<condition>"


def negate_condition_text(expr: ast.expr) -> str:
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1 and len(expr.comparators) == 1:
        left = expr.left
        right = expr.comparators[0]
        if isinstance(left, ast.Name) and isinstance(right, ast.Constant):
            if isinstance(expr.ops[0], ast.Is):
                return f"{left.id} is not {right.value!r}" if right.value is not None else f"{left.id} is not None"
            if isinstance(expr.ops[0], ast.IsNot):
                return f"{left.id} is {right.value!r}" if right.value is not None else f"{left.id} is None"
    text = condition_text(expr)
    return f"not ({text})"


def combine_condition(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix} and {suffix}"


def parse_cache_prim_symbol(call: ast.Call) -> Optional[str]:
    if not isinstance(call.func, ast.Call):
        return None
    inner = call.func
    if not isinstance(inner.func, ast.Name) or inner.func.id != "_get_cache_prim":
        return None
    if not inner.args:
        return None
    arg = inner.args[0]
    if isinstance(arg, ast.Name):
        return arg.id
    return None


def make_binding_from_primitive(symbol: str, op_by_symbol: Dict[str, List]) -> Tuple[str, str]:
    key = "".join(ch for ch in symbol.lower() if ch.isalnum())
    dedup = {(entry.op_branch, entry.primitive): entry for entry in op_by_symbol.get(key, [])}
    candidates = list(dedup.values())
    if len(candidates) == 1:
        entry = candidates[0]
        return "unit", branch_unit_id(entry.op_branch, entry.primitive)
    return "primitive_symbol", symbol


def statements_always_return(statements: List[ast.stmt]) -> bool:
    if not statements:
        return False
    last = statements[-1]
    if isinstance(last, ast.Return):
        return True
    if isinstance(last, ast.If):
        if not last.body or not last.orelse:
            return False
        return statements_always_return(last.body) and statements_always_return(last.orelse)
    return False


def resolve_call_target(
    call: ast.Call,
    bindings: Dict[str, Tuple[str, str]],
    module_name: str,
    module_info,
    module_path: Path,
    ms_root: Path,
    api_def_catalog,
    op_by_symbol,
    op_by_branch,
    function_impl_index: Dict[str, List[Path]],
) -> Optional[Tuple[str, str, str]]:
    func = call.func
    if isinstance(func, ast.Name):
        if func.id in bindings:
            child_ref_type, child_ref = bindings[func.id]
            return child_ref_type, child_ref, func.id
        if func.id in module_info.from_imports:
            binding = module_info.from_imports[func.id]
            resolved = resolve_symbol_to_entries(
                ms_root=ms_root,
                target_module=binding.module,
                target_symbol=binding.symbol or func.id,
                api_def_catalog=api_def_catalog,
                op_by_symbol=op_by_symbol,
            )
            branches = {row["op_branch"] for row in resolved if row.get("op_branch")}
            if len(branches) == 1:
                op_branch = next(iter(branches))
                unit = op_by_branch.get(op_branch)
                if unit is not None:
                    return "unit", branch_unit_id(unit.op_branch, unit.primitive), binding.symbol or func.id
            if len(branches) > 1 and binding.module.startswith("mindspore.ops"):
                return "public_api", f"mindspore.ops.{binding.symbol or func.id}", binding.symbol or func.id
        if func.id in module_info.functions:
            return "unit", composite_unit_id(ms_root, module_path, func.id), func.id
        impl_paths = function_impl_index.get(func.id, [])
        if len(impl_paths) == 1:
            return "unit", composite_unit_id(ms_root, impl_paths[0], func.id), func.id
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        base = func.value.id
        symbol = func.attr
        if base == "ops":
            resolved = resolve_symbol_to_entries(
                ms_root=ms_root,
                target_module="mindspore.ops",
                target_symbol=symbol,
                api_def_catalog=api_def_catalog,
                op_by_symbol=op_by_symbol,
            )
            branches = {row["op_branch"] for row in resolved if row.get("op_branch")}
            if len(branches) == 1:
                op_branch = next(iter(branches))
                unit = op_by_branch.get(op_branch)
                if unit is not None:
                    return "unit", branch_unit_id(unit.op_branch, unit.primitive), symbol
            if len(branches) > 1:
                return "public_api", f"mindspore.ops.{symbol}", symbol
        if base in module_info.imports:
            target_module = module_info.imports[base]
            resolved = resolve_symbol_to_entries(
                ms_root=ms_root,
                target_module=target_module,
                target_symbol=symbol,
                api_def_catalog=api_def_catalog,
                op_by_symbol=op_by_symbol,
            )
            branches = {row["op_branch"] for row in resolved if row.get("op_branch")}
            if len(branches) == 1:
                op_branch = next(iter(branches))
                unit = op_by_branch.get(op_branch)
                if unit is not None:
                    return "unit", branch_unit_id(unit.op_branch, unit.primitive), symbol
            if len(branches) > 1 and target_module.startswith("mindspore.ops"):
                return "public_api", f"mindspore.ops.{symbol}", symbol
    return None


def collect_graph_edges_for_function(
    parent_unit_id: str,
    func: ast.FunctionDef,
    module_name: str,
    module_info,
    module_path: Path,
    ms_root: Path,
    api_def_catalog,
    op_by_symbol,
    op_by_branch,
    function_impl_index: Dict[str, List[Path]],
) -> List[dict]:
    drafts: List[GraphEdgeDraft] = []
    seen = set()

    def record(edge: Tuple[str, str, str], condition: str) -> None:
        child_ref_type, child_ref, via_symbol = edge
        key = (condition, child_ref_type, child_ref, via_symbol)
        if key in seen:
            return
        seen.add(key)
        drafts.append(
            GraphEdgeDraft(
                parent_unit_id=parent_unit_id,
                child_ref_type=child_ref_type,
                child_ref=child_ref,
                condition=condition,
                via_symbol=via_symbol,
                via_path=str(module_path.relative_to(ms_root.parent)),
            )
        )

    def walk(statements: List[ast.stmt], condition: str, bindings: Dict[str, Tuple[str, str]]) -> None:
        local_bindings = dict(bindings)
        current_condition = condition
        for stmt in statements:
            if isinstance(stmt, ast.If):
                cond_text = condition_text(stmt.test)
                walk(stmt.body, combine_condition(current_condition, cond_text), dict(local_bindings))
                walk(stmt.orelse, combine_condition(current_condition, negate_condition_text(stmt.test)), dict(local_bindings))
                if stmt.orelse:
                    continue
                if statements_always_return(stmt.body):
                    current_condition = combine_condition(current_condition, negate_condition_text(stmt.test))
                continue
            if isinstance(stmt, ast.Assign):
                if isinstance(stmt.value, ast.Call):
                    prim_symbol = parse_cache_prim_symbol(stmt.value)
                    if prim_symbol is not None:
                        binding = make_binding_from_primitive(prim_symbol, op_by_symbol)
                        for target in stmt.targets:
                            if isinstance(target, ast.Name):
                                local_bindings[target.id] = binding
                        continue
                    edge = resolve_call_target(
                        stmt.value,
                        local_bindings,
                        module_name,
                        module_info,
                        module_path,
                        ms_root,
                        api_def_catalog,
                        op_by_symbol,
                        op_by_branch,
                        function_impl_index,
                    )
                    if edge is not None:
                        record(edge, current_condition)
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                edge = resolve_call_target(
                    stmt.value,
                    local_bindings,
                    module_name,
                    module_info,
                    module_path,
                    ms_root,
                    api_def_catalog,
                    op_by_symbol,
                    op_by_branch,
                    function_impl_index,
                )
                if edge is not None:
                    record(edge, current_condition)

    walk(func.body, "", {})
    rows: List[dict] = []
    for call_order, draft in enumerate(drafts):
        rows.append(
            {
                "graph_edge_id": graph_edge_id(draft.parent_unit_id, draft.child_ref_type, draft.via_symbol, call_order),
                "parent_unit_id": draft.parent_unit_id,
                "child_ref_type": draft.child_ref_type,
                "child_ref": draft.child_ref,
                "call_order": call_order,
                "condition": draft.condition,
                "via_symbol": draft.via_symbol,
                "via_path": draft.via_path,
            }
        )
    return rows


def build_rows(ms_root: Path, entry_edges_path: Path) -> List[dict]:
    api_def_catalog = build_api_def_catalog(ms_root)
    op_by_branch, op_by_symbol = build_op_catalog(ms_root)
    function_impl_index = build_function_impl_index(ms_root)
    rows: List[dict] = []

    for unit_id in load_composite_unit_ids(entry_edges_path):
        parsed = parse_composite_unit_id(ms_root, unit_id)
        if parsed is None:
            continue
        impl_path, impl_symbol = parsed
        module_name = module_name_from_impl_path(ms_root, impl_path)
        module_info = load_module_info(ms_root, module_name)
        if module_info is None:
            continue
        func = module_info.functions.get(impl_symbol)
        if func is None:
            continue
        rows.extend(
            collect_graph_edges_for_function(
                parent_unit_id=unit_id,
                func=func,
                module_name=module_name,
                module_info=module_info,
                module_path=impl_path,
                ms_root=ms_root,
                api_def_catalog=api_def_catalog,
                op_by_symbol=op_by_symbol,
                op_by_branch=op_by_branch,
                function_impl_index=function_impl_index,
            )
        )
    return sorted(rows, key=lambda item: (item["parent_unit_id"], item["call_order"], item["graph_edge_id"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ms_unit_graph_edges index for operator-facts.")
    parser.add_argument("--ms-root", type=Path, default=DEFAULT_MS_ROOT)
    parser.add_argument(
        "--entry-unit-edges",
        type=Path,
        default=DEFAULT_OUT_DIR / "ms_entry_unit_edges.jsonl",
    )
    parser.add_argument("--out-jsonl", type=Path, default=DEFAULT_OUT_DIR / "ms_unit_graph_edges.jsonl")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_DIR / "ms_unit_graph_edges.csv")
    args = parser.parse_args()

    rows = build_rows(args.ms_root, args.entry_unit_edges)
    write_jsonl(args.out_jsonl, rows)
    fieldnames = [
        "graph_edge_id",
        "parent_unit_id",
        "child_ref_type",
        "child_ref",
        "call_order",
        "condition",
        "via_symbol",
        "via_path",
    ]
    write_csv(args.out_csv, rows, fieldnames)
    print(f"ms_unit_graph_edges_rows={len(rows)}")
    print(f"out_jsonl={args.out_jsonl}")
    print(f"out_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
