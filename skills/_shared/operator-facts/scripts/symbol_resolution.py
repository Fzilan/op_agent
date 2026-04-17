from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from common import ApiBranchEntry, OpDefEntry, extract_call_target, iter_return_calls, load_module_info, normalize_token


def resolve_symbol_to_entries(
    ms_root: Path,
    target_module: str,
    target_symbol: str,
    api_def_catalog: Dict[str, List[ApiBranchEntry]],
    op_by_symbol: Dict[str, List[OpDefEntry]],
    visited: Optional[Set[Tuple[str, str]]] = None,
    depth: int = 0,
) -> List[dict]:
    if visited is None:
        visited = set()
    state = (target_module, target_symbol)
    if state in visited or depth > 4:
        return []
    visited = set(visited)
    visited.add(state)

    if target_symbol in api_def_catalog:
        rows = []
        for branch in api_def_catalog[target_symbol]:
            if branch.deprecated:
                continue
            rows.append(
                {
                    "op_branch": branch.op_branch,
                    "py_method": branch.py_method,
                    "interface": branch.interface,
                    "resolver_kind": "api_def",
                    "resolver_path": f"{target_module}.{target_symbol}",
                }
            )
        if rows:
            return rows

    candidates = [
        target_symbol,
        target_symbol.removesuffix("_"),
        target_symbol.removesuffix("_impl"),
    ]
    seen = set()
    direct_rows = []
    for candidate in candidates:
        key = normalize_token(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        for entry in op_by_symbol.get(key, []):
            direct_rows.append(
                {
                    "op_branch": entry.op_branch,
                    "py_method": "",
                    "interface": "",
                    "resolver_kind": "op_def",
                    "resolver_path": f"{target_module}.{target_symbol}",
                }
            )
    if direct_rows:
        dedup = {}
        for row in direct_rows:
            dedup[row["op_branch"]] = row
        return list(dedup.values())

    info = load_module_info(ms_root, target_module)
    if info is None:
        return []
    func = info.functions.get(target_symbol)
    if func is None:
        return []

    rows = []
    for call in iter_return_calls(func):
        target = extract_call_target(call)
        if target is None:
            continue
        base, attr = target
        if attr is None:
            if base in info.from_imports:
                binding = info.from_imports[base]
                rows.extend(
                    resolve_symbol_to_entries(
                        ms_root,
                        binding.module,
                        binding.symbol or base,
                        api_def_catalog,
                        op_by_symbol,
                        visited=visited,
                        depth=depth + 1,
                    )
                )
            else:
                rows.extend(
                    resolve_symbol_to_entries(
                        ms_root,
                        target_module,
                        base,
                        api_def_catalog,
                        op_by_symbol,
                        visited=visited,
                        depth=depth + 1,
                    )
                )
        else:
            if base in info.imports:
                module_name = info.imports[base]
                rows.extend(
                    resolve_symbol_to_entries(
                        ms_root,
                        module_name,
                        attr,
                        api_def_catalog,
                        op_by_symbol,
                        visited=visited,
                        depth=depth + 1,
                    )
                )
            elif base in info.from_imports:
                binding = info.from_imports[base]
                rows.extend(
                    resolve_symbol_to_entries(
                        ms_root,
                        binding.module,
                        attr,
                        api_def_catalog,
                        op_by_symbol,
                        visited=visited,
                        depth=depth + 1,
                    )
                )
            else:
                rows.extend(
                    resolve_symbol_to_entries(
                        ms_root,
                        target_module,
                        attr,
                        api_def_catalog,
                        op_by_symbol,
                        visited=visited,
                        depth=depth + 1,
                    )
                )

    dedup = {}
    for row in rows:
        dedup[row["op_branch"]] = row
    return list(dedup.values())
