from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import yaml


OPERATOR_FACTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MS_ROOT = REPO_ROOT / "mindspore" / "mindspore"
DEFAULT_OUT_DIR = OPERATOR_FACTS_ROOT / "data"
MINT_INIT = DEFAULT_MS_ROOT / "python" / "mindspore" / "mint" / "__init__.py"


def snake_to_pascal(name: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", name) if part]
    return "".join(part[:1].upper() + part[1:] for part in parts)


def normalize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value).lower()


ALIAS_PREFIXES = (
    "test_ops_",
    "test_op_",
    "test_func_",
    "test_",
    "ops_",
)

ALIAS_SUFFIXES = (
    "_aclnn_kernel",
    "_kernel_register_auto",
    "_kernel_mod",
    "_kernel",
    "_doc",
    "_op",
)


def build_alias_keys(value: str) -> Set[str]:
    raw = Path(value).name
    stem = Path(raw).stem
    queue = [raw, stem]
    aliases: Set[str] = set()
    visited: Set[str] = set()
    while queue:
        current = queue.pop()
        current = current.strip("_")
        if not current or current in visited:
            continue
        visited.add(current)
        normalized = normalize_token(current)
        if normalized:
            aliases.add(normalized)
        for prefix in ALIAS_PREFIXES:
            if current.startswith(prefix):
                queue.append(current[len(prefix) :])
        for suffix in ALIAS_SUFFIXES:
            if current.endswith(suffix):
                queue.append(current[: -len(suffix)])
    return aliases


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def json_array_string(values: Iterable[str]) -> str:
    unique = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return json.dumps(unique, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_yaml_obj(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


@dataclass(frozen=True)
class OpDefEntry:
    op: str
    primitive: str
    class_name: str
    op_branch: str
    op_yaml_path: str
    dispatch_enable: bool
    dispatch_ascend: str


def build_op_catalog(ms_root: Path) -> Tuple[Dict[str, OpDefEntry], Dict[str, List[OpDefEntry]]]:
    op_def_root = ms_root / "ops" / "op_def" / "yaml"
    by_branch: Dict[str, OpDefEntry] = {}
    by_symbol: Dict[str, List[OpDefEntry]] = {}
    for path in sorted(op_def_root.glob("*_op.yaml")):
        obj = load_yaml_obj(path)
        for op_name, cfg in obj.items():
            if not isinstance(op_name, str) or not isinstance(cfg, dict):
                continue
            class_cfg = cfg.get("class", {})
            class_name = class_cfg.get("name") if isinstance(class_cfg, dict) else None
            class_name = class_name if isinstance(class_name, str) and class_name else snake_to_pascal(op_name)
            function_cfg = cfg.get("function", {})
            function_name = function_cfg.get("name") if isinstance(function_cfg, dict) else None
            function_name = function_name if isinstance(function_name, str) and function_name else ""
            dispatch = cfg.get("dispatch", {})
            dispatch_enable = isinstance(dispatch, dict) and to_bool(dispatch.get("enable", False))
            dispatch_ascend = ""
            if isinstance(dispatch, dict):
                dispatch_ascend = str(dispatch.get("Ascend", "default")) if dispatch_enable else ""
            entry = OpDefEntry(
                op=op_name,
                primitive=class_name,
                class_name=class_name,
                op_branch=path.name,
                op_yaml_path=str(path.relative_to(ms_root.parent)),
                dispatch_enable=dispatch_enable,
                dispatch_ascend=dispatch_ascend,
            )
            by_branch[entry.op_branch] = entry
            candidates = {
                op_name,
                class_name,
                path.stem,
                path.stem.removesuffix("_op"),
            }
            if function_name:
                candidates.add(function_name)
            for candidate in candidates:
                key = normalize_token(candidate)
                by_symbol.setdefault(key, []).append(entry)
    return by_branch, by_symbol


@dataclass(frozen=True)
class ApiBranchEntry:
    api_name: str
    op_branch: str
    py_method: str
    interface: str
    deprecated: bool
    branch_order: int
    source_file: str


def _normalize_api_def_branches(value) -> List[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def build_api_def_catalog(ms_root: Path) -> Dict[str, List[ApiBranchEntry]]:
    api_def_root = ms_root / "ops" / "api_def"
    catalog: Dict[str, List[ApiBranchEntry]] = {}
    for path in sorted(api_def_root.glob("*.yaml")):
        obj = load_yaml_obj(path)
        if len(obj) != 1:
            continue
        api_name, branch_value = next(iter(obj.items()))
        if not isinstance(api_name, str):
            continue
        branches = []
        for index, branch in enumerate(_normalize_api_def_branches(branch_value)):
            op_branch = branch.get("op_yaml", "")
            if not isinstance(op_branch, str) or not op_branch:
                continue
            entry = ApiBranchEntry(
                api_name=api_name,
                op_branch=Path(op_branch).name,
                py_method=str(branch.get("py_method", "")),
                interface=str(branch.get("interface", "")),
                deprecated=op_branch.startswith("deprecated/"),
                branch_order=index,
                source_file=str(path.relative_to(ms_root.parent)),
            )
            branches.append(entry)
        if branches:
            catalog[api_name] = branches
    return catalog


def module_name_to_path(ms_root: Path, module_name: str) -> Optional[Path]:
    if not module_name.startswith("mindspore."):
        return None
    rel = module_name.replace(".", "/") + ".py"
    path = ms_root / "python" / rel
    if path.exists():
        return path
    package_init = ms_root / "python" / module_name.replace(".", "/") / "__init__.py"
    if package_init.exists():
        return package_init
    return None


def resolve_importfrom_module(module_name: str, path: Path, target_module: str, level: int) -> str:
    if level <= 0:
        return target_module
    if path.name == "__init__.py":
        current_package = module_name
    else:
        current_package = module_name.rsplit(".", 1)[0]
    base_parts = current_package.split(".")
    if level > 1:
        base_parts = base_parts[: -(level - 1)]
    if target_module:
        return ".".join(base_parts + [target_module])
    return ".".join(base_parts)


@dataclass
class ImportBinding:
    module: str
    symbol: Optional[str]


@dataclass
class ModuleInfo:
    module_name: str
    path: Path
    from_imports: Dict[str, ImportBinding]
    imports: Dict[str, str]
    functions: Dict[str, ast.FunctionDef]


_MODULE_CACHE: Dict[Tuple[Path, str], Optional[ModuleInfo]] = {}


def load_module_info(ms_root: Path, module_name: str) -> Optional[ModuleInfo]:
    cache_key = (ms_root, module_name)
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]
    path = module_name_to_path(ms_root, module_name)
    if path is None:
        _MODULE_CACHE[cache_key] = None
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        _MODULE_CACHE[cache_key] = None
        return None
    from_imports: Dict[str, ImportBinding] = {}
    imports: Dict[str, str] = {}
    functions: Dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            resolved_module = resolve_importfrom_module(module_name, path, node.module or "", node.level)
            if not resolved_module:
                continue
            for alias in node.names:
                local_name = alias.asname or alias.name
                from_imports[local_name] = ImportBinding(resolved_module, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[-1]
                imports[local_name] = alias.name
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = node
    info = ModuleInfo(
        module_name=module_name,
        path=path,
        from_imports=from_imports,
        imports=imports,
        functions=functions,
    )
    _MODULE_CACHE[cache_key] = info
    return info


def iter_return_calls(func: ast.FunctionDef) -> List[ast.Call]:
    calls: List[ast.Call] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
            calls.append(node.value)
    return calls


def extract_call_target(call: ast.Call) -> Optional[Tuple[str, Optional[str]]]:
    if isinstance(call.func, ast.Name):
        return call.func.id, None
    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name):
        return call.func.value.id, call.func.attr
    return None
