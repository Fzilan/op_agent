#!/usr/bin/env python3
"""Reconstruct top-op -> ACLNN chains using clangd call hierarchy + definition hops."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
from lsp_client import LspClient, Position

ACLNN_RE = re.compile(r"\baclnn[A-Z][A-Za-z0-9_]*\b")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
FUNC_DEF_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
FUNC_YAML_RE = re.compile(r"^\s*-\s*func:\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*\(")
FUNC_YAML_LINE_RE = re.compile(r"^\s*-\s*func:\s*(.+?)\s*$")
EXEC_YAML_RE = re.compile(r"^\s*exec:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")
INHERIT_YAML_RE = re.compile(r"^\s*structured_inherit:\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*$")
DERIV_NAME_RE = re.compile(r"^\s*-\s*name:\s*(.+?)\s*$")
COND_PREFIX = ("if (", "if(", "else if", "switch (", "switch(", "case ")
KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "static_cast",
    "reinterpret_cast",
    "const_cast",
    "dynamic_cast",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reconstruct top-op -> ACLNN chains (clangd required)")
    p.add_argument("--workspace", type=Path, default=Path(".."), help="workspace root containing op-plugin")
    p.add_argument("--op-plugin-root", type=Path, default=None, help="path to op-plugin repository root")
    p.add_argument(
        "--op-plugin-functions-yaml",
        type=Path,
        default=None,
        help="path to op_plugin_functions.yaml (for config fallback)",
    )
    p.add_argument(
        "--derivatives-yaml",
        type=Path,
        default=None,
        help="path to derivatives.yaml (for backward bindings)",
    )
    p.add_argument("--top-ops", type=str, default="", help="comma-separated top-op names")
    p.add_argument("--top-ops-file", type=Path, default=None, help="file with top-op names or func ids")
    p.add_argument(
        "--aclnn-set",
        type=Path,
        required=True,
        help="txt/json ACLNN full set path",
    )
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--max-nodes-per-op", type=int, default=300)
    p.add_argument("--max-outgoing-per-node", type=int, default=30)
    p.add_argument("--max-def-hops-per-node", type=int, default=20)
    p.add_argument(
        "--enable-aclnn-gap-scan",
        action="store_true",
        help="scan related C++ files for ACLNN APIs and emit gap candidates",
    )
    p.add_argument("--out-dir", type=Path, default=Path("./output"))
    return p.parse_args()


def load_aclnn_set(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        vals = obj.get("aclnn_apis") or obj.get("apis") or []
        out: set[str] = set()
        for x in vals:
            if isinstance(x, str):
                out.add(x)
            elif isinstance(x, dict) and "aclnn_api" in x:
                out.add(str(x["aclnn_api"]))
        return out
    return {ln.strip() for ln in text.splitlines() if ln.strip()}


def normalize_top_name(x: str) -> str:
    s = x.strip()
    if not s:
        return ""
    if s.startswith("- func:"):
        s = s.split(":", 1)[1].strip()
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    if "." in s:
        s = s.split(".", 1)[0].strip()
    return s


def normalize_root_name(name: str) -> str:
    n = (name or "").strip()
    if not n:
        return n
    n = re.sub(r"\.(Tensor|Scalar|out|Tensor_mode|Scalar_mode|out_mode)$", "", n)
    if "." in n:
        n = n.split(".", 1)[0]
    if n.endswith("_"):
        n = n[:-1]
    return n


def load_top_ops(inline_ops: str, file_path: Path | None) -> list[str]:
    vals: list[str] = []
    if inline_ops.strip():
        vals.extend([x.strip() for x in inline_ops.split(",") if x.strip()])
    if file_path and file_path.exists():
        vals.extend([ln.strip() for ln in file_path.read_text(encoding="utf-8").splitlines() if ln.strip()])
    out: list[str] = []
    seen: set[str] = set()
    for x in vals:
        n = normalize_top_name(x)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def parse_op_plugin_functions_yaml(path: Path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    if not path.exists():
        return {}, path.resolve().as_uri()
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    by_name: dict[str, list[dict[str, Any]]] = {}
    cur: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        nm = str(cur.get("func_name", ""))
        if nm:
            by_name.setdefault(nm, []).append(cur)
        cur = None

    for i, ln in enumerate(lines):
        m_func_line = FUNC_YAML_LINE_RE.match(ln)
        m_func = FUNC_YAML_RE.match(ln)
        if m_func:
            flush()
            full_decl = (m_func_line.group(1).strip() if m_func_line else f"{m_func.group(1)}(...)")
            cur = {
                "func_name": m_func.group(1),
                "func_root": normalize_root_name(m_func.group(1)),
                "func_decl": full_decl,
                "func_line": i + 1,
                "exec": None,
                "exec_line": None,
                "structured_inherit": None,
                "inherit_line": None,
            }
            continue
        if not cur:
            continue
        m_exec = EXEC_YAML_RE.match(ln)
        if m_exec:
            cur["exec"] = m_exec.group(1)
            cur["exec_line"] = i + 1
            continue
        m_inh = INHERIT_YAML_RE.match(ln)
        if m_inh:
            cur["structured_inherit"] = m_inh.group(1)
            cur["inherit_line"] = i + 1
            continue
    flush()
    return by_name, path.resolve().as_uri()


def build_front_catalog(entries_by_name: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    by_root: dict[str, list[dict[str, Any]]] = {}
    for _, arr in entries_by_name.items():
        for e in arr:
            root = str(e.get("func_root") or normalize_root_name(str(e.get("func_name", ""))))
            if not root:
                continue
            by_root.setdefault(root, []).append(e)
    for k in by_root:
        by_root[k].sort(key=lambda x: int(x.get("func_line", 0)))
    return by_root


def find_first_entry(entries_by_name: dict[str, list[dict[str, Any]]], name: str) -> dict[str, Any] | None:
    arr = entries_by_name.get(name) or []
    return arr[0] if arr else None


def resolve_exec_from_yaml(
    name: str,
    entries_by_name: dict[str, list[dict[str, Any]]],
    seen: set[str] | None = None,
) -> tuple[str | None, list[str], int | None]:
    if seen is None:
        seen = set()
    if name in seen:
        return None, [name], None
    seen.add(name)
    e = find_first_entry(entries_by_name, name)
    if not e:
        return None, [name], None
    ex = e.get("exec")
    if isinstance(ex, str) and ex:
        return ex, [name], int(e.get("exec_line") or e.get("func_line") or 0)
    inh = e.get("structured_inherit")
    if isinstance(inh, str) and inh:
        ex2, route, line = resolve_exec_from_yaml(inh, entries_by_name, seen)
        return ex2, [name] + route, line
    return None, [name], int(e.get("func_line") or 0)


def infer_paths_from_yaml(
    op: str,
    entries_by_name: dict[str, list[dict[str, Any]]],
    yaml_uri: str,
    aclnn_set: set[str],
) -> list[dict[str, Any]]:
    candidates = [op, f"{op}.out", f"{op}_"]
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for cand in candidates:
        ex, route, exec_line = resolve_exec_from_yaml(cand, entries_by_name)
        if not ex:
            continue
        if aclnn_set and ex not in aclnn_set:
            continue
        k = (cand, ex)
        if k in seen:
            continue
        seen.add(k)

        chain_nodes: list[dict[str, Any]] = []
        for nm in route:
            e = find_first_entry(entries_by_name, nm)
            ln = int((e or {}).get("func_line") or 1) - 1
            chain_nodes.append(make_item(nm, yaml_uri, max(0, ln), 0))

        out.append(
            {
                "aclnn_api": ex,
                "chain": chain_nodes,
                "path_conditions": [f"yaml_fallback: gen_opapi.exec via {' -> '.join(route)}"],
                "endpoint": {"uri": yaml_uri, "line": int(exec_line or 1), "column": 1},
                "path_source": "config_fallback",
            }
        )
    return out


def parse_derivatives_yaml(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], str]:
    if not path.exists():
        return {}, {}, path.resolve().as_uri()
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out_by_root: dict[str, list[dict[str, Any]]] = {}
    out_by_name: dict[str, list[dict[str, Any]]] = {}
    cur: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal cur
        if not cur:
            return
        root = str(cur.get("name_root", ""))
        nm = str(cur.get("name", ""))
        if root:
            out_by_root.setdefault(root, []).append(cur)
        if nm:
            out_by_name.setdefault(nm, []).append(cur)
        cur = None

    for i, ln in enumerate(lines):
        m_name = DERIV_NAME_RE.match(ln)
        if m_name:
            flush()
            sig = m_name.group(1).strip()
            op_name = sig.split("(", 1)[0].strip()
            cur = {
                "name_decl": sig,
                "name": op_name,
                "name_root": normalize_root_name(op_name),
                "line": i + 1,
                "differentiable_inputs": [],
                "formula_keys": [],
            }
            continue
        if not cur:
            continue
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("version:") or s.startswith("output_differentiability:"):
            continue
        if ":" not in s:
            continue
        k, v = s.split(":", 1)
        key = k.strip()
        val = v.strip()
        if key in {"name", "version", "output_differentiability", "result"}:
            continue
        cur["formula_keys"].append(key)
        if val and "non_differentiable" not in val:
            for token in key.split(","):
                t = token.strip()
                if t:
                    cur["differentiable_inputs"].append(t)
    flush()

    for root, arr in out_by_root.items():
        arr.sort(key=lambda x: int(x.get("line", 0)))
        for e in arr:
            # keep order but de-duplicate
            vals = e.get("differentiable_inputs", [])
            e["differentiable_inputs"] = list(dict.fromkeys(vals))
    return out_by_root, out_by_name, path.resolve().as_uri()


def match_backward_bindings(
    op: str,
    front_defs: list[dict[str, Any]],
    by_root: dict[str, list[dict[str, Any]]],
    by_name: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], str]:
    candidates = [op, f"{op}.out", f"{op}_"]
    for d in front_defs:
        nm = str(d.get("func_name", ""))
        if nm:
            candidates.append(nm)

    exact_hits: list[dict[str, Any]] = []
    for c in candidates:
        exact_hits.extend(by_name.get(c, []))
    if exact_hits:
        ded: dict[tuple[str, int], dict[str, Any]] = {}
        for h in exact_hits:
            ded[(str(h.get("name_decl", "")), int(h.get("line", 0)))] = h
        return list(ded.values()), "exact"

    norm_hits = by_root.get(normalize_root_name(op), [])
    if norm_hits:
        ded2: dict[tuple[str, int], dict[str, Any]] = {}
        for h in norm_hits:
            ded2[(str(h.get("name_decl", "")), int(h.get("line", 0)))] = h
        return list(ded2.values()), "normalized"
    return [], "none"


def collect_cpp_roots(op_plugin_root: Path) -> list[Path]:
    return [op_plugin_root / "op_plugin" / "ops" / "opapi", op_plugin_root / "op_plugin" / "ops" / "aclops"]


def read_lines(path: Path, cache: dict[Path, list[str]]) -> list[str]:
    if path not in cache:
        cache[path] = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return cache[path]


def find_conditions(lines: list[str], at_line: int, lookback: int = 18) -> list[str]:
    out: list[str] = []
    lo = max(0, at_line - lookback)
    for i in range(at_line, lo - 1, -1):
        s = lines[i].strip()
        if s.startswith(COND_PREFIX):
            out.append(s)
    out.reverse()
    return out


def node_key(item: dict[str, Any]) -> tuple[str, int, int, str]:
    sel = item.get("selectionRange") or item.get("range") or {}
    st = sel.get("start") or {}
    return (
        str(item.get("uri", "")),
        int(st.get("line", 0)),
        int(st.get("character", 0)),
        str(item.get("name", "")),
    )


def make_item(name: str, uri: str, line0: int, col0: int) -> dict[str, Any]:
    return {
        "name": name,
        "uri": uri,
        "selectionRange": {
            "start": {"line": line0, "character": col0},
            "end": {"line": line0, "character": col0 + max(1, len(name))},
        },
        "range": {
            "start": {"line": line0, "character": 0},
            "end": {"line": line0, "character": 200},
        },
    }


def find_candidates_with_rg(roots: list[Path], top_name: str) -> list[tuple[Path, int, int]]:
    out: list[tuple[Path, int, int]] = []
    pattern = rf"\b{re.escape(top_name)}\s*\("
    cmd = ["rg", "-n", "--no-heading", pattern, *[str(r) for r in roots if r.exists()]]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        return out
    for ln in proc.stdout.splitlines():
        parts = ln.split(":", 3)
        if len(parts) < 4:
            continue
        fp = Path(parts[0])
        line_no = int(parts[1])
        text = parts[3]
        col = text.find(f"{top_name}(")
        if col < 0:
            col = max(0, text.find(top_name))
        out.append((fp, line_no - 1, col))
    return out


def prepare_entry_items(
    client: LspClient,
    roots: list[Path],
    top_name: str,
    line_cache: dict[Path, list[str]],
    opened: set[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for f, i, idx in find_candidates_with_rg(roots, top_name):
        if not f.exists():
            continue
        lines = read_lines(f, line_cache)
        uri = f.resolve().as_uri()
        if uri not in opened:
            client.did_open(uri, "cpp", "\n".join(lines))
            opened.add(uri)
        try:
            res = client.prepare_call_hierarchy(uri, Position(line=i, character=max(0, idx)))
        except Exception:
            continue
        if not res:
            continue
        arr = res if isinstance(res, list) else [res]
        for it in arr:
            if it.get("name") == top_name:
                items.append(it)
    uniq: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for it in items:
        uniq[node_key(it)] = it
    return list(uniq.values())




def infer_signature_condition(item: dict[str, Any], lines: list[str]) -> list[str]:
    rng = item.get("range") or {}
    s = int(((rng.get("start") or {}).get("line", 0)))
    s = max(0, min(s, len(lines) - 1))
    name = str(item.get("name", ""))

    # Capture only the current function signature block.
    sig_parts: list[str] = []
    started = False
    for i in range(s, min(len(lines), s + 20)):
        ln = lines[i]
        if not started:
            if name and f"{name}(" in ln:
                started = True
                sig_parts.append(ln)
            continue
        sig_parts.append(ln)
        if ")" in ln or "{" in ln:
            break

    if not sig_parts:
        sig_parts = [lines[s]]

    sig_blob = "\n".join(sig_parts)
    conds: list[str] = []
    if "const at::Scalar &other" in sig_blob or "const at::Scalar& other" in sig_blob:
        conds.append("overload(other: Scalar)")
    if "const at::Tensor &other" in sig_blob or "const at::Tensor& other" in sig_blob:
        conds.append("overload(other: Tensor)")
    if "c10::optional<c10::string_view> rounding_mode" in sig_blob:
        conds.append("overload(rounding_mode: optional)")
    return conds

def extract_direct_aclnn_hits(item: dict[str, Any], lines: list[str], aclnn_set: set[str]) -> list[dict[str, Any]]:
    rng = item.get("range") or {}
    s = int(((rng.get("start") or {}).get("line", 0)))
    e = int(((rng.get("end") or {}).get("line", s + 120)))
    s = max(0, s)
    e = min(len(lines) - 1, max(s, e))
    out: list[dict[str, Any]] = []
    sig_conds = infer_signature_condition(item, lines)
    for i in range(s, e + 1):
        for m in ACLNN_RE.finditer(lines[i]):
            api = m.group(0)
            if aclnn_set and api not in aclnn_set:
                continue
            pc = sig_conds + find_conditions(lines, i)
            out.append({"aclnn_api": api, "line": i, "col": m.start(), "path_conditions": pc})
    return out


def parse_def_location(call_name: str, loc: dict[str, Any], lines: list[str]) -> dict[str, Any] | None:
    if "targetUri" in loc:
        uri = str(loc.get("targetUri"))
        st = ((loc.get("targetSelectionRange") or {}).get("start") or {})
    else:
        uri = str(loc.get("uri"))
        st = ((loc.get("range") or {}).get("start") or {})
    if not uri.startswith("file://"):
        return None
    line0 = int(st.get("line", 0))
    col0 = int(st.get("character", 0))
    nm = call_name
    if 0 <= line0 < len(lines):
        m = FUNC_DEF_RE.search(lines[line0])
        if m:
            nm = m.group(1)
    return make_item(nm, uri, line0, col0)


def find_infile_def_lines(lines: list[str], call_name: str) -> list[int]:
    out: list[int] = []
    pat = re.compile(rf"\b{re.escape(call_name)}\s*\(")
    for i, ln in enumerate(lines):
        st = ln.strip()
        if st.startswith(("if ", "if(", "for ", "for(", "while", "switch", "return")):
            continue
        if pat.search(ln):
            out.append(i)
    return out


def resolve_definition_calls(
    client: LspClient,
    item: dict[str, Any],
    lines: list[str],
    line_cache: dict[Path, list[str]],
    opened: set[str],
    max_hops: int,
) -> list[tuple[dict[str, Any], list[str]]]:
    rng = item.get("range") or {}
    s = int(((rng.get("start") or {}).get("line", 0)))
    e = int(((rng.get("end") or {}).get("line", s + 120)))
    s = max(0, s)
    e = min(len(lines) - 1, max(s, e))

    uri = str(item.get("uri", ""))
    out: list[tuple[dict[str, Any], list[str]]] = []
    seen_calls: set[tuple[str, int]] = set()

    for i in range(s, e + 1):
        line = lines[i]
        for m in CALL_RE.finditer(line):
            call_name = m.group(1)
            if call_name in KEYWORDS or call_name == item.get("name"):
                continue
            key = (call_name, i)
            if key in seen_calls:
                continue
            seen_calls.add(key)
            conds = find_conditions(lines, i)

            # First choice: LSP call hierarchy prepared at callsite
            try:
                prep = client.prepare_call_hierarchy(uri, Position(line=i, character=m.start(1)))
            except Exception:
                prep = None
            if prep:
                arrp = prep if isinstance(prep, list) else [prep]
                for it in arrp:
                    if not isinstance(it, dict):
                        continue
                    if str(it.get("name")) == call_name and str(it.get("name")) != str(item.get("name")):
                        out.append((it, conds))
                if len(out) >= max_hops:
                    return out

            # Fallback: definition hops
            got_any = False
            try:
                defs = client.definition(uri, Position(line=i, character=m.start(1)))
            except Exception:
                defs = None
            if defs:
                arr = defs if isinstance(defs, list) else [defs]
                for loc in arr[:2]:
                    tgt_uri = str(loc.get("targetUri") or loc.get("uri") or "")
                    if not tgt_uri.startswith("file://"):
                        continue
                    fp = Path(tgt_uri[7:])
                    if not fp.exists():
                        continue
                    tgt_lines = read_lines(fp, line_cache)
                    if tgt_uri not in opened:
                        client.did_open(tgt_uri, "cpp", "\n".join(tgt_lines))
                        opened.add(tgt_uri)
                    it = parse_def_location(call_name, loc, tgt_lines)
                    if it is not None:
                        out.append((it, conds))
                        got_any = True
                if len(out) >= max_hops:
                    return out

            # Final fallback: locate in current file and verify with LSP prepareCallHierarchy
            if not got_any:
                for di in find_infile_def_lines(lines, call_name)[:3]:
                    try:
                        prep2 = client.prepare_call_hierarchy(uri, Position(line=di, character=max(0, lines[di].find(call_name))))
                    except Exception:
                        prep2 = None
                    if not prep2:
                        continue
                    arr2 = prep2 if isinstance(prep2, list) else [prep2]
                    for it in arr2:
                        if isinstance(it, dict) and str(it.get("name")) == call_name:
                            out.append((it, conds))
                            got_any = True
                    if not got_any:
                        local_item = make_item(call_name, uri, di, max(0, lines[di].find(call_name)))
                        out.append((local_item, conds))
                        got_any = True
                    if got_any:
                        break
                if len(out) >= max_hops:
                    return out
    return out


def chain_signature(chain: list[dict[str, Any]]) -> str:
    return " -> ".join([str(x.get("name", "?")) for x in chain])


def is_cpp_uri(uri: str) -> bool:
    u = (uri or "").lower()
    return u.endswith((".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"))


def path_dispatch_kind(path: dict[str, Any]) -> str:
    chain = path.get("chain", []) or []
    conds = [str(x) for x in (path.get("path_conditions") or [])]
    non_overload = [c for c in conds if not c.startswith("overload(")]
    if len(chain) > 1:
        return "helper"
    if non_overload:
        return "logic_preprocessed"
    return "strict_direct"


def build_dispatch_summary(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_api: dict[str, dict[str, Any]] = {}
    for p in paths:
        api = str(p.get("aclnn_api", ""))
        if not api:
            continue
        rec = per_api.setdefault(
            api,
            {"aclnn_api": api, "has_strict_direct": False, "has_helper": False, "has_logic_preprocessed": False, "path_count": 0},
        )
        k = path_dispatch_kind(p)
        rec["path_count"] = int(rec["path_count"]) + 1
        if k == "strict_direct":
            rec["has_strict_direct"] = True
        elif k == "helper":
            rec["has_helper"] = True
        else:
            rec["has_logic_preprocessed"] = True

    out: list[dict[str, Any]] = []
    for api, rec in sorted(per_api.items()):
        has_direct = bool(rec["has_strict_direct"])
        has_helper = bool(rec["has_helper"])
        has_logic = bool(rec["has_logic_preprocessed"])
        if has_direct and (has_helper or has_logic):
            shape = "mixed"
        elif has_direct:
            shape = "direct"
        elif has_helper:
            shape = "helper"
        else:
            shape = "logic_preprocessed"
        out.append(
            {
                "aclnn_api": api,
                "dispatch_shape": shape,
                "has_strict_direct": has_direct,
                "path_count": rec["path_count"],
            }
        )
    return out


def scan_aclnn_mentions_in_cpp(paths: list[dict[str, Any]], entries: list[dict[str, Any]], line_cache: dict[Path, list[str]]) -> dict[str, list[dict[str, Any]]]:
    cpp_files: set[Path] = set()
    for p in paths:
        for n in p.get("chain", []) or []:
            uri = str((n or {}).get("uri", ""))
            if uri.startswith("file://") and is_cpp_uri(uri):
                cpp_files.add(Path(uri[7:]))
        ep = p.get("endpoint") or {}
        euri = str(ep.get("uri", ""))
        if euri.startswith("file://") and is_cpp_uri(euri):
            cpp_files.add(Path(euri[7:]))
    for en in entries:
        uri = str((en or {}).get("uri", ""))
        if uri.startswith("file://") and is_cpp_uri(uri):
            cpp_files.add(Path(uri[7:]))

    out: dict[str, list[dict[str, Any]]] = {}
    for fp in cpp_files:
        if not fp.exists():
            continue
        lines = read_lines(fp, line_cache)
        for i, ln in enumerate(lines):
            for m in ACLNN_RE.finditer(ln):
                api = m.group(0)
                out.setdefault(api, []).append({"uri": fp.resolve().as_uri(), "line": i + 1, "snippet": ln.strip()[:300]})
    return out


def build_aclnn_completeness(
    op: str,
    front_defs: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    line_cache: dict[Path, list[str]],
) -> dict[str, Any]:
    observed = sorted({str(p.get("aclnn_api", "")) for p in paths if str(p.get("aclnn_api", ""))})
    mentions = scan_aclnn_mentions_in_cpp(paths, entries, line_cache)
    scanned = sorted(mentions.keys())
    gaps = [x for x in scanned if x not in set(observed)]

    gap_candidates: list[dict[str, Any]] = []
    for g in gaps:
        gap_candidates.append({"aclnn_api": g, "evidence": mentions.get(g, [])[:5]})

    # LLM judgement is intentionally not executed in this script.
    # Skill workflow may backfill suspected_missing_apis later.
    suspected_missing: list[str] = []
    final_catalog = sorted(set(observed))
    return {
        "operator": op,
        "front_signatures": [str(d.get("func_decl", "")) for d in front_defs],
        "observed_apis": observed,
        "scanned_mentions": scanned,
        "gap_candidates": gap_candidates,
        "suspected_missing_apis": sorted(suspected_missing),
        "final_api_catalog": final_catalog,
        "judge_source": "pending_skill_step",
    }


def build_key_record(result: dict[str, Any]) -> dict[str, Any]:
    op = str(result.get("operator", ""))
    front = result.get("front_signatures", []) or []
    backward = result.get("backward_bindings", []) or []
    paths = result.get("paths", []) or []
    dispatch = result.get("dispatch_summary", []) or []
    completeness = result.get("aclnn_completeness", {}) or {}

    key_paths: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for p in paths:
        acl = str(p.get("aclnn_api", ""))
        src = str(p.get("path_source", "lsp"))
        conds = [str(x) for x in (p.get("path_conditions") or [])]
        # Keep short, unique branch signals for writer input.
        cond_short = " | ".join(conds[:3])
        sig = (acl, src, cond_short)
        if sig in seen:
            continue
        seen.add(sig)
        key_paths.append(
            {
                "aclnn_api": acl,
                "path_source": src,
                "conditions": conds,
                "dispatch_note": p.get("dispatch_note", "unknown"),
            }
        )

    key_backward = []
    for b in backward:
        key_backward.append(
            {
                "signature": b.get("signature"),
                "differentiable_inputs": b.get("differentiable_inputs", []),
                "evidence": f"{b.get('uri','')}:{b.get('line','')}",
            }
        )

    return {
        "operator": op,
        "status": result.get("status"),
        "front_signatures": [x.get("signature") for x in front],
        "forward_paths": key_paths,
        "dispatch_summary": dispatch,
        "aclnn_mapping_catalog": completeness.get("final_api_catalog", []),
        "aclnn_gap_suspects": completeness.get("suspected_missing_apis", []),
        "has_backward": bool(result.get("has_backward", False)),
        "backward_match": result.get("backward_match", "none"),
        "backward_bindings": key_backward,
    }


def traverse_paths(
    client: LspClient,
    root_item: dict[str, Any],
    aclnn_set: set[str],
    max_depth: int,
    max_nodes: int,
    max_outgoing: int,
    max_def_hops: int,
    line_cache: dict[Path, list[str]],
    opened: set[str],
) -> tuple[list[dict[str, Any]], int]:
    paths: list[dict[str, Any]] = []
    visited: set[tuple[str, int, int, str]] = set()
    q: list[tuple[dict[str, Any], list[dict[str, Any]], int, list[str]]] = [(root_item, [root_item], 0, [])]
    visited_nodes = 0

    while q and visited_nodes < max_nodes:
        item, chain, depth, conds = q.pop(0)
        k = node_key(item)
        if k in visited:
            continue
        visited.add(k)
        visited_nodes += 1

        uri = str(item.get("uri", ""))
        lines: list[str] = []
        if uri.startswith("file://"):
            fp = Path(uri[7:])
            if fp.exists():
                lines = read_lines(fp, line_cache)
                for hit in extract_direct_aclnn_hits(item, lines, aclnn_set):
                    paths.append(
                        {
                            "aclnn_api": hit["aclnn_api"],
                            "chain": chain,
                            "path_conditions": conds + hit["path_conditions"],
                            "endpoint": {"uri": uri, "line": hit["line"] + 1, "column": hit["col"] + 1},
                        }
                    )

        if depth >= max_depth:
            continue

        next_edges: list[tuple[dict[str, Any], list[str]]] = []

        # 1) call hierarchy outgoing
        try:
            out_calls = client.outgoing_calls(item) or []
        except Exception:
            out_calls = []
        for oc in out_calls[:max_outgoing]:
            to_item = oc.get("to")
            if not isinstance(to_item, dict):
                continue
            edge_conds: list[str] = []
            frs = oc.get("fromRanges") or []
            if lines and frs:
                st = int(((frs[0].get("start") or {}).get("line", -1)))
                if 0 <= st < len(lines):
                    edge_conds = find_conditions(lines, st)
            next_edges.append((to_item, edge_conds))

        # 2) definition hops from callsites inside function body (补中层)
        if lines:
            next_edges.extend(resolve_definition_calls(client, item, lines, line_cache, opened, max_def_hops))

        # enqueue
        uniq: dict[tuple[str, int, int, str], tuple[dict[str, Any], list[str]]] = {}
        for to_item, econds in next_edges:
            uniq[node_key(to_item)] = (to_item, econds)

        for _, (to_item, econds) in uniq.items():
            m = ACLNN_RE.search(str(to_item.get("name", "")))
            if m and ((not aclnn_set) or m.group(0) in aclnn_set):
                paths.append(
                    {
                        "aclnn_api": m.group(0),
                        "chain": chain + [to_item],
                        "path_conditions": conds + econds,
                        "endpoint": {
                            "uri": to_item.get("uri"),
                            "line": int(((to_item.get("selectionRange") or {}).get("start") or {}).get("line", 0)) + 1,
                            "column": int(((to_item.get("selectionRange") or {}).get("start") or {}).get("character", 0)) + 1,
                        },
                    }
                )
            else:
                q.append((to_item, chain + [to_item], depth + 1, conds + econds))

    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for p in paths:
        sig = chain_signature(p.get("chain", []))
        cond = " | ".join(p.get("path_conditions", []))
        dedup[(str(p.get("aclnn_api")), sig, cond)] = p
    out = list(dedup.values())
    out.sort(key=lambda x: (x.get("aclnn_api", ""), len(x.get("chain", []))))
    return out, visited_nodes


def main() -> int:
    args = parse_args()
    t0 = time.time()

    workspace = args.workspace.resolve()
    op_root = args.op_plugin_root.resolve() if args.op_plugin_root else (workspace / "op-plugin").resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aclnn_set = load_aclnn_set(args.aclnn_set.resolve())
    top_ops = load_top_ops(args.top_ops, args.top_ops_file.resolve() if args.top_ops_file else None)
    if not top_ops:
        raise SystemExit("no top ops provided")
    op_funcs_yaml = (
        args.op_plugin_functions_yaml.resolve()
        if args.op_plugin_functions_yaml
        else (op_root / "op_plugin" / "config" / "op_plugin_functions.yaml").resolve()
    )
    yaml_entries_by_name, yaml_uri = parse_op_plugin_functions_yaml(op_funcs_yaml)
    front_catalog = build_front_catalog(yaml_entries_by_name)
    derivatives_yaml = (
        args.derivatives_yaml.resolve()
        if args.derivatives_yaml
        else (op_root / "op_plugin" / "config" / "derivatives.yaml").resolve()
    )
    backward_by_root, backward_by_name, derivatives_uri = parse_derivatives_yaml(derivatives_yaml)

    roots = collect_cpp_roots(op_root)
    line_cache: dict[Path, list[str]] = {}
    opened: set[str] = set()

    client = LspClient(["clangd"], cwd=workspace)
    client.initialize(workspace.as_uri(), os.getpid())

    results: list[dict[str, Any]] = []
    try:
        for op in top_ops:
            entries = prepare_entry_items(client, roots, op, line_cache, opened)
            entries_count = len(entries)
            vis_sum = 0
            final_paths: list[dict[str, Any]] = []
            if entries:
                merged: list[dict[str, Any]] = []
                for en in entries:
                    ps, vis = traverse_paths(
                        client,
                        en,
                        aclnn_set,
                        max_depth=args.max_depth,
                        max_nodes=args.max_nodes_per_op,
                        max_outgoing=args.max_outgoing_per_node,
                        max_def_hops=args.max_def_hops_per_node,
                        line_cache=line_cache,
                        opened=opened,
                    )
                    merged.extend(ps)
                    vis_sum += vis

                ded: dict[tuple[str, str, str], dict[str, Any]] = {}
                for p in merged:
                    sig = chain_signature(p.get("chain", []))
                    cond = " | ".join(p.get("path_conditions", []))
                    ded[(str(p.get("aclnn_api")), sig, cond)] = p
                final_paths = list(ded.values())
                final_paths.sort(key=lambda x: (x.get("aclnn_api", ""), len(x.get("chain", []))))

            if not final_paths:
                final_paths = infer_paths_from_yaml(op, yaml_entries_by_name, yaml_uri, aclnn_set)

            front_defs = front_catalog.get(op, [])
            backward_defs, backward_match = match_backward_bindings(op, front_defs, backward_by_root, backward_by_name)
            rendered_paths = [
                {
                    "aclnn_api": p.get("aclnn_api"),
                    "chain": [
                        {
                            "name": x.get("name"),
                            "uri": x.get("uri"),
                            "line": int(((x.get("selectionRange") or {}).get("start") or {}).get("line", 0)) + 1,
                            "character": int(((x.get("selectionRange") or {}).get("start") or {}).get("character", 0)) + 1,
                        }
                        for x in p.get("chain", [])
                    ],
                    "path_conditions": p.get("path_conditions", []),
                    "endpoint": p.get("endpoint"),
                    "path_source": p.get("path_source", "lsp"),
                }
                for p in final_paths
            ]
            for rp in rendered_paths:
                rp["dispatch_note"] = path_dispatch_kind(rp)
            dispatch_summary = build_dispatch_summary(rendered_paths)
            aclnn_completeness = (
                build_aclnn_completeness(
                    op=op,
                    front_defs=front_defs,
                    paths=rendered_paths,
                    entries=entries,
                    line_cache=line_cache,
                )
                if args.enable_aclnn_gap_scan
                else {
                    "operator": op,
                    "front_signatures": [str(d.get("func_decl", "")) for d in front_defs],
                    "observed_apis": sorted({str(x.get("aclnn_api", "")) for x in rendered_paths if str(x.get("aclnn_api", ""))}),
                    "scanned_mentions": [],
                    "gap_candidates": [],
                    "suspected_missing_apis": [],
                    "final_api_catalog": sorted({str(x.get("aclnn_api", "")) for x in rendered_paths if str(x.get("aclnn_api", ""))}),
                    "judge_source": "disabled",
                }
            )

            results.append(
                {
                    "operator": op,
                    "status": "ok" if final_paths else ("no_path" if entries_count > 0 else "no_entry"),
                    "entries": entries_count,
                    "visited_nodes": vis_sum,
                    "front_signatures": [
                        {
                            "name": d.get("func_name"),
                            "signature": d.get("func_decl"),
                            "line": d.get("func_line"),
                            "uri": yaml_uri,
                        }
                        for d in front_defs
                    ],
                    "overload_count": len(front_defs),
                    "has_backward": bool(backward_defs),
                    "backward_match": backward_match,
                    "dispatch_summary": dispatch_summary,
                    "aclnn_completeness": aclnn_completeness,
                    "backward_bindings": [
                        {
                            "name": b.get("name"),
                            "signature": b.get("name_decl"),
                            "differentiable_inputs": b.get("differentiable_inputs", []),
                            "line": b.get("line"),
                            "uri": derivatives_uri,
                            "matched_by": backward_match,
                        }
                        for b in backward_defs
                    ],
                    "paths": rendered_paths,
                }
            )

    finally:
        client.close()

    (out_dir / "chains.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    (out_dir / "chains.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    key_results = [build_key_record(x) for x in results]
    (out_dir / "key_chains.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in key_results) + "\n", encoding="utf-8"
    )
    (out_dir / "key_chains.json").write_text(json.dumps(key_results, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "total_top_ops": len(top_ops),
        "ok": sum(1 for x in results if x.get("status") == "ok"),
        "with_paths": sum(1 for x in results if x.get("paths")),
        "with_backward": sum(1 for x in results if x.get("has_backward")),
        "with_gap_scan": sum(1 for x in results if (x.get("aclnn_completeness") or {}).get("judge_source") != "disabled"),
        "with_gap_suspects": sum(1 for x in results if (x.get("aclnn_completeness") or {}).get("suspected_missing_apis")),
        "elapsed_sec": round(time.time() - t0, 2),
        "out_dir": str(out_dir),
        "files": {
            "jsonl": str(out_dir / "chains.jsonl"),
            "json": str(out_dir / "chains.json"),
            "key_jsonl": str(out_dir / "key_chains.jsonl"),
            "key_json": str(out_dir / "key_chains.json"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("reconstruct done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
