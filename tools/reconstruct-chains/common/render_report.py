#!/usr/bin/env python3
"""Render markdown/mermaid reports from reconstruct-chains JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render markdown reports from chains.json")
    p.add_argument("--chains-json", type=Path, required=True, help="path to chains.json")
    p.add_argument("--summary-json", type=Path, default=None, help="optional summary.json path")
    p.add_argument("--out-dir", type=Path, required=True, help="output directory for *.md/*.mermaid")
    return p.parse_args()


def classify_layer_by_uri(uri: str) -> str:
    u = (uri or "").lower()
    if not u:
        return "Unknown"
    if u.endswith(".py"):
        return "Python"
    if u.endswith(".yaml") or u.endswith(".yml") or u.endswith(".json") or u.endswith(".toml"):
        return "Config"
    if u.endswith(".cpp") or u.endswith(".cc") or u.endswith(".cxx") or u.endswith(".c") or u.endswith(".h") or u.endswith(".hpp"):
        return "C++"
    return "Unknown"


def layer_tag_for_node(item: dict[str, Any]) -> str:
    return classify_layer_by_uri(str(item.get("uri", "")))


def _mm_clean(text: str) -> str:
    s = str(text)
    s = s.replace('"', "'")
    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("|", "/")
    s = " ".join(s.split())
    return s


def to_mermaid(op: str, paths: list[dict[str, Any]]) -> str:
    lines = ["graph TD"]
    id_map: dict[str, str] = {}
    c = 0

    def nid(lbl: str) -> str:
        nonlocal c
        if lbl in id_map:
            return id_map[lbl]
        c += 1
        id_map[lbl] = f"N{c}"
        return id_map[lbl]

    r = nid(f"top:{op}")
    lines.append(f'{r}["Top/API: {_mm_clean(op)}"]')
    for p in paths:
        chain = p.get("chain", [])
        prev = r
        for it in chain:
            nm = str(it.get("name", "?"))
            layer = layer_tag_for_node(it)
            cur = nid(f"{layer}:{nm}")
            lines.append(f'{cur}["{_mm_clean(layer)}: {_mm_clean(nm)}"]')
            lines.append(f"{prev} --> {cur}")
            prev = cur
        acl = str(p.get("aclnn_api", ""))
        leaf = nid(f"aclnn:{acl}")
        lines.append(f'{leaf}["ACLNN: {_mm_clean(acl)}"]')
        conds = p.get("path_conditions", [])
        label = ""
        if conds:
            label = '"' + _mm_clean("; ".join(conds[:3])) + '"'
        if label:
            lines.append(f"{prev} -->|{label}| {leaf}")
        else:
            lines.append(f"{prev} --> {leaf}")
    return "\n".join(dict.fromkeys(lines))


def append_related_mermaid(mermaid: str, related_items: list[dict[str, str]]) -> str:
    if not related_items:
        return mermaid
    lines = mermaid.splitlines()
    used_ids: set[str] = set()
    for ln in lines:
        if ln.startswith("N"):
            used_ids.add(ln.split("[", 1)[0].split(" ", 1)[0])
    max_id = 0
    for nid in used_ids:
        if nid.startswith("N"):
            try:
                max_id = max(max_id, int(nid[1:]))
            except Exception:
                pass
    top_id = None
    for ln in lines:
        if '["Top/API:' in ln:
            top_id = ln.split("[", 1)[0]
            break
    if not top_id:
        top_id = "N1"
    out = list(lines)
    for it in related_items:
        api = str(it.get("aclnn_api", ""))
        reason = str(it.get("reason", ""))
        if not api:
            continue
        max_id += 1
        rel = f"N{max_id}"
        max_id += 1
        leaf = f"N{max_id}"
        out.append(f'{rel}["Related: {api}"]')
        out.append(f"{top_id} --> {rel}")
        out.append(f'{leaf}["ACLNN: {api}"]')
        label = _mm_clean(reason) if reason else "related_by_gap_review"
        out.append(f'{rel} -->|"{label}"| {leaf}')
    return "\n".join(dict.fromkeys(out))


def to_tree_text(op: str, paths: list[dict[str, Any]], related_items: list[dict[str, str]] | None = None) -> str:
    lines: list[str] = [f"[Top/API] {op}"]
    if not paths:
        lines.append("└─ (no path)")
    else:
        for i, p in enumerate(paths):
            branch = "└─" if i == len(paths) - 1 else "├─"
            chain_nodes = [f"[{layer_tag_for_node(x)}] {x.get('name', '?')}" for x in p.get("chain", [])]
            chain = " -> ".join(chain_nodes) if chain_nodes else op
            lines.append(f"{branch} {chain} -> [ACLNN] {p.get('aclnn_api', '')}")
            conds = p.get("path_conditions", [])
            if conds:
                for j, c in enumerate(conds):
                    c_branch = "   └─" if j == len(conds) - 1 else "   ├─"
                    lines.append(f"{c_branch} cond: {c}")
    rels = related_items or []
    if rels:
        for it in rels:
            api = str(it.get("aclnn_api", ""))
            reason = str(it.get("reason", "")) or "related by gap review"
            if not api:
                continue
            lines.append(f"├─ [Related] {op} -> [ACLNN] {api}")
            lines.append(f"   └─ reason: {reason}")
    return "\n".join(lines)


def render_md(op: str, op_result: dict[str, Any], mermaid: str) -> str:
    paths = op_result.get("paths", [])
    front_sigs = op_result.get("front_signatures", [])
    backward = op_result.get("backward_bindings", [])
    dispatch = op_result.get("dispatch_summary", []) or []
    completeness = op_result.get("aclnn_completeness", {}) or {}
    gap_candidates = completeness.get("gap_candidates", []) or []
    related_items: list[dict[str, str]] = []
    for g in gap_candidates:
        j = (g or {}).get("llm_judgement", {}) or {}
        if bool(j.get("likely_related", False)):
            related_items.append(
                {"aclnn_api": str((g or {}).get("aclnn_api", "")), "reason": str(j.get("reason", ""))}
            )
    related_items.sort(key=lambda x: x.get("aclnn_api", ""))
    tree = to_tree_text(op, paths, related_items=related_items)
    out: list[str] = []
    out.append(f"# `{op}` Call Chain")
    out.append("")
    out.append(f"- status: `{op_result.get('status')}`")
    out.append(f"- entries: `{op_result.get('entries', 0)}`")
    out.append(f"- visited_nodes: `{op_result.get('visited_nodes', 0)}`")
    out.append(f"- paths: `{len(paths)}`")
    out.append(f"- overload_count: `{op_result.get('overload_count', 0)}`")
    out.append(f"- has_backward: `{op_result.get('has_backward', False)}`")
    out.append(f"- backward_match: `{op_result.get('backward_match', 'none')}`")
    out.append(f"- aclnn_catalog_size: `{len(completeness.get('final_api_catalog', []) or [])}`")
    out.append(f"- aclnn_gap_suspects: `{len(completeness.get('suspected_missing_apis', []) or [])}`")
    out.append(f"- gap_judge_source: `{completeness.get('judge_source', 'n/a')}`")
    out.append("")
    out.append("## Front Signatures")
    out.append("")
    if front_sigs:
        for s in front_sigs:
            out.append(f"- `{s.get('signature', '')}`")
    else:
        out.append("- `(none)`")
    out.append("")
    out.append("## Backward Bindings")
    out.append("")
    if backward:
        for b in backward:
            out.append(f"- `{b.get('signature', '')}` | differentiable_inputs=`{','.join(b.get('differentiable_inputs', []))}`")
    else:
        out.append("- `(none)`")
    out.append("")
    out.append("## Dispatch Summary")
    out.append("")
    if dispatch:
        for d in dispatch:
            out.append(
                f"- `{d.get('aclnn_api', '')}` | shape=`{d.get('dispatch_shape', '')}` | "
                f"strict_direct=`{d.get('has_strict_direct', False)}` | paths=`{d.get('path_count', 0)}`"
            )
    else:
        out.append("- `(none)`")
    out.append("")
    out.append("## ACLNN Completeness")
    out.append("")
    out.append(f"- observed_apis: `{', '.join(completeness.get('observed_apis', []) or [])}`")
    out.append(f"- final_api_catalog: `{', '.join(completeness.get('final_api_catalog', []) or [])}`")
    suspects = completeness.get("suspected_missing_apis", []) or []
    out.append(f"- suspected_missing_apis: `{', '.join(suspects) if suspects else '(none)'}`")
    gaps = completeness.get("gap_candidates", []) or []
    if gaps:
        out.append("- gap_candidates:")
        for g in gaps:
            j = g.get("llm_judgement", {}) or {}
            likely = (j.get("likely_related") if "likely_related" in j else None)
            conf = (j.get("confidence") if "confidence" in j else None)
            out.append(
                f"  - `{g.get('aclnn_api', '')}` | likely_related=`{likely}` "
                f"| confidence=`{conf}` | reason=`{j.get('reason', '')}`"
            )
    out.append("")
    out.append("## Layer Legend")
    out.append("")
    out.append("- `Top/API`: top op entry name")
    out.append("- `Python`: symbol from `.py`")
    out.append("- `Config`: symbol from `.yaml/.yml/.json/...`")
    out.append("- `C++`: symbol from `.cpp/.h/...`")
    out.append("- `ACLNN`: backend aclnn operator")
    out.append("")
    out.append("## Tree")
    out.append("")
    out.append("```text")
    out.append(tree)
    out.append("```")
    out.append("")
    out.append("## Graph")
    out.append("")
    out.append("```mermaid")
    out.append(append_related_mermaid(mermaid, related_items))
    out.append("```")
    out.append("")
    out.append("## Paths")
    out.append("")
    for idx, p in enumerate(paths, 1):
        out.append(f"### {idx}. `{p.get('aclnn_api', '')}`")
        out.append("")
        chain = " -> ".join([f"[{layer_tag_for_node(x)}] {x.get('name', '?')}" for x in p.get("chain", [])])
        out.append(f"- chain: `{chain}`")
        out.append(f"- source: `{p.get('path_source', 'lsp')}`")
        out.append(f"- dispatch_note: `{p.get('dispatch_note', 'unknown')}`")
        endpoint = p.get("endpoint") or {}
        if endpoint:
            out.append(f"- endpoint: `{endpoint.get('uri','')}`:{endpoint.get('line','')}:{endpoint.get('column','')}")
        conds = p.get("path_conditions", [])
        if conds:
            out.append("- conditions:")
            for c in conds:
                out.append(f"  - `{c}`")
        else:
            out.append("- conditions: `(none)`")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results = json.loads(args.chains_json.read_text(encoding="utf-8"))
    summary: dict[str, Any] | None = None
    if args.summary_json and args.summary_json.exists():
        summary = json.loads(args.summary_json.read_text(encoding="utf-8"))

    for r in results:
        op = str(r.get("operator", ""))
        if not op:
            continue
        mermaid = to_mermaid(op, r.get("paths", []))
        (out_dir / f"{op}.mermaid").write_text(mermaid, encoding="utf-8")
        (out_dir / f"{op}.md").write_text(render_md(op, r, mermaid), encoding="utf-8")

    report_lines = ["# Reconstruct Chains Report", ""]
    if summary is not None:
        report_lines.extend(["## Summary", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    report_lines.extend(["## Operators", ""])
    for r in results:
        op = r.get("operator", "")
        report_lines.append(f"- `{op}`: status=`{r.get('status')}`, paths=`{len(r.get('paths', []))}` ([md]({op}.md))")
    report_lines.append("")
    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("render done")
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
