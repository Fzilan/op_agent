# Torch NPU Chain Reconstruction Strategy

This document describes the practical reconstruction strategy for `torch_npu -> ACLNN` mapping and how to improve ACLNN completeness using gap scan + LLM judgement.

## Goals

1. Front-end semantic set (front signatures + overload hints)
2. Branch-condition semantic set (path conditions from code)
3. ACLNN mapping catalog (primary output)
4. Dispatch shape note:
   - strict direct call
   - helper-mediated call
   - mixed
5. Backward bindings (from `derivatives.yaml`)

## Reconstruction Pipeline

1. Entry discovery:
   - find candidate C++ symbols with `rg`
   - initialize call hierarchy by `clangd prepareCallHierarchy`
2. Path traversal:
   - LSP outgoing calls
   - definition hops inside function body
   - direct ACLNN text hit extraction in current function range
3. Config fallback:
   - if no path in C++/LSP, use `op_plugin_functions.yaml` (`exec` / `structured_inherit`)
4. Backward matching:
   - match with `derivatives.yaml` by exact name, then normalized root
5. LLM review:
   - scan the exact related C++ files and collect all `aclnnXxx`
   - backfill judgement into output artifacts if any `aclnnXxx` if not detected by lsp-based reconstruct

## Details for the run scripts

`tools/reconstruct-chains/torch_npu/run.py`

Use `clangd` call hierarchy to reconstruct `top-op -> aclnn` chains and infer path conditions.
Computation and rendering are decoupled:
- `run.py`: compute chains and write structured data
- `../common/render_report.py`: read structured data and generate markdown/mermaid reports

### Inputs

1. Top ops: `--top-ops` (comma separated) or `--top-ops-file` (one per line)
2. ACLNN full set: `--aclnn-set` (txt/json)
3. op-plugin repo root: `--op-plugin-root`
4. Optional config fallback source: `--op-plugin-functions-yaml` (default: `<op-plugin-root>/op_plugin/config/op_plugin_functions.yaml`)
5. Optional backward source: `--derivatives-yaml` (default: `<op-plugin-root>/op_plugin/config/derivatives.yaml`)

### Output

Under `--out-dir`:

- `chains.jsonl`: one operator per line
- `chains.json`: full array
- `summary.json`: run summary

Then run renderer to generate:

- `report.md`: markdown summary index
- `<op>.md`: markdown report (tree + mermaid + path conditions)
- `<op>.mermaid`: mermaid graph text

Structured outputs are split into two levels:

- Full trace/audit: `chains.jsonl`, `chains.json`
- Key writer input: `key_chains.jsonl`, `key_chains.json`

Each path includes:

- `aclnn_api`
- `chain` (symbol nodes with file/line)
- `path_conditions` (heuristic `if/else if/switch/case` near call edge)
- `path_source` (`lsp` or `config_fallback`)

Operator-level metadata includes:

- `front_signatures` + `overload_count` (from `op_plugin_functions.yaml`)
- `has_backward` + `backward_bindings` (from `derivatives.yaml`)
- `backward_match` (`exact` / `normalized` / `none`)
- `dispatch_summary` (`direct` / `helper` / `mixed` / `logic_preprocessed`)
- `aclnn_completeness` (observed ACLNN, scanned ACLNN mentions, gap suspects, LLM judgement)

Path-level metadata includes:

- `dispatch_note`:
  - `strict_direct`: top function directly calls ACLNN, with no extra non-overload branch condition
  - `helper`: ACLNN reached through at least one helper function
  - `logic_preprocessed`: top function path has non-overload logic condition before ACLNN

### Example

```bash
python3 run.py \
  --workspace /path/to/workspace \
  --op-plugin-root /path/to/workspace/op-plugin \
  --top-ops abs,mul,div \
  --aclnn-set /path/to/torch_npu_aclnn_op_full_set.txt \
  --enable-aclnn-gap-scan \
  --max-depth 3 \
  --out-dir /path/to/aclnn-analyzer/workspace/runs/run-001/reconstruct-chains

python3 ../common/render_report.py \
  --chains-json /path/to/aclnn-analyzer/workspace/runs/run-001/reconstruct-chains/chains.json \
  --summary-json /path/to/aclnn-analyzer/workspace/runs/run-001/reconstruct-chains/summary.json \
  --out-dir /path/to/aclnn-analyzer/workspace/runs/run-001/reconstruct-chains
```

### Notes

- This tool **must use LSP** (`clangd`) for call hierarchy traversal.
- Some operators may return `no_path` if C++ call hierarchy cannot reach ACLNN symbol (e.g., generated path not materialized in scanned C++ sources).
- When LSP/C++ has no ACLNN hit, `run.py` falls back to `op_plugin_functions.yaml` (`gen_opapi.exec` + `structured_inherit`) to recover config-declared ACLNN mapping.
- For code-writer/agent consumption, prefer `chains.jsonl` (stream-friendly) or `chains.json` (full batch).

