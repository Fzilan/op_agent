---
name: torch-npu-aclnn-reconstruct
description: Reconstruct top-op to ACLNN tree call chains for Torch NPU, including path conditions and report artifacts.
---

# Torch NPU ACLNN Reconstruct

This skill outputs top-op downstack tree chains to ACLNN with chain conditions.

## Step 1: Reconstruct and Render

Reconstruct `top-op -> aclnn` chains and infer path conditions based on lsp server. It use `clangd` to call hierarchy and infer simple path conditions from nearby `if/switch/case` snippets.

```bash
RUN_DIR="$PWD/workspace/runs/run-001/reconstruct-chains"
python3 skills/torch-npu-aclnn-reconstruct/scripts/run_torch_npu_chain.py \
  --top-ops add,div \
  --out-dir "$RUN_DIR"
```

Outputs:
- `$RUN_DIR/chains.json`
- `$RUN_DIR/key_chains.json`
- `$RUN_DIR/report.md`
- `$RUN_DIR/<op>.md`
- `$RUN_DIR/<op>.mermaid`

## Step 2: Fallback Review

This is an optional supplement step for coverage completion. Run it only when you need gap-filling, or when `aclnn_completeness.gap_candidates` in `chains.json` is non-empty.

1. For each operator in `$RUN_DIR/chains.json`, check `aclnn_completeness.gap_candidates` and `evidence` item (`uri`, `line`, `snippet`). 
2. Open the related C++ implementation file (`.cpp`) and read the surrounding function/helper body.
3. Compare candidate evidence with reconstructed `paths` of the same operator:
- same operator/helper/out/inplace family
- branch/condition consistency
- likely missing LSP link vs clearly unrelated operator logic

Why this step is required:
- Step 1 is path reconstruction, not a complete proof of all reachable ACLNN calls.
- `gap_candidates` highlights ACLNN mentions found in related files but not linked in current chains.
- Source-level verification prevents both false negatives (missed helper paths) and false positives (same-file but other-op calls).

Write the gap review output to:
- `$RUN_DIR/llm_gap_results.json`

Required schema as:

```json
{
  "results": [
    {
      "operator": "add",
      "aclnn_api": "aclnnAddV3",
      "likely_related": true,
      "confidence": 0.96,
      "reason": "helper path in add_out_npu_nocheck matches add Tensor branch context"
    },
    {
      "operator": "add",
      "aclnn_api": "aclnnInplaceAdd",
      "likely_related": true,
      "confidence": 0.93,
      "reason": "belongs to add_ inplace branch in same operator family"
    }
  ]
}
```

## Step 3: Backfill and Regenerate Report

Apply Step 2 results:

```bash
python3 tools/reconstruct-chains/postprocess/apply_gap_backfill.py \
  --chains-json "$RUN_DIR/chains.json" \
  --key-json "$RUN_DIR/key_chains.json" \
  --llm-results-json "$RUN_DIR/llm_gap_results.json"
```

Regenerate markdown and mermaid:

```bash
python3 tools/reconstruct-chains/common/render_report.py \
  --chains-json "$RUN_DIR/chains.json" \
  --summary-json "$RUN_DIR/summary.json" \
  --out-dir "$RUN_DIR"
```

Verify backfill effects:
- `chains.json`: `aclnn_completeness.judge_source == "skill_backfill"`
- `chains.json`: `final_api_catalog` includes accepted gap APIs
- `<op>.md` tree includes `[Related]` nodes for `likely_related=true`
- `[Related]` reason comes from `gap_candidates[*].llm_judgement.reason`

## Step 4: Display Answer

When replying to the user, show results in this order:

1. Operator-level tree chain (from `<op>.md` `## Tree` section).

example:
```
[Top/API] add
├─ [C++] add -> [ACLNN] aclnnAdd
   ├─ cond: overload(other: Tensor)
   └─ cond: if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
├─ [C++] add -> [ACLNN] aclnnAdds
   ├─ cond: overload(other: Tensor)
   └─ cond: if (other.dim() == 0 && !torch_npu::utils::is_npu(other)) {
└─ [C++] add -> [ACLNN] aclnnAdds
   └─ cond: overload(other: Scalar)
├─ [Related] add -> [ACLNN] aclnnAddV3
   └─ reason: add_out_npu_nocheck helper branch confirms aclnnAddV3 when scalar-self and kernel available
├─ [Related] add -> [ACLNN] aclnnInplaceAdd
   └─ reason: add_ Tensor overload enters inplace_add_out_npu_no_check and non-scalar branch uses aclnnInplaceAdd
├─ [Related] add -> [ACLNN] aclnnInplaceAdds
   └─ reason: add_ Tensor/Scalar overloads include aclnnInplaceAdds paths
```

2. Final ACLNN mapping list (from `aclnn_completeness.final_api_catalog`).
3. Backfill summary review/backfill. If no gap was performed, report it.
4. Evidence paths:
- `$RUN_DIR/<op>.md`
- `$RUN_DIR/chains.json`
- `$RUN_DIR/key_chains.json`
