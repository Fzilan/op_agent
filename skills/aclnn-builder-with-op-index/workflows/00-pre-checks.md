# Workflow 0: Pre-check v3

Path convention: unless stated otherwise, `operator-facts` means `../_shared/operator-facts`.

## Goal

Use `operator-facts` to answer four questions before coding:

- what MindSpore target should be adapted
- whether there is an ACLNN-related implementation gap
- whether the path should be `auto` or `customize`
- whether backward work is needed

## Rule Of Use

- `operator-facts` is the default fact source for pre-check.
- In normal cases, do not broadly search `mindspore` or `op-plugin`.
- Open source files only when facts are missing, conflicting, or still not enough to make a decision.

Why this rule exists:

- `operator-facts` is already the distilled index of the codebase.
- Pre-check is for making a decision, not for repeating repository-wide search.
- A short, stable facts-first flow is easier for both agents and human readers to follow.

## Facts You Will Use

- `operator-facts/bundles/entries/*.json`
  Start here when the inputs is a public MindSpore API such as `mindspore.ops.xxx`, `mindspore.Tensor.xxx`, or `mindspore.mint.xxx`. It tells you how that API routes in MindSpore: a single branch, multiple branches, or a composite implementation.
- `operator-facts/bundles/units/*.json`
  Start here when the inputs is an execution unit such as an operator, primitive, or yaml branch. It tells you which execution unit you are looking at, which public APIs route to it, and what aclnn coverage already exists.
- `operator-facts/data/pta_facts.jsonl`
  Use this to find the closest PTA reference and judge whether the path is closer to `auto` or `customize`.
- `operator-facts/data/ms_*.jsonl`
  Use these base facts tables only when the needed bundle is missing or unclear.

You do not need to understand the full schema of these files.
For pre-check, you only need to know which one to start from.

## Inputs

- MindSpore API, or
- MindSpore operator / primitive / yaml branch, or
- the above plus an optional PTA reference API from the user

## Step 1. Resolve the execution unit(s) and the PTA reference

First resolve the user input to the MindSpore execution unit(s), such as a regular operator branch, or a composite implementation. Then read current implementations and coverage facts of those execution unit(s).

1. Resolve the execution unit(s).
   - Public MindSpore API -> start from `bundles/entries/*.json`
   - Operator / primitive / yaml branch -> start from `bundles/units/*.json`
   - Keep all routed branches, and keep composite results as composite execution units.

2. Read the current MindSpore facts.
   - Identify the actual implementation and current coverage of the resolved execution unit(s).
   - If the needed bundle is missing or unclear, recover the facts from `data/ms_*.jsonl`.

3. Select the PTA reference.
   - Use `data/pta_facts.jsonl`.
   - Use the user-provided PTA API first when available; otherwise match by MindSpore names or ACLNN names.
   - Open source files only as a targeted fallback when facts are missing, conflicting, or still not enough.

Example: `mindspore.Tensor.max`

Start from `bundles/entries/mindspore.Tensor.max.json`.

```json
{
  "entry": {
    "public_api": "mindspore.Tensor.max",
    "entry_type": "overload"
  },
  "branches": [
    {
      "unit_id": "branch::max_op.yaml::Max",
      "primitive": "Max",
      "coverage": {"aclnn": ["aclnnMax"], "infer": true, "kbk": true, "pyboost": true, "bprop": true}
    },
    {
      "unit_id": "branch::max_dim_op.yaml::MaxDim",
      "primitive": "MaxDim",
      "coverage": {"aclnn": ["aclnnMaxDim"], "infer": true, "kbk": true, "pyboost": true, "bprop": true}
    },
    {
      "unit_id": "branch::maximum_op.yaml::Maximum",
      "primitive": "Maximum",
      "coverage": {"aclnn": ["aclnnMaximum"], "infer": true, "kbk": true, "pyboost": true, "bprop": true}
    }
  ]
}
```

What to read here:
- `entry_type: overload` means this API resolves to multiple execution units.
- `branches[*].unit_id` and `primitive` tell you which execution units are under analysis.
- `coverage` tells you what already exists for each execution unit.

Then check `data/pta_facts.jsonl`, for example:

```json
{
  "pta_key": "torch.max::(Tensor self, int dim, bool keepdim=False)",
  "pta_api": "torch.max",
  "forward_aclnn": ["aclnnMaximum", "aclnnMax", "aclnnMaxDim"],
  "composite": true,
  "preprocess_needed": true,
  "custom_output_needed": true
}
```

What to read here:
- `pta_key` tells you which PTA overload you are comparing against.
- `composite`, `preprocess_needed`, and `custom_output_needed` help you judge whether the PTA side is closer to `auto` or `customize`.



## Step 2. Make the decision

For the resolved execution unit(s), give a short direct conclusion.

1. ACLNN gap
   - Compare aclnn implementation facts within PTA and MindSpore.
   - Gaps or mismatch mean implementation is still needed to align the resolved execution unit(s) with the PTA target form.
   - For composite results, always separate composite-root status from leaf execution unit coverage status.

2. Primitive / interface strategy
   - Reuse the current primitive: the existing primitive is already compatible in functionality and parameter definition. Keep the existing YAML and add `dispatch` later.
   - Add a new `_ext` primitive: a related primitive already exists, but its signature or semantics are not compatible.
   - Add a new primitive directly: no suitable primitive or YAML exists for this operator.

3. Integration path
    - For each execution unit to implement, decide whether the target path should be `auto` or `customize`.
    - Use `auto` only when the target unit is a direct non-composite ACLNN path and the PTA facts show no preprocessing, custom output handling, or interface rewriting.
    - Otherwise use `customize`.

4. Backward
4. Backward
    - Compare the current MindSpore backward form with the PTA target backward form by checking both `bprop` and `bprop_units`.
    - If they do not match, or if PTA has dedicated backward operators, include the required backward execution unit(s) in the implementation plan.

For composite results, always separate:

- composite root status
- leaf execution unit coverage status

Covered leaf units do not automatically mean the composite wrapper itself is complete.

## Output Format

```text
Implementation plan:

Forward:
- ACLNN implementation needed: {yes / no}
- If yes:
  - Operator(s): {op1}, {op2}, ...
  - Execution detail: {primitive / unit_id / yaml_path}
  - Primitive / interface strategy: {reuse current primitive / add `_ext` primitive / add new primitive}
  - Integration path: {auto / customize}
- If no:
  - Reason: {already aligned / already covered / no forward work needed}

Backward:
- ACLNN implementation needed: {yes / no}
- If yes:
  - Operator(s): {op1}, {op2}, ...
  - Execution detail: {primitive / unit_id / yaml_path}
  - Primitive / interface strategy: {reuse current backward route / add backward unit}
  - Integration path: {auto / customize}
- If no:
  - Reason: {PTA has no backward / current backward already aligned}

Key diff:
- Forward: {PTA form} vs {MindSpore form}
- Backward: {PTA form} vs {MindSpore form}

PTA refs:
- {role} | {path} | {pattern}
- ...

Fallback opened:
- none
```

## Important Notes

- Keep the pre-check short. It is a decision step, not a design document.
- Do not explain the whole `operator-facts` model inside pre-check.
- If the facts are sufficient, do not reopen broad source analysis.
