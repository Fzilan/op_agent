---
name: aclnn-builder-with-op-index
description: Guides end-to-end ACLNN custom operator development and adaptation in MindSpore (PyBoost/Pynative + KBK/Graph paths), including YAML definitions, code generation, GeneralInfer, kernel registration, bprop wiring, tests (UT), and docs. Use when the user mentions ACLNN, Ascend, operator adaptation/operator development, PyBoost, KBK, or Ascend operator adaptation tasks.
---

# ACLNN Operator Development End-to-End Flow (MindSpore Adaptation)

## Goal

Land an ACLNN operator on the Ascend platform in MindSpore **end to end**: forward and backward paths, both PyBoost (Pynative) and KBK (Graph), dynamic shape/rank support, UT, documentation, export, and the required quality checks and validation.

For the pre-check stage, use the distilled `operator-facts` index as the default entry point so you can quickly resolve the MindSpore execution unit, PTA target form, implementation gap, and `auto` / `customize` conclusion before doing broader source inspection.

## How To Use This Skill

- When the user says things like "integrate/adapt an ACLNN operator into MindSpore", "add an xxx interface to MindSpore that matches `torch_npu`", "use the skill to add an NPU operator", "add `xxx_op.yaml`", "how should PyBoost/KBK be written", proceed directly through this skill's workflow.

## Execution Flow

### Workflow Index

When using this skill to develop an ACLNN operator, **create a TODOLIST** and execute the following workflows in order.
**Steps marked `🔒 must not be skipped` are mandatory in every scenario.**
**Places marked `⛔ HARD GATE` must be completed before you continue, otherwise stop and wait for user confirmation.**

> The Feature template is `templates/feature-document.md`.

### [Step 0](workflows/00-pre-checks.md): facts-first pre-check

`🔒 must not be skipped`

- Input: MindSpore API / operator / primitive / yaml branch, with optional PTA reference API
- Primary source: `../_shared/operator-facts`
- Open source files in codebase only when facts are missing or conflicting
- Output: pre-check handoff + initialized Feature document path
- `⛔ HARD GATE`: finish Step 0 before entering Step 1

### [Step 1](workflows/01-yaml-definition.md): YAML definition

- Follow the workflow to complete `op_def` / `api_def`.
- Input: pre-check handoff, Feature document
- Output: `op_def` and `api_def`
- Backfill: `feature-document.md#feature-yaml-definition`

### [Step 2](workflows/02-general-infer.md): GeneralInfer and infer UT

- Follow the workflow to complete infer logic, dynamic support, and infer UT.
- Backfill: `feature-document.md#feature-dynamic-shape`, `feature-document.md#feature-validation-and-errors`, `feature-document.md#feature-test-plan`

### [Step 3](workflows/03-aclnn-kernel.md): ACLNN kernel (PyBoost + KBK)

- Follow the workflow to complete kernel adaptation.
- Path 1: validate auto-generated outputs
- Path 2: handwrite Customize implementation
- Backfill: `feature-document.md#feature-execution-modes`

### [Step 4](workflows/04-bprop.md): BPROP implementation

- Follow the workflow to complete backward adaptation.
- Backfill: `feature-document.md#feature-bprop`

### [Step 5](workflows/05-export.md): export python interface

- Follow the workflow to complete public interface export.
- Backfill: `feature-document.md#feature-task-list`

### [Step 6](workflows/06-docs.md): documentation

- Follow the workflow to complete API documentation.
- Public `mint` / `ops` / `nn` / `Tensor` interfaces must not skip Chinese RST
- Backfill: `feature-document.md#feature-task-list`

### Step 7: Feature document finalization

`🔒 must not be skipped`

- Follow the workflow to complete final Feature cleanup.
- Backfill: remaining Feature sections and `feature-document.md#feature-task-list`
- Backfill remaining sections and finish the task list

## Validation Loop (Evidence Required At Every Step) `🔒 must not be skipped`

After every completed step, an execution report **must** be presented to the user using the template below. It may not be omitted, merged away, or deferred.
**This is a mandatory user-facing deliverable, not an internal note.**

```text
━━━ Step X Execution Report ━━━

Execution basis (which skill requirement I followed):
- workflow file: workflows/XX-xxx.md
- corresponding skill requirement: (quote the relevant item from SKILL.md / the workflow)
- success criteria for this step: (copied from the workflow success criteria)

What I did (deliverables):
- ...

Key evidence (code snippets / file paths / search results):
- ...
- Which existing operator implementation I compared against: ...

Validation result:
- ...

Item-by-item success criteria check:
- [ ] Criterion 1: ✅/❌
- [ ] Criterion 2: ✅/❌
- ...

Open issues / risks / next step:
- ...
```
