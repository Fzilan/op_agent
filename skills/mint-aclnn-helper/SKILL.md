---
name: mint-aclnn-helper
description: Auto-invoked when users ask for MindSpore mint.* to ACLNN call-chain reconstruction and inventory, including mint entry lookup, api_def/op_yaml mapping, primitive/Ascend dispatch checks (kbk/pyboost), and backward definition/ops verification.
---

# Mint ACLNN Helper

This skill helps reconstruct the `mindspore.mint` to ACLNN chain and perform inventory.

## When to Use

Use this skill when:
- Questions about `mindspore.mint.*` to `api_def`/`op_yaml`/`primitive` mapping.
- Requests to check Ascend dispatch status (`auto_generate`, `customize`, or unsupported).
- Requests to verify `kbk`, `pyboost`, `backward_defined`, and `backward_ops`.
- Requests for mint-to-aclnn inventory or verification reports.

## Instructions

### Step 1: read `./reference/mint_to_aclnn_precheck_guide.md`

### Step 2: deliver an inventory report using the result fields and display style in the reference
