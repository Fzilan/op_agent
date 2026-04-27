# Workflow 6: Documentation

Path convention: unless stated otherwise, `reference.md` means `../_shared/reference.md` and `aclnn_doc` means `../_shared/aclnn_doc/`.

## Goal

Complete the English `function_doc` (YAML) and the **Chinese RST documentation**, keeping the two strictly aligned.

**What EN/CN consistency means**: parameter names, default values, required/optional status, semantics, and examples must match.
Each language should still follow its own documentation conventions; literal sentence-by-sentence identity is not required.

## Inputs

- **OP and API(optional) YAML definition**

## Outputs (Two Documentation Types + Interface Lists, Confirmed One By One)

| Type | File Location |
| --- | --- | --- | --- |
| **English `function_doc`** | `ops/op_def/yaml/doc/{op}_doc.yaml` |
| **Chinese RST** | `docs/api/api_python/ops/*.rst` (or the matching `mint` / `nn` directory) |
| **Interface Registration List** | `docs/api/api_python/mindspore.xxx.rst`, `docs/api/api_python_en/mindspore.xxx.rst` |

---

## Steps

### Step 1: Public Interface Registration

For functional interface, register public functional interface in `docs/api/api_python/mindspore.mint.rst` for Chinese, and `docs/api/api_python_en/mindspore.mint.rst` for English.

Only public interfaces should be registered. The internal implementation, e.g. interfaces with suffix `_ext`, or interfaces like `xxx_tensor` and `xxx_scalar` that are used in `xxx` to deal with different inputs, should never be registered. Only functional interface should be registered here, don't register for Primitive.

Tensor method should be registered in `docs/api/api_python/mindspore/mindspore.Tensor.rst` for Chinese and `docs/api/api_python_en/mindspore/mindspore.Tensor.rst` for English.


### Step 2: Add The English Doc
For newly added `xxx.yaml` in `mindspore/ops/api_def`, add docs in `mindspore/ops/api_def/function_doc` and `mindspore/ops/api_def/method_doc`, according to the interface type.

If there's no corresonding `api_def` defined for the op, for each newly added `xxx_op.yaml` in `mindspore/ops/op_def`, add the corresponding doc in `mindspore/ops/op_def/yaml/doc`. 

Make sure the doc contains following sections:
- `desc`: short description of the operator; for public APIs, include principles, formulas, paper references, or other necessary background when appropriate
- `args`: description for each parameter
- `returns`: return-value description
- `examples`: a complete runnable example including imports

### Step 2: Chinese RST (`docs/api/api_python`)

- file location: under `docs/api/api_python/mint/`
- **first inspect existing Chinese RST files for similar operators** to confirm the format and directory structure
- **filename, in-file title, and interface definition must match exactly** (for functional interfaces, usually only the filename has the extra `func_` prefix)
- the underline of `=` below the title must be at least as long as the title itself
- update interface index files in alphabetical order

### Step 3: Consistency Check (`reference.md#documentation-general-principles`)

| Check Item | English | Chinese |
| --- | --- | --- |
| Parameter names | ✅ consistent | ✅ consistent |
| Default values | ✅ consistent | ✅ consistent |
| Required/optional status | ✅ consistent | ✅ consistent |
| Examples | ✅ runnable | ✅ runnable |

---

## 🔒 Mandatory Check Before Marking Step 6 Complete

```text
Documentation deliverable checklist:

English function_doc (YAML):
  - File path: mindspore/ops/api_def/function_doc/{op}_doc.yaml, mindspore/ops/api_def/tensor_doc/{op}_doc.yaml, ops/op_def/yaml/doc/{op}_doc.yaml
  - Status: ✅ created / ❌ not written (reason: ___)

Chinese RST:
  - File path: docs/api/api_python/ops/mindspore.ops.func_{op}.rst (or the matching mint/Tensor directory)
  - Status: ✅ newly created / ✅ already exists and covers the new interface / ❌ not written (reason: ___)
  - If skipped: is this an internal operator (not a public API)? yes / no

Interface lists (English + Chinese):
  - Added to the matching mindspore.xxx.rst file in alphabetical order? yes / no

EN/CN consistency:
  - Parameter names consistent: yes / no
  - Default values consistent: yes / no
  - Examples consistent and runnable: yes / no
```

> **Public APIs (functional / mint / nn / Tensor) must have Chinese RST.**
> Only **internal operators** that are not exported in `__all__` and do not need public docs may skip it.
> When skipping, you must state the reason explicitly. Silent skipping is not allowed.

## Success Criteria

- [ ] The English `function_doc` is complete (`desc` / `args` / `returns` / `examples`)
- [ ] **The Chinese RST file has been created** for public APIs, or the operator is explicitly marked as internal and skippable
- [ ] Parameter names, default values, and examples are strictly consistent between English and Chinese
- [ ] Examples are runnable and include complete imports
- [ ] The interface lists have been updated in alphabetical order
- [ ] Filename, title, and interface definition all match

---
