# Workflow 3: Aclnn Kernel (Pynative, C++)

Path convention: unless stated otherwise, `reference.md` means `../_shared/reference.md` and `aclnn_doc` means `../_shared/aclnn_doc/`.

## Goal

Implement the ACLNN kernel for Pynative And Graph(KBK) mode.
**The workload of this step differs greatly depending on the integration path:**
- **Path 1 (auto-generated)**: `gen_ops.py` already generated the full call code, so this step only needs **validation**
- **Path 2 (Customize)**: you must handwrite kernel implementation

## Inputs

- **Integration path**: auto / customize
- **YAML definition**: parameter list, `dispatch` configuration
- **PTA source analysis**: ACLNN call details and parameter preprocessing logic
- **(Composite scenarios)** ACLNN call chain: sub-operator list and dependency order

## Outputs

- **Path 1**: validated auto-generated aclnn kernel code
- **Path 2**: handwritten aclnn kernel implementation files

---

## Steps

### Path 1(auto generate): Validate The Auto-Generated Artifacts

1. **Confirm the generated code exists**
   - For Pyboost, check registraion `MS_REG_PYBOOST_OP(Ascend, {PrimitiveName})` in `mindspore/ops/kernel/ascend/aclnn/pyboost_impl/auto_generate/pyboost_ascend_ops_*.cc`
   - For KBK, check registration `MS_ACLNN_COMMON_KERNEL_FACTORY_REG({PrimitiveName}, ...})` or `MS_ACLNN_KERNEL_FACTORY_REG({PrimitiveName}, ...`.

### Path 2(customize kernel): Handwrite Customize Files

#### Pyboost

##### Single-Operator Direct Call Pattern

Standard three-part structure (`reference.md#pyboost-reference`):
1. output tensor allocation
2. argument conversion (`tuple -> vector`, `None` handling, and so on)
3. two-stage ACLNN invocation (`LAUNCH_ACLNN` or the project's equivalent macro)

##### Composite Operator Pattern (Composing Small C++ Operator APIs)

When the target operator is implemented as a composition of multiple smaller operators (`reference.md#composite-pyboost-pattern`):
1. include `#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"`
2. call small-operator C++ APIs such as `add()`, `mul()`, `sum_ext()`, and so on to compose the logic directly, **without manually calling `LAUNCH_ACLNN`**
3. set `bprop_expander: False` in YAML so each small operator handles autodiff by itself
4. if the large operator already has its own explicit bprop, use `RequireGradGuard(false)` to prevent the small operators from triggering autodiff twice

##### Input Argument Conversion (`reference.md#pyboost-argument-normalization`)

- `tuple` / `list` -> `std::vector<int64_t>`
- optional `None` inputs -> define the `None` semantics and handle them consistently in PyBoost / Infer / KBK
- scalar parameters -> extract using the project's scalar wrapper pattern

##### Inplace Op

For inplace operator, its input tensor is used as the output tensor, so there's no need for calling infer and allocating output tensor.

Check `mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/inplace_add_ext.cc` as an example for inplace operator pyboost kernel.

#### Aclnn Kernelmode(KBK mode)

- The aclnn interface name is passed to base Class(`AclnnKernelMod`)'s constructor.
- `GetWorkSpaceInfo()`: call `GetWorkspaceForResize` for workspace allocation
- `Launch()`: call `RunOp` or the equivalent execution path
- For unneeded outputs, override `GetUseLessOutputIdx()`
- non-Tensor type should be extracted from `KernelTensor` to primitive c++ type using `device::ascend::ConvertKernelTensor<>`
- registration: `MS_ACLNN_KERNEL_FACTORY_REG`

##### Composite Operator Pattern (Meta DSL, `reference.md#composite-kbk-pattern`)

Meta DSL uses C++ graph construction instead of manual `GetWorkSpaceInfo` / `Launch` / `RunOp`, and the framework then handles type inference and autodiff automatically:
1. create a new `.cc` file under `mindspore/ccsrc/frontend/operator/meta_dsl/func_op/`
2. register the operator with `REGISTER_FUNCTION_OP(OpName)` and optionally pass a validation function
3. inside `BeginFunction(OpName, args...) { ... } EndFunction(OpName)`, compose sub-operators with `Call(Prim(SubOp), ...)`
4. the framework handles multi-platform adaptation automatically, so **no handwritten KBK kernel file is needed**

Code skeletons are available in `reference.md#kbk-skeleton` for single operators and `reference.md#composite-kbk-pattern` for Meta DSL, but **the repository's current code remains the final reference**.

---

## Success Criteria

**Path 1**:
- [ ] Confirm the auto-generated PyBoost and Aclnn Kernelmod registration

**Path 2**:
- [ ] Customize PyBoost call function and Aclnn Kernelmod are implemented
- [ ] Argument conversion is correct (`tuple` / `None` / scalar)
- [ ] In composite scenarios, the calculation logic matches PTA, and intermediate tensors are correctly allocated in Pyboost kernel
---
