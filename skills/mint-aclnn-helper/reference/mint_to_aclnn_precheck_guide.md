# MindSpore Mint-to-aclnn Pre-check Guide

This guide is used to inventory whether a `mindspore.mint` API has an aclnn sink path, and whether its backward definition exists.

### Scope

This document focuses on the following questions:
- Whether the target API has a `mint` entry.
- Which non-deprecated `op_yaml` branches the API maps to.
- For each `op_yaml`, which Primitive it maps to.
- Whether the Primitive enters the Ascend dispatch chain.
- Whether the forward path is `auto_generate` or `customize`.
- Whether `kbk` and `pyboost` are present.
- Whether the backward definition exists.
- Which backward ops are used in the backward body.

### Result Fields

API-level fields:
- `mint_entry`
- `api_def`

Per-branch fields:
- `op_yaml`
- `primitive`
- `ascend_dispatch`
- `kbk`
- `pyboost`
- `backward_defined`
- `backward_ops`

### Naming Conventions

- use `class.name` as `primitive` whenever it exists, e.g. `abs_op.yaml` -> `Abs`
- if `class.name` is missing, derive `primitive` from the op name in yaml, e.g. `inplace_fill_scalar_op.yaml` -> `InplaceFillScalar`
- `op_yaml` is usually `snake_case`, while `primitive` is usually `CamelCase`, e.g. `sub_ext_op.yaml` -> `SubExt`
- suffixes such as `inplace`, `ext`, `scalar`, and `grad` are usually kept in `primitive`, e.g. `AddScalar`, `SoftmaxBackward`

### Step 1. mint entry -> op yaml

check whether the target `mint.xxx` is a public API
- yes -> locate the inner function for api_def -> op_def and the primitive
- no -> go to step x -> check ops function

method: search in `mindspore/python/mindspore/mint/__init__.py` and resolev all corresponding non-deprecated `op_yaml`

output:
- `mint_entry` - ✅/✖️
- `api_def` - <api_name>.yaml / ✖️
- `op_yaml` - <op_name>_op.yaml / ✖️

#### Case 1: `functional_overload` export

exported from `mindspore.ops.functional_overload` -> do not infer a single primitive directly -> check `ops/api_def/<api>.yaml` first -> continue with each non-deprecated `op_yaml` branch

#### Case 2 : alias export

exported `xxx_ext as xxx` -> primitive is `XxxExt` not `Xxx` -> `xxx_ext_op.yaml`

#### Case 3: `auto_generate` direct export

`xxx` exported from `mindspore.ops.auto_generate` -> primitive is `Xxx` -> `xxx_op.yaml`

#### Case 4: `ops.function` wrapper export

exported from `mindspore.ops.function` -> inspect the function body first -> `api_def` or directly through `op_def`

Example 1:

`from mindspore.ops.function.math_func import divide` -> navigate to the definition of `divide` by LSP -> `return div(input, other, rounding_mode=rounding_mode)` -> check `ops/api_def/div.yaml` -> continue with its non-deprecated branches `divs_op.yaml`, `div_op.yaml`, `divmods_op.yaml`, `divmod_op.yaml`

Example 2

`from mindspore.ops.function.array_func import full_ext as full` -> navigate to the definition of `full_ext` by LSP -> `return fill_scalar_(size, fill_value, dtype)` -> `ops/op_def/yaml/fill_scalar_op.yaml`

Example 3:

`from mindspore.ops.function.nn_func import softmax_ext` -> navigate to the definition of `softmax_ext` by LSP -> the function first normalizes `dim` and `dtype`, then `return softmax_impl(input, dim)` -> directly check `ops/op_def/yaml/softmax_op.yaml`


#### Case 5: does not exists in mint
no entry in mint -> step x for ops (TBD)


### 2. Resolve Forward Fields from `op_yaml`

for each non-deprecated `op_yaml`, check the corresponding file in `mindspore/ops/op_def/yaml/`

- read `primitive`: use the UpperCamelCase form of the YAML top-level operator name. 
``` yaml
#operator fill_scalar
fill_scalar:
    ...
```
means `primitive=FillScalar`

- read `dispatch.enable` 

output:
- `primitive` - Xxx
- `ascend_dispatch` - auto_generate/customize/✖️
- `kbk` - ✅/✖️
- `pyboost` - ✅/✖️


#### Case 1  `dispatch.enable: True` and no `Ascend: XxxAscend` 

auto-generated Ascend path, no handwritten customize needed -> `kbk=✅`, `pyboost=✅`, `ascend_dispatch=auto_generate` 

#### Case 2  `dispatch.enable: True` and `Ascend: XxxAscend` 
customized Ascend path, handwritten implementation required -> `ascend_dispatch=customize`
- hit `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/xxx_aclnn_kernel.cc` or valid registered implementation -> `kbk=✅`
- hit `ops/kernel/ascend/aclnn/pyboost_impl/customize/xxx.cc` -> `pyboost=✅`

#### Case 3  no `dispatch.enable` -> not yet supported aclnn
-> `kbk=✖️`, `pyboost=✖️`, `ascend_dispatch=✖️` 


### 3. Resolve Backward Fields from `primitive`

for each `primitive`, search `mindspore/ccsrc/frontend/expander/grad/` for `REG_BPROP_BUILDER("<primitive>")`

- if hit `REG_BPROP_BUILDER("<primitive>")` -> `backward_defined=✅`
- if no hit `REG_BPROP_BUILDER("<primitive>")` -> `backward_defined=✖️`, `backward_ops=✖️`
- if `backward_defined=✅`, list all backward ops used in the bprop body
- list `ib->Xxx(...)` calls as `backward_ops`
- if the bprop body uses `Emit("XxxGrad", ...)`, list `XxxGrad` in `backward_ops`

output:
- `backward_defined` - ✅/✖️
- `backward_ops` - `Xxx, Yyy` / ✖️

Example:

`primitive=Abs` -> hit `REG_BPROP_BUILDER("Abs")` -> bprop body uses `Mul` and `Sign` -> `backward_defined=✅`, `backward_ops=Mul, Sign`

### Example: `mint.add`

```text
mint.add
- mint_entry: ✅
- api_def: add.yaml

add_scalar_op.yaml
- primitive: AddScalar
- ascend_dispatch: customize
- kbk: ✅
- pyboost: ✅
- backward_defined: ✅
- backward_ops: Real, OutZeros

add_ext_op.yaml
- primitive: AddExt
- ascend_dispatch: auto_generate
- kbk: ✅
- pyboost: ✅
- backward_defined: ✅
- backward_ops: Muls, Cast, OutZeros
```

Display answer as:

```text
mint.add
├── mint_entry: ✅
│   └── from mindspore.ops.functional_overload import add
├── api_def: add.yaml
│   ├── add_scalar_op.yaml
│   │   ├── primitive: AddScalar
│   │   ├── dispatch.enable: True
│   │   ├── Ascend: AddScalarAscend
│   │   ├── ascend_dispatch: customize
│   │   ├── kbk: ✅
│   │   ├── pyboost: ✅
│   │   ├── backward_defined: ✅
│   │   └── backward_ops: Real, OutZeros
│   └── add_ext_op.yaml
│       ├── primitive: AddExt
│       ├── dispatch.enable: True
│       ├── Ascend: not specified
│       ├── ascend_dispatch: auto_generate
│       ├── kbk: ✅
│       ├── pyboost: ✅
│       ├── backward_defined: ✅
│       └── backward_ops: Muls, Cast, OutZeros
└── deprecated/add_method.yaml
    └── ignored
```



