# operator-facts

`operator-facts` 是 ACLNN `Pre` 阶段的第一版蒸馏库。

当前阶段只产出两张索引：

- `api_identity`
- `ms_coverage`
- `op_bundle`
- `pta_facts` schema（已定义，抽取脚本待实现）

## 当前主键约定

### `api_identity`

- 主键：`public_api + op_branch`
- `public_api` 是全限定公共接口，例如 `mindspore.mint.max`
- `op_branch` 是 `api_def` 或 direct `op_def` 落到的分支文件名，例如 `max_dim_op.yaml`

### `ms_coverage`

- 主键：`op + primitive`
- `op` 是 `op_def` 顶层算子名，例如 `max_dim`
- `primitive` 是该分支对应的 Primitive 名，例如 `MaxDim`

## `aclnn` 字段约定

- 字段名统一叫 `aclnn`
- 单值和多值都用 JSON 数组字符串保存
- 例如：`["aclnnSoftmax"]`
- 例如：`["aclnnSoftmax", "aclnnCast", "aclnnReduceSum"]`

当前阶段不拆附表，不做 `branch -> aclnn` 正规化。

## 输出路径

生成结果写到：

- `operator-facts/data/api_identity.jsonl`
- `operator-facts/data/api_identity.csv`
- `operator-facts/data/ms_coverage.jsonl`
- `operator-facts/data/ms_coverage.csv`
- `operator-facts/data/pta_facts.jsonl`
- `operator-facts/data/pta_facts.csv`
- `operator-facts/data/op_bundles.jsonl`
- `operator-facts/bundles/<public_api>/<op_branch>.json`

PTA 最小 schema 和样例放在：

- `operator-facts/schemas/pta_facts.schema.json`
- `operator-facts/examples/pta_facts.example.json`

## 使用方式

要求：

- Python 3.9+
- `--ms-root` 指向 MindSpore 仓库里的源码根目录，也就是包含 `ops/`、`python/`、`ccsrc/` 的那个 `mindspore/` 目录

### 一键重建 MS 部分

如果工作区根目录下有 `regen_operator_facts.sh`，推荐直接使用：

```bash
./regen_operator_facts.sh
```

也可以显式指定 MindSpore 源码根目录：

```bash
./regen_operator_facts.sh /path/to/mindspore/mindspore
```

这个脚本会：

- 清理 `bundles/` 和 MS 相关索引
- 保留 `data/pta_facts.jsonl` / `data/pta_facts.csv`
- 重建 `api_identity`、`ms_coverage`、`op_bundle`

### 直接运行 phase1

```bash
python operator-facts/scripts/build_phase1.py --ms-root /path/to/mindspore/mindspore
```

`build_phase1.py` 会把 `--ms-root` 透传给各个子脚本。

也可以指定输出目录：

```bash
python operator-facts/scripts/build_phase1.py \
  --ms-root /path/to/mindspore/mindspore \
  --out-dir operator-facts/data \
  --bundle-root operator-facts/bundles
```

### 分步重建

```bash
python operator-facts/scripts/build_api_identity.py --ms-root /path/to/mindspore/mindspore
python operator-facts/scripts/build_ms_coverage.py --ms-root /path/to/mindspore/mindspore
python operator-facts/scripts/build_bundles.py
python operator-facts/scripts/build_pta_facts.py
```

## 当前覆盖范围

### `api_identity`

第一版覆盖：

- `mindspore.ops.*`
- `mindspore.Tensor.*`
- `mindspore.mint.*`

其中：

- `ops/Tensor` 主要来自 `api_def`
- `mint` 主要来自 `mint/__init__.py` 的实际导出关系
- wrapper 解析采用“直接符号 -> `api_def` / `op_def` -> 一层到数层 return 调用追踪”的轻量策略

### `ms_coverage`

第一版覆盖：

- `dispatch`
- `aclnn`
- `infer`
- `pyboost`
- `kbk`
- `bprop`
- `ut`
- `st`
- `docs_cn`
- `docs_en`

其中 `dashboard` 只作为扫描经验参考，不作为 source-of-truth。
Ascend 侧判断优先遵循 `api-knowledge/backend-lens-ascend.md` 的规则：

- 先看 `op_yaml.dispatch`
- 再看 `aclnn_config.yaml`
- customize 走源码取证

其中：

- `dispatch.enable: True` 且没有 `Ascend: XxxAscend`，记为 `dispatch_kind = auto_generate`
- 这类 branch 视为 ACLNN 已接入
- `aclnn_source` 只使用：
  - `aclnn_config`
  - `auto_generate_source`
  - `customize_source`
- `auto_generate` 场景下，`pyboost` 和 `kbk` 直接记为 `true`
- `pyboost_evidence` 和 `kbk_evidence` 在 `auto_generate` 场景允许为空

### `op_bundle`

第一版 `bundle` 只聚合 `api_identity + ms_coverage`，不包含：

- PTA
- callchain
- decision

当前字段分成：

- `identity`
- `resolver`
- `coverage`
- `evidence`
- `refs`

### `pta_facts`

`pta_facts` 是 PTA 侧的最小事实包 schema，目标不是替代 `op-plugin` 源码，而是让 `bundle` 后续可以直接回答：

- 走 `auto` 还是 `customize`
- 是否需要进入组合场景 `Pre-C`
- 是否存在 backward
- Step 1 的 YAML 参数和返回结构如何起草
- 写代码时应该打开哪些 PTA 文件和锚点

当前字段收缩为：

- `pta_key`
- `pta_api`
- `overload_signature`
- `params`
- `returns`
- `forward_aclnn`
- `backward_exists`
- `backward_aclnn`
- `composite`
- `preprocess_needed`
- `custom_output_needed`
- `refs`

其中：

- `preprocess_needed` 表示 PTA 在调用 ACLNN 前是否存在需要手写代码处理的参数逻辑
- `custom_output_needed` 表示输出 tensor / 中间 tensor 是否需要手工构造
- `refs` 保存 PTA 锚点，写代码阶段只需要定点打开这些文件，不再全仓分析

## 已知限制

- `api_identity` 目前没有覆盖所有复杂 wrapper 的深层控制流。
- `ms_coverage` 当前不区分多个 `aclnn` 的角色，只做 branch 级摘要。
- `op_bundle` 目前只是 phase-1 最小事实包，还没有 PTA 和组合调用链信息。
- 复杂组合算子的调用链还未展开到单独附表。
