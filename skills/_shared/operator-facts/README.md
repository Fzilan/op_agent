# operator-facts

`operator-facts` 是 ACLNN `Pre` 阶段使用的 MindSpore 侧结构化事实仓。

当前只保留重构后的模型和消费层：

- `ms_entry_identity`
- `ms_unit_identity`
- `ms_entry_unit_edges`
- `ms_unit_graph_edges`
- `ms_entry_bundle`

不再保留旧版：

- `api_identity`
- `ms_coverage`
- `op_bundle`
- 基于 `public_api/op_branch` 的旧 bundle 目录

## 当前目录

```text
operator-facts/
├── bundles/
│   └── entries/                               # MindSpore entry bundles, one public API per file
│       ├── mindspore.Tensor.abs.json          # one single-entry bundle
│       ├── mindspore.Tensor.max.json          # one overload-entry bundle, contains multiple branch targets
│       └── mindspore.Tensor.aminmax.json      # one composite-entry bundle, contains composite components
├── data/
│   ├── ms_entry_identity.jsonl                # MS public entry identity index
│   ├── ms_entry_unit_edges.jsonl              # MS entry -> unit routing index
│   ├── ms_unit_graph_edges.jsonl              # MS composite unit -> child graph index
│   ├── ms_unit_identity.jsonl                 # MS execution unit identity + coverage index
│   └── pta_facts.jsonl                        # PTA API facts + source refs
├── examples/
│   ├── ms_entry_bundle.abs.example.json       # example single-entry bundle
│   ├── ms_entry_bundle.aminmax.example.json   # example composite-entry bundle
│   └── pta_facts.example.json                 # PTA facts example
├── schemas/
│   ├── ms_entry_bundle.schema.json            # entry bundle format, for bundles/entries/*.json
│   ├── ms_entry_identity.schema.json          # entry identity format, for data/ms_entry_identity.jsonl
│   ├── ms_entry_unit_edges.schema.json        # entry -> unit edge format, for data/ms_entry_unit_edges.jsonl
│   ├── ms_unit_graph_edges.schema.json        # composite graph edge format, for data/ms_unit_graph_edges.jsonl
│   ├── ms_unit_identity.schema.json           # unit identity + coverage format, for data/ms_unit_identity.jsonl
│   └── pta_facts.schema.json                  # PTA facts format, for data/pta_facts.jsonl
├── scripts/
│   ├── build_ms_facts.py                      # build all 4 normalized MS facts tables
│   ├── build_entry_bundles.py                 # build entry bundles from the 4 MS facts tables
│   ├── build_operator_facts.py                # end-to-end pipeline: facts -> validate -> bundles -> validate
│   ├── build_ms_entry_identity.py             # build entry identity rows
│   ├── build_ms_entry_unit_edges.py           # build entry -> unit routing rows
│   ├── build_ms_unit_graph_edges.py           # build composite unit graph rows
│   ├── build_ms_unit_identity.py              # build unit identity + coverage rows
│   ├── symbol_resolution.py                   # shared symbol-to-branch / symbol-to-entry resolver
│   ├── unit_coverage_scan.py                  # shared branch coverage scanning logic
│   └── build_pta_facts.py                     # build PTA facts
└── validation/
    ├── validate_ms_facts.py                   # validate the 4 MS facts tables + golden cases
    ├── validate_entry_bundles.py              # validate generated entry bundles against schema
    └── golden/ms_facts.golden.json            # golden regression cases for key APIs
```

## 数据模型

### `ms_entry_identity`

描述 public API 入口身份。

关键字段：

- `public_api`
- `public_surface`
- `public_name`
- `entry_type`
- `source_type`
- `source_path`

### `ms_unit_identity`

描述执行单元身份与覆盖事实。

支持两类 unit：

- `branch`
- `composite`

branch unit 覆盖字段统一为：

- `aclnn`
- `infer`
- `kbk`
- `pyboost`
- `bprop`
- `bprop_units`

### `ms_entry_unit_edges`

描述 `entry -> unit` 路由关系。

关键字段：

- `entry_id`
- `unit_id`
- `edge_type`
- `resolver_type`
- `target_symbol`

### `ms_unit_graph_edges`

描述 composite unit 的内部展开关系。

关键字段：

- `parent_unit_id`
- `child_ref_type`
- `child_ref`
- `condition`
- `via_symbol`

## bundle 设计

当前消费层只保留 `entry bundle`。

### branch bundle 形态

当实际 routed unit 是 `branch` 时，bundle 使用：

- `entry`
- `branches`

`branches[*]` 直接挂 branch unit 和 coverage。

### composite bundle 形态

当实际 routed unit 是 `composite` 时，bundle 使用：

- `entry`
- `composite`

`composite` 只保留：

- `resolver_type`
- `target_symbol`
- `impl_path`
- `components`

注意：

- `entry.entry_type` 只表达入口类型，不直接决定 bundle 顶层结构
- 例如某些 `entry_type = single` 的 Tensor method，实际会路由到一个 composite root，此时 bundle 仍然使用 `composite`

其中：

- `component_type = public_api` 只保留 `public_api`
- `component_type = primitive_symbol` 只保留 `primitive_symbol`
- `component_type = unit` 时：
  - 总是保留 `unit_id`、`unit_name`、`unit_type`
  - 若 `unit_type = branch`，再保留 `op`、`primitive`、`yaml_path`、`coverage`
  - 若 `unit_type = composite`，只保留 `impl_path`

## 从头构建流程

### 一键构建

推荐直接运行：

```bash
python op_agent/skills/_shared/operator-facts/scripts/build_operator_facts.py
```

如果要显式指定 MindSpore 源码根目录：

```bash
python op_agent/skills/_shared/operator-facts/scripts/build_operator_facts.py \
  --ms-root /path/to/mindspore/mindspore
```

这个脚本会按顺序执行：

1. `build_ms_facts.py`
2. `validation/validate_ms_facts.py`
3. `build_entry_bundles.py`
4. `validation/validate_entry_bundles.py`

### 分步构建

#### 1. 构建 4 张基础表

```bash
python op_agent/skills/_shared/operator-facts/scripts/build_ms_facts.py
```

产出：

- `data/ms_entry_identity.jsonl`
- `data/ms_entry_unit_edges.jsonl`
- `data/ms_unit_graph_edges.jsonl`
- `data/ms_unit_identity.jsonl`

#### 2. 校验基础表

```bash
python op_agent/skills/_shared/operator-facts/validation/validate_ms_facts.py
```

校验内容：

- schema 结构
- 主键唯一性
- 表间引用完整性
- golden case

#### 3. 构建 entry bundles

```bash
python op_agent/skills/_shared/operator-facts/scripts/build_entry_bundles.py
```

产出目录：

- `bundles/entries/*.json`

#### 4. 校验 entry bundles

```bash
python op_agent/skills/_shared/operator-facts/validation/validate_entry_bundles.py
```

### PTA facts

PTA facts 仍然独立构建，不参与当前 MindSpore facts/bundle 流水线：

```bash
python op_agent/skills/_shared/operator-facts/scripts/build_pta_facts.py
```

## 设计文档

当前唯一有效的重构设计文档是：

- `operator-facts-refactor-design.md`

它对应当前 4 张 facts 表和 `ms_entry_bundle`。
