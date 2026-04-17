# operator-facts MindSpore Facts 重构设计

## 1. 目标

本文定义 `operator-facts` 中 MindSpore 侧事实模型的长期重构方案。

目标是把当前偏平面的模型：

- `api_identity`
- `ms_coverage`
- `bundles/<public_api>/<op_branch>.json`

重构为一套更适合图结构和 precheck 消费的模型，使其能够稳定表达：

- 公共 API 路由关系
- overload 分支
- Tensor 老入口
- Python composite wrapper
- branch 级 Ascend / ACLNN 覆盖状态
- composite 级 direct coverage 与 leaf 聚合 coverage

这份设计是为 ACLNN `Pre-A / Pre-B / Pre-C` 服务的，不是为通用知识检索设计的。

## 2. 当前问题

当前模型只在下面这种场景下表现稳定：

- 一个 public API 最终可以唯一规约到一个 `op_branch`

但在以下场景中会失真：

- 一个 API 对应多个 branch，例如 `mindspore.Tensor.max`
- 一个 API 从 `common/tensor.py` 或 `tensor_operator_registry` 进入
- 一个 API 实际是 Python composite wrapper，例如 `mindspore.Tensor.aminmax`
- 某些执行单元没有直接暴露 public API，但实际是有效的覆盖分析对象，例如 `argmin_with_value_op.yaml`

根本问题在于：

- 当前 `api_identity` 把“public entry identity”和“execution unit identity”混在了一起
- 当前 `ms_coverage` 只适合描述 branch 级覆盖，不适合描述 composite 行为

## 3. 设计原则

### 3.1 分离 Entry 与 Execution Unit

用户视角的对象是 public API entry。
precheck 视角真正关心的对象是 execution unit。

这两者必须拆开建模。

### 3.2 Unit 应可被多个 Entry 复用

不同 public entry 可能指向同一个 execution unit。
例如 `mindspore.ops.xxx` 与 `mindspore.Tensor.xxx` 不应该重复生成两个相同的 composite implementation unit。

### 3.3 Coverage 应附着在 Unit 上

coverage 表达的是“这个 execution unit 在 Ascend 上是否已具备支持能力”，而不是“这个 public API 是否存在”。

因此 coverage 应属于 `unit`，而不是 `entry`。

### 3.4 Composite 必须是一等公民

composite wrapper 不是例外情况，而是必须被稳定表达的对象。
因此应把 composite root 作为一等 execution unit，并允许它拥有内部 graph edges。

## 4. 命名决策

当前确认的数据模型名称为：

- `ms_entry_identity`
- `ms_unit_identity`
- `ms_entry_unit_edges`
- `ms_unit_graph_edges`

说明：

- `entry` 表示用户可见的公共 API 入口，例如 `mindspore.Tensor.add`
- `unit` 表示 precheck 实际消费的执行单元，例如一个 `op_branch`，或一个 composite root implementation
- `ms_unit_identity` 虽然会承载 coverage 字段，但为了与现有仓内命名风格保持一致，仍保留 `identity` 后缀

### 4.1 为什么继续使用 `entry` / `unit`

备选命名包括：

- `surface` / `exec_unit`
- `public_api` / `execution_unit`
- `endpoint` / `implementation`

推荐继续使用 `entry` / `unit`。

原因：

- `entry -> unit` 很适合表达路由关系
- `unit -> unit` 很适合表达 composite 内部展开关系
- `surface` 容易与已有字段 `public_surface` 混淆
- `implementation` 更偏源码实现视角，不适合作为 branch unit 的统一名字
- `execution_unit` 虽然准确，但作为文件名、schema 名和字段前缀过长

后续文档里应统一解释：

- `entry = public API entry`
- `unit = execution unit`

## 5. 数据模型

### 5.1 `ms_entry_identity`

主键：

- `entry_id = public_api`

职责：

- 只表达 public entry 身份
- 不要求 `op_branch`
- 不承载 coverage

建议核心字段：

- `entry_id`
- `public_api`
- `public_surface`
- `public_name`
- `entry_type`
- `source_type`
- `source_path`

建议 `entry_type` 枚举：

- `single`
- `overload`
- `composite`
- `unresolved`

示例：

- `mindspore.Tensor.abs` -> `single`
- `mindspore.Tensor.max` -> `overload`
- `mindspore.Tensor.aminmax` -> `composite`

### 5.2 `ms_unit_identity`

主键：

- `unit_id`

职责：

- 表达真正的 execution unit
- 同时承载 identity 字段与 coverage 字段
- 取代当前独立存在的 `ms_coverage`

建议 `unit_type` 枚举：

- `branch`
- `composite`
- `compat`

推荐 canonical key 形式：

- branch：`branch::add_ext_op.yaml::AddExt`
- composite：`composite::ops/function/array_func.py::aminmax`
- compat：`compat::ops/tensor_method.py::deprecated_tensor_add`

不建议使用 `public_api` 作为 composite key，否则 `mindspore.ops.aminmax` 与 `mindspore.Tensor.aminmax` 会重复生成 unit。

#### Branch Unit 字段

identity 字段：

- `unit_id`
- `unit_type`
- `unit_name`
- `display_id`
- `op`
- `primitive`
- `yaml_path`

coverage 字段：

- `dispatch_enable`
- `dispatch_type`
- `dispatch_ascend`
- `aclnn`
- `infer`
- `pyboost`
- `kbk`
- `bprop`
- `bprop_units`

#### Composite Unit 字段

identity 字段：

- `unit_id`
- `unit_type`
- `unit_name`
- `display_id`
- `impl_path`
- `impl_symbol`

coverage 字段分为两组语义：

- `direct_*`：这个 composite 自己是否直接具备 infer / kernel / aclnn
- `leaf_units`：通过 `ms_unit_graph_edges` 归一化后的可达 leaf execution units 摘要，每个 leaf 单独记录 `aclnn / infer / pyboost / kbk / bprop / bprop_units`

这能让 precheck 明确区分：

- 这个 composite 自己已经直接接了 ACLNN
- 这个 composite 自己没接 ACLNN，但其内部叶子单元已经复用了 ACLNN

#### Compat Unit 字段

identity 字段：

- `unit_id`
- `unit_type`
- `unit_name`
- `display_id`
- `impl_path`
- `impl_symbol`
- `compat_source`

第一阶段中 `compat unit` 是可选的。
如果实现成本较高，第一版可以先不单独建 `compat unit`，而是把兼容路径信息先保留在 `ms_entry_unit_edges` 上。

### 5.3 `ms_entry_unit_edges`

主键：

- `edge_id = entry_id::edge_type::unit_name::dispatch_order`

职责：

- 表达 public entry 如何路由到一个或多个 execution unit
- 只表达路由关系，不表达 composite 内部展开

建议字段：

- `edge_id`
- `entry_id`
- `unit_id`
- `edge_type`
- `dispatch_order`
- `match_condition`
- `resolver_type`
- `resolver_path`
- `target_symbol`

建议 `edge_type` 枚举：

- `direct`
- `overload`
- `composite`

建议 `resolver_type` 枚举：

- `api_def`
- `py_method`
- `alias`
- `tensor_registry`
- `wrapper`
- `deprecated`
- `unknown`

示例：

- `mindspore.Tensor.abs -> branch::abs_op.yaml::Abs`
- `mindspore.Tensor.max -> branch::max_op.yaml::Max`
- `mindspore.Tensor.max -> branch::max_dim_op.yaml::MaxDim`
- `mindspore.Tensor.aminmax -> composite::ops/function/array_func.py::aminmax`

### 5.4 `ms_unit_graph_edges`

主键：

- `graph_edge_id`

职责：

- 表达 composite unit 的内部展开结构
- 只服务于 `unit_type = composite`

建议字段：

- `graph_edge_id`
- `parent_unit_id`
- `child_ref_type`
- `child_ref`
- `condition`
- `call_order`
- `via_symbol`
- `via_path`

建议 `child_ref_type` 枚举：

- `unit`
- `public_api`
- `primitive_symbol`

规则：

- 优先把 leaf 规约到已有的 `unit_id`
- 只有无法唯一规约为已知 unit 时，才保留 `public_api`
- `primitive_symbol` 仅作为最后兜底

## 6. 具体例子

### 6.1 `mindspore.Tensor.abs`

`ms_entry_identity`

- `entry_id = mindspore.Tensor.abs`
- `entry_type = single`

`ms_entry_unit_edges`

- `mindspore.Tensor.abs -> branch::abs_op.yaml::Abs`

`ms_unit_identity`

- 一个 branch unit，`yaml_path = mindspore/ops/op_def/yaml/abs_op.yaml`
- 携带常规 branch coverage 字段

`ms_unit_graph_edges`

- 无

### 6.2 `mindspore.Tensor.max`

`ms_entry_identity`

- `entry_id = mindspore.Tensor.max`
- `entry_type = overload`

`ms_entry_unit_edges`

- `mindspore.Tensor.max -> branch::max_op.yaml::Max`
- `mindspore.Tensor.max -> branch::max_dim_op.yaml::MaxDim`
- `mindspore.Tensor.max -> branch::maximum_op.yaml::Maximum`

`ms_unit_identity`

- 三个 branch units

`ms_unit_graph_edges`

- 无

### 6.3 `mindspore.Tensor.aminmax`

MindSpore 入口：

- `common/tensor.py::Tensor.aminmax`

MindSpore root implementation：

- `ops/function/array_func.py::aminmax`

`ms_entry_identity`

- `entry_id = mindspore.Tensor.aminmax`
- `entry_type = composite`

`ms_entry_unit_edges`

- `mindspore.Tensor.aminmax -> composite::ops/function/array_func.py::aminmax`

`ms_unit_identity`

- 一个 composite unit，root 为 `array_func.py::aminmax`

`ms_unit_graph_edges`

- 当 `axis is not None`：
  - `-> branch::argmin_with_value_op.yaml::ArgMinWithValue`
  - `-> branch::argmax_with_value_op.yaml::ArgMaxWithValue`
- 当 `axis is None`：
  - `-> public_api mindspore.ops.min`
  - `-> public_api mindspore.ops.max`
  - 当 `keepdims is True` 时，`-> branch::reshape_op.yaml::Reshape`

这样 precheck 可以明确区分：

- MindSpore 侧：这是 composite wrapper
- PTA 侧：这是 direct `aminmax` 前向接口，并且是 direct ACLNN 执行路径

## 7. 构建流程

### Step 1：构建 `ms_unit_identity`

目标：

- 先构建所有 branch units
- 在一张表中同时写入 identity 字段与 coverage 字段

输入：

- `ops/op_def/yaml/*_op.yaml`
- `python/mindspore/ops_generate/pyboost/aclnn_config.yaml`
- 当前 coverage 扫描逻辑已使用的 infer / kernel / bprop 证据路径

实现建议：

- 复用当前 `build_op_catalog()` 的 branch 发现能力
- 复用当前 coverage 扫描逻辑，并将其沉淀到独立模块 `unit_coverage_scan.py`
- 主键从 `op::primitive` 改为 `unit_id`
- `argmin_with_value_op.yaml`、`argmax_with_value_op.yaml` 这类内部 unit 必须保留

### Step 2：构建 `ms_entry_identity`

目标：

- 归一化所有 public API entry

输入：

- `ops/api_def/*.yaml`
- `python/mindspore/mint/__init__.py`
- `python/mindspore/common/tensor.py`
- `python/mindspore/ops/tensor_method.py`
- `python/mindspore/ops/functional.py`

entry 来源包括：

- `api_def` entry
- `mint` export entry
- `Tensor` public method entry
- `py_method` bridge entry
- registry 反解 entry

### Step 3：构建 `ms_entry_unit_edges`

目标：

- 把 public entry 路由到 execution units

规则：

- `api_def` 直接落单 branch 的，产 `direct`
- `api_def` 落多个候选 branch 的，产 `overload`
- `common/tensor.py` 经 registry 解析后直接落 branch 的，也产 `direct`
- 路由到 Python composite root 的，产 `composite`
- deprecated 旧 Tensor entry 不单独引入 edge type，而是通过 `resolver_type = deprecated` 表达

副作用：

- 同时收集新发现的 composite root units，用于下一步补充 `ms_unit_identity`

### Step 4：回填 Composite `ms_unit_identity`

目标：

- 为 Step 3 中发现的 composite roots 创建对应的 composite units

规则：

- 增加 `unit_type = composite`
- 填充 `impl_path`、`impl_symbol`、`unit_name`、`display_id`
- 初始化 `direct_*` 与 `leaf_units` 字段

### Step 5：构建 `ms_unit_graph_edges`

目标：

- 只展开 composite units

规则：

- 记录控制流条件与调用顺序
- 优先把叶子解析到已有 `unit_id`
- 只有在无法规约为 unit 时，才退回 `public_api` 或 `primitive_symbol`

### Step 6：回填 Composite Coverage 到 `ms_unit_identity`

目标：

- 完成 composite units 的 coverage 语义

规则：

- `direct_*` = composite 自己直接拥有的支持能力
- `leaf_units` = 从可达 leaf units 归一化得到的逐叶子 coverage 摘要，而不是只记录第一跳 child

### Step 7：构建 Bundles

推荐输出：

- `entry bundle` 作为 builder 主输入
- `unit bundle` 作为复用缓存

当前 `entry bundle` 收敛为面向消费层的轻量视图：

- 顶层固定保留：`bundle_id`、`bundle_type`、`schema_version`、`generated_at`、`entry`
- 当实际 routed unit 为 `branch` 时，bundle 使用 `branches`
- 当实际 routed unit 为 `composite` 时，bundle 使用 `composite`
- 不再额外保留 `root/child`、`implementations`、`units map`、`graphs map`

说明：

- `entry.entry_type` 只表达入口类型，不直接决定 bundle 顶层形态
- 某些 `entry_type = single/overload` 的 Tensor method 可能会直接路由到 composite root，此时 bundle 仍然使用 `composite`

`branches` 形态：

- 每个元素对应一个 direct / overload branch target
- 字段收敛为：`route_type`、`resolver_type`、`target_symbol`、`dispatch_order`、`unit_id`、`unit_name`、`op`、`primitive`、`yaml_path`、`coverage`
- `coverage` 只保留：`aclnn`、`infer`、`kbk`、`pyboost`、`bprop`、`bprop_units`

`composite` 形态：

- 顶层字段收敛为：`resolver_type`、`target_symbol`、`impl_path`、`components`
- `components` 只表达 composite 实现直接涉及到的节点
- `component_type = public_api` 时保留：`public_api`、`condition`、`via_symbol`
- `component_type = primitive_symbol` 时保留：`primitive_symbol`、`condition`、`via_symbol`
- `component_type = unit` 时始终保留：`unit_id`、`unit_name`、`unit_type`、`condition`、`via_symbol`
- 当 `component_type = unit` 且 `unit_type = branch` 时，额外保留：`op`、`primitive`、`yaml_path`、`coverage`
- 当 `component_type = unit` 且 `unit_type = composite` 时，额外保留：`impl_path`

参考样例：

- `examples/ms_entry_bundle.abs.example.json`
- `examples/ms_entry_bundle.aminmax.example.json`

## 8. 推荐目录结构

```text
operator-facts/
├── bundles/
│   ├── entries/
│   │   └── mindspore.Tensor.aminmax.json
│   └── units/
│       ├── branch--argmax_with_value_op.yaml--ArgMaxWithValue.json
│       └── composite--mindspore_python_mindspore_ops_function_array_func_py--aminmax.json
├── data/
│   ├── ms_entry_identity.jsonl
│   ├── ms_unit_identity.jsonl
│   ├── ms_entry_unit_edges.jsonl
│   ├── ms_unit_graph_edges.jsonl
│   ├── pta_facts.jsonl
│   ├── legacy_api_identity.jsonl
│   └── legacy_ms_coverage.jsonl
└── schemas/
│   ├── ms_entry_identity.schema.json
│   ├── ms_unit_identity.schema.json
│   ├── ms_entry_unit_edges.schema.json
│   ├── ms_unit_graph_edges.schema.json
│   ├── ms_entry_bundle.schema.json
│   └── ms_unit_bundle.schema.json
```

## 9. 兼容迁移策略

旧输出不应立即删除。

建议迁移计划如下：

### Phase 1

- 保留当前 `api_identity` 与 `ms_coverage`
- 并行生成新索引
- 在可能的情况下，由新模型投影生成旧输出

### Phase 2

- bundle 生成切到 entry-centered bundles
- 旧的 `<public_api>/<op_branch>.json` 仅作为兼容产物保留

### Phase 3

- builder 不再直接消费 legacy flat indexes
- legacy indexes 只保留用于回归比对和人工检查

## 10. 脚本重构建议

建议拆分成以下脚本：

- `build_ms_facts.py`
- `build_ms_unit_identity.py`
- `build_ms_entry_identity.py`
- `build_ms_entry_unit_edges.py`
- `build_ms_unit_graph_edges.py`
- `validation/validate_ms_facts.py`
- `build_entry_bundles.py`
- `build_unit_bundles.py`

其中：

- `build_ms_facts.py` 作为统一入口，按依赖顺序串起 4 张基础表的生成
- 其余 4 个 `build_ms_*` 脚本仍然保留，便于单表调试和局部回归
- `validation/validate_ms_facts.py` 负责基础 schema 校验、引用完整性校验和 golden case 校验
- `validation/validate_entry_bundles.py` 负责 `ms_entry_bundle` schema 校验
- `scripts/build_operator_facts.py` 负责串联 `build_ms_facts -> validate_ms_facts -> build_entry_bundles -> validate_entry_bundles`

建议同时维护一份最小 golden 数据：

- `validation/golden/ms_facts.golden.json`
- 第一版覆盖 `mindspore.Tensor.abs`、`mindspore.mint.gather`、`mindspore.Tensor.max`、`mindspore.Tensor.aminmax`

旧版 `build_api_identity.py` 已被移除。
branch-centered 的旧模型不再作为当前实现基础，避免继续污染 composite 场景。

## 11. 待定问题

### 11.1 第一版是否必须引入 `compat unit`

当前决策：

- 第一版不单独引入 `compat unit`
- compat 语义先保留在 `ms_entry_unit_edges`
- 通过 `resolver_type = deprecated` 表达 deprecated / old Tensor path
- 当 precheck 明确需要 compat callchain 推理时，再引入 `compat unit`

### 11.2 `ms_unit_identity` 应该平铺字段还是分节

建议：

- JSONL 里保持平铺字段，便于脚本扫描和 `rg`
- 在 bundle 输出里再组织成 `identity`、`coverage`、`aggregation` 等结构块

### 11.3 是否需要增加 `display_id`

建议：

- 需要
- 机器 join / 去重 使用长 canonical `unit_id`
- 报告、日志、调试界面使用短 `display_id`，例如 `composite::aminmax`

## 12. 最终建议

长期稳定的模型建议固定为：

- `ms_entry_identity`
- `ms_unit_identity`
- `ms_entry_unit_edges`
- `ms_unit_graph_edges`

并遵守以下约束：

- `entry` 只建模 public API 身份
- `unit` 建模 execution identity 与 coverage
- `entry -> unit` edges 建模路由
- `unit -> child` edges 建模 composite 展开

这是当前支持以下场景所需的最小稳定结构：

- 单 branch API
- overload API
- Tensor legacy path
- Python composite wrapper
- branch-level coverage
- composite-level direct coverage 与 leaf coverage

同时避免重新退回当前这种平面且有信息损失的表示方式。
