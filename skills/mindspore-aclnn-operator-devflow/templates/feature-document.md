# {算子名} 算子开发 Feature 文档

> **说明**：本文档是算子评审和转测交付的**必须产物**。
> Pre-B 阶段初始化 §1-§4、§6、§8；其余章节在对应开发步骤完成后回填。
> 每个章节标注了填写时机：`[Pre-B 阶段]` / `[Step X 回填]` / `[开发完成后]`。

---

## 1. 背景描述 `[Pre-B 阶段]`

{描述算子的背景、来源（如 NSA/DSA 等算法论文）、MindSpore 为何需要此算子}

## 2. 标杆与接口（Benchmark & API） `[Pre-B 阶段]`

- **标杆接口**：`torch_npu.npu_xxx` / `torch.xxx`
- **功能**：{一句话描述}
- **MindSpore 接口**：
  - functional：`mindspore.ops.xxx` / `mindspore.mint.xxx`
  - nn：`mindspore.mint.nn.Xxx`（若需要）
  - Tensor：`Tensor.xxx`（若需要）

## 3. 任务清单（Tasks） `[Pre-B 阶段初始化，开发中更新状态]`

> 标准 13 大类，逐项标注状态。

| 序号 | 任务项 | 任务子项 | 状态（新增/修改/无变更/不涉及） | 备注 |
| ---- | ------ | -------- | ------------------------------ | ---- |
| 1 | 接口基本功能 | Primitive | | |
| | | functional | | |
| | | nn | | |
| | | tensor | | |
| 2 | 后端及数据类型支持 | Ascend | | |
| | | GPU | | |
| | | CPU | | |
| 3 | 支持 vmap | | | |
| 4 | 支持动态 Shape | 动态 Shape | | |
| | | 动态 Rank | | |
| 5 | 支持反向 | bprop 函数 | | |
| | | 复数支持 | | |
| 6 | 补齐资料 | API 映射 | | |
| | | 接口中英文资料 | | |
| 7 | 性能优化 | CPU | | |
| | | GPU | | |
| | | Ascend | | |
| 8 | 功能 | 空 Tensor 支持 | | |
| | | inf/nan 支持 | | |
| | | 0~8 维支持 | | |
| | | 其他功能点 | | |
| 9 | 门禁用例补齐 | UT | | |
| | | ST | | |
| | | TEST_OP | | |
| 10 | 支持 MS Adapter | | | |
| 11 | 自动并行切分 | | | |
| 12 | 混合精度（AMP） | | | |
| 13 | 安全与异常 | 异常用例与报错规范 | | |

## 4. 功能与接口说明 `[Pre-B 阶段]`

### 功能概述

{算子的计算公式、语义、核心行为}

### 对外接口

```python
mindspore.ops.xxx(
    param1: Tensor,      # [shape], dtype: xxx
    param2: int,         # 说明
    ...
) -> Tensor | Tuple[Tensor, ...]
```

### 参数说明

| 参数 | 类型 | 必选/可选 | 默认值 | 说明 |
| ---- | ---- | -------- | ------ | ---- |
| param1 | Tensor | 必选 | — | {描述} |
| ... | | | | |

## 5. YAML 定义（参考） `[Step 1 完成后]`

```yaml
# operator xxx
xxx:
    args:
        # {填入实际 YAML}
    returns:
        # {填入实际 YAML}
    dispatch:
        enable: True
        Ascend: XxxAscend
```

## 6. 约束与类型 `[Pre-B 阶段]`

- **设备**：Ascend（{具体芯片系列}）
- **输入/输出 dtype**：{列出}
- **形状与范围**：{列出各输入的 shape 约束}
- **空 Tensor**：{支持/不支持，说明}

## 7. 执行模式与适配 `[Step 4/5 完成后]`

### Pynative（PyBoost）
- {实现说明}

### Graph（KBK/O0）
- {实现说明}

## 8. 与 PTA 的差异与对齐 `[Pre-B 阶段初始化，开发中补齐]`

- **功能对齐**：{与 PTA 的对齐情况}
- **精度**：{对比策略，0 偏差/rtol-atol}
- **差异**：{列出与 PTA 的差异及原因}

## 9. 动态 Shape/Rank 支持 `[Step 3 完成后]`

- {动态维/动态秩推导策略}
- {编译期未知时的回退方案}

## 10. 异常与校验 `[Step 3/4 完成后]`

### 推导期（Infer）
- {列出推导期校验项}

### 运行期（ACLNN）
- {列出运行期校验项}

## 11. 反向（BPROP） `[Step 6 完成后]`

- {BPROP 注册方式、反向输入输出、梯度处理}
- 若通过自动微分实现则说明"无需显式 bprop"

## 12. 测试方案 `[Step 8 完成后]`

### UT（C++ GeneralInfer）
- {覆盖场景}

### ST（Ascend）
- **功能对比**：{对比策略}
- **场景**：{模式/dtype/特殊输入/异常}

### TEST_OP
- {覆盖说明}

## 13. 代码与文件改动说明 `[开发完成后]`

| 类别 | 文件路径 |
| ---- | -------- |
| YAML | `mindspore/ops/op_def/yaml/xxx_op.yaml` |
| Infer | `mindspore/ops/infer/ops_func_impl/xxx.cc/.h` |
| PyBoost | `mindspore/ops/kernel/.../customize/xxx.cc/.h` |
| KBK | `mindspore/ops/kernel/.../customize/xxx_aclnn_kernel.cc/.h` |
| BPROP | {路径或"不涉及"} |
| API 导出 | `mindspore/ops/api_def/xxx.yaml`、`__init__.py` |
| 文档(EN) | `mindspore/ops/op_def/yaml/doc/xxx_doc.yaml`（`_ext` 风格）或 `api_def/function_doc/`（旧风格） |
| 文档(CN) | `docs/api/api_python/ops/mindspore.ops.xxx.rst` |
| 测试(UT) | `tests/ut/cpp/ops/test_xxx_general_infer.cc` |
| 测试(ST) | `tests/st/ops/ascend/test_xxx.py` |

## 14. 验收报告 `[转测前填写]`

### 基本信息

- **要验收的算子**：`mindspore.ops.xxx`
- **对比标杆版本**：torch X.X + torch_npu（对应版本）
- **对比标杆算子**：`torch_npu.npu_xxx`
- **是否为副作用算子**：否/是
- **报错规范**：MindSpore 日志与错误信息规范

### 资料验证

| 自测内容 | 自测结果 | 备注 |
| -------- | -------- | ---- |
| 新增接口列表 | | |
| 提供典型场景的 UT/ST 用例 | | |
| 接口是否提供中文 RST 并与英文注释对应 | | |
| 接口描述是否详细准确 | | |
| 与 PyTorch 的接口是否一致 | | |
| summary 部分是否有提供公式 | | |
| 属性描述是否完整正确 | | |
| input 描述是否完整正确 | | |
| 输出描述是否完整正确 | | |
| 输出尺寸和 input 是否一样 | | {说明 reduction 等参数对输出 shape 的影响} |
| Raises 项描述完整正确 | | |
| 支持的平台都有填写 | | |
| 资料格式（包括样例格式）检查是否 ok | | |
| 样例是否有提供 | | |
| 样例是否有打印结果 | | |
| 样例执行情况是否 ok | | |
| 算子与 API 能力沙盘是否补齐 | | |

### 功能验证

| 自测内容 | 自测结果 | 备注 |
| -------- | -------- | ---- |
| 默认参数场景是否验证 | | |
| 空 Tensor 输入的正反向是否验证 | | |
| inf 和 nan 是否验证 | | |
| 算子支持数据类型是否与标杆对齐（pytorch npu/gpu/cpu） | | |
| 输入取值范围是否有验证 | | |
| 输入维度是否有覆盖 0D-8D | | |
| 输入支持的 dtype 是否全覆盖 | | |
| 输入是否支持隐式类型转换？ | | |
| 输入是否支持广播 | | |
| 输入之间的约束是否有验证 | | |
| 正向的精度验证是否通过 | | |
| 反向是否支持 | | |
| 反向是否是单算子实现 | | |
| 异常用例是否校验具体报错信息 | | |
| 是否提供报错白名单 | | |
| 是否提供 functional 用例 | | |
| 动态 shape/rank/属性是否都支持 | | |
| 是否关闭退避功能验证 export MS_DISABLE_KERNEL_BACKOFF=1 | | |
| 测试仓接口相关用例是否全部可以 PASS，接口没有遗留问题单 | | |
| 是否支持 bf16 | | |
| 多输入算子的 bprop 函数是否有考虑反向按需求导 | | |
| 算子输出 shape 是否依赖于算子的计算结果 | | |
| 是否支持非连续输入支持情况 | | |
| 是否与 PTA 计算结果 0 偏差（请将 MD5 对比截图贴于右侧） | | |
| 是否会使得运算符或存量 ops 接口调用到新增的原语 | | |
| 是否已支持 amp（混合精度）特性 | | |
| 若多 Tensor 输入，是否支持各 Tensor 数据类型不一致 | | |

### 性能验证

| 自测内容 | 自测结果 | 备注 |
| -------- | -------- | ---- |
| 性能是否验证广播场景 | | |
| 是否考虑到反向显存优化，未用到的输入是否添加到了 SetUnusedInputs 中（串讲时需展示代码） | | |
| 性能测试是否覆盖不同规格的数据（3 种以上），且算子性能不低于友商，允许波动范围 10% | | |
| 显存是否持平 PTA | | |

### 安全编码检视

| 自测内容 | 自测结果 | 备注 |
| -------- | -------- | ---- |
| 指针是否未判空 | | {Infer 中 Primitive/输入输出/GetShape/GetType/GetValue 不需要判空外，其他需判空} |
| 指针是否先使用后校验 | | |
| 数组、指针是否访存越界 | | |
| 是否存在除零 | | |
| 是否存在内存泄露（new/malloc 内存未释放） | | |
| 是否存在异常、错误处理分支未对内存、文件句柄等资源释放 | | |
| 是否存在使用 new 创建对象未声明为 nothrow | | |
| 是否未使用安全函数库进行内存操作 | | |
| 是否存在数据类型转换导致数值移除（上溢或下溢） | | |
| 是否存在冗余代码（冗余校验、不可达代码等） | | |
| 是否存在暴露敏感信息 | | |
| 是否使用了弱随机数生成器 | | |
