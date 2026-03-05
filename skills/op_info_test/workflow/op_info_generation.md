# 算子st用例生成

## 目标

新增算子Python ST，确保功能、精度、动态 shape 全覆盖。

**注意事项**：库上可能已有相似算子用例，但覆盖场景不全。参考类似算子用例实现时不应该以其覆盖场景作为目标覆盖场景。需要严格遵循各场景对生成用例的覆盖范围要求。

## 输入

- **接口名**: 使用接口名收集相关信息，如算子接口定义yaml, 算子接口文档
- **torch对标接口**：测试标杆

## 输出（两类测试，逐项确认）

> **⚠️ 以下是必须产出**

| 类型 | 文件位置 |
| --- | --- | --- | --- |
| **Python ST** | `tests/st/ops/share/_op_info/op_database.py`（OpInfo 注册） |

---

## 执行步骤

> 当前 ST 使用**测试框架 2.0**（`tests/st/ops/share/`），核心操作是在 `op_database.py` 中注册 OpInfo，禁止另外手写独立测试文件。框架原理详见 `reference.md` §8.2。

**两种场景**：

| 适用场景 | 操作 |
| --- | --- |
| Unary/Binary/Reduction 等常规算子 | 在 `op_database.py` 添加 OpInfo → 加入对应 `xxx_op_db` → 自动纳入前端参数化用例 |
| 需要自定义测试逻辑的算子 | 继承 OpsFactory 写自定义测试套 + 新建前端测试文件 |

### Unary/Binary/Reduction 等常规算子 

Unary/Binary/Reduction 类算子在 `op_info.py` 中已提供丰富的通用输入生成函数（各种 shape 组合、
广播、非连续、特殊值、极端值等），注册 OpInfo 后自动覆盖。

1. **确定算子类别**：Unary → `UnaryOpInfo` / Binary → `BinaryOpInfo` / Reduction → `ReductionOpInfo` / 其他 → `OpInfo`
2. **在 `op_database.py` 添加 OpInfo 实例**：配置 `name`、`op`、`ref`、`dtypes_support`（以及 `dtypes_grad`、`dtypes_dynamic` 等）
3. **将算子名加入对应 `xxx_op_db` 列表**（如 `binary_op_db`、`unary_op_db`）
4. **如需自定义输入场景**：编写 `op_basic_reference_inputs_func` / `op_extra_reference_inputs_func`，返回 `OpSampleInput` 列表
5. **判断是否需要加入 `xxx_op_kbk_db` 列表**（见下方约束）
6. **验证覆盖**：确认前端测试文件（如 `test_binary_ops.py`）的参数化用例已包含新算子

> **关于 KBK 列表（`xxx_op_kbk_db`）的添加约束**：
>
> KBK 场景耗时较长，不需要每个算子都加入。仅在以下情况下将算子加入对应的 `xxx_op_kbk_db`（如 `binary_op_kbk_db`、`unary_op_kbk_db`、`reduction_op_kbk_db` 等），使前端测试文件跑 KBK 前向/反向/动态 shape 用例：
>
> - 算子包含**较复杂的动态 shape 推导逻辑**（如输出 shape 依赖输入值、多分支推导）
> - 算子采用**组合实现**（PyBoost/KBK 中串联多个 ACLNN 调用）
> - 算子包含**前端接口重载**（如同时支持 Tensor-Tensor 和 Tensor-Scalar 两种调用形态）
>
>
> **不需要添加的情况**：
> - 简单直通算子（单 ACLNN、无参数预处理），pynative 已充分覆盖
> - KBK 列表中**已有同类型/同实现模式的算子**——例如 `unary_op_kbk_db` 已有 `mint.tanh`，则 `mint.cosh` 等同类三角函数无需重复添加

### 需要自定义测试逻辑的算子

**other 类算子**（加入 `other_op_db`）需要在 `op_database.py` 中**自行编写** `op_basic_reference_inputs_func`和`op_extra_reference_inputs_func`等函数，
且必须覆盖 `checklists.md` §6 的场景要求（不能只写 2-3 个简单 case）：

| 必覆盖场景 | 编写方式 | 示例 |
| --- | --- | --- |
| **多种 shape**（含 0D scalar、1D、2D-3D 中间维、高维） | 多个 yield，不同 shape | `make_arg(())`, `make_arg((S,))`, `make_arg((S,M,S))` |
| **空 tensor**（某维为 0） | shape 中含 0 | `make_arg((0, S))`, `make_arg((S, 0, M))` |
| **非连续 tensor** | `discontiguous=True` 参数 | `make_tensor(shape, discontiguous=True)` |
| **边界参数值** | 覆盖参数的极端/边界 | `dim=0`, `dim=-1`, `dim=最后一维`; `p=1`, `p=2`, `p=inf` |
| **大 tensor** | 至少一个较大 shape | `make_arg((LARGE_DIM_SIZE, M))` |

编写参考：`op_info.py` 中 `basic_reference_inputs_binary_op_common_func` 和
`_generate_binary_op_broadcasting_and_discontiguous_tensor_inputs_func` 的写法模式。

如果算子支持 `op_extra_reference_inputs_func`（额外精度场景）或 `op_dynamic_inputs_func`
（动态 shape/rank），也应参照 `op_info.py` 中的同类写法编写。

### 精度零偏差验证（`reference.md` §14.1，按需）

- 固定随机种子，保存输出为 `.npy`
- `md5sum` 对比 MS/PTA 输出哈希

### 显存对齐验证（`reference.md` §14.2，按需）

- MS：`mindspore.runtime.max_memory_allocated()`
- PTA：`torch_npu.npu.max_memory_allocated()`
- 在相同阶段统计

---

## 🔒 Step 8 完成前强制检查（不可跳过）

**在标记 Step 8 为完成之前，必须逐项确认以下清单：**

```text
测试产出检查清单：

Python ST（OpInfo 注册）：
  - 注册文件：tests/st/ops/share/_op_info/op_database.py
  - OpInfo 已注册？ ✅是（算子名：___）/ ❌否（原因：___）
  - 已加入对应 xxx_op_db 列表？ ✅是 / ❌否
  - 前端参数化用例已覆盖？ ✅是（测试文件：___）/ ❌否
  - 若需自定义输入：inputs_func 已编写？ ✅是 / ⏭不需要
  - 🚫 是否新建了独立测试脚本？ 必须为否（如误建需删除并迁移到 OpInfo）
```

> 如果 Python ST 的状态为 ❌，**必须说明原因并暂停等用户确认后再继续**。
> 不允许静默跳过。

## 成功标准

- [ ] **Python ST OpInfo 已注册且纳入前端参数化用例**（自动覆盖多模式 + 前向精度 + 动态 shape）
- [ ] 覆盖场景：动态 shape / 静态 shape / 非连续 tensor / 空 tensor / 特殊值