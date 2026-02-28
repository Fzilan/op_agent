# Workflow 8: 测试

## 目标

完成 C++ UT + Python ST，确保功能、精度、动态 shape 全覆盖。

## 输入

- **算子实现**：YAML / Infer / PyBoost / KBK / BPROP
- **PTA 对标实现**：用于 ST 数值对齐

## 输出（两类测试，逐项确认）

> **⚠️ 以下两类测试是 Step 8 的必须产出，每一类都要明确标注状态。**
> 不允许只写其中一类就认为"测试步骤完成"。

| 类型 | 文件位置 | 必须程度 | 状态标注 |
| --- | --- | --- | --- |
| **C++ UT** | `tests/ut/cpp/ops/test_ops_{op_name}.cc` | `[MUST]` 必须新建 | ✅已写 / ❌未写（说明原因） |
| **Python ST** | `tests/st/ops/share/_op_info/op_database.py`（OpInfo 注册） | `[MUST]` 新增 OpInfo 或确认已有 | ✅已注册 / ✅已有（标明算子名） / ❌未注册 |

### 关于"已存在"的判断

在 `op_database.py` 中搜索算子名，**确认 OpInfo 是否已注册且覆盖新算子路径**：
- OpInfo 的 `op` 指向新接口（如 `mint.acos`）且已加入对应 `xxx_op_db` → 确认覆盖
- OpInfo 不存在或 `op` 指向旧接口 → **不算覆盖**，必须新增/更新
- OpInfo 存在但缺少反向 dtype 或动态 shape 配置 → 需要补充字段

---

## 执行步骤

### Step 1：C++ UT（`reference.md` §8.1）—— 必须新建

> agent 可以完全自主完成，不需要设备。**没有理由跳过。**

典型构造：
- 标量：`ShapeVector{}` + `CreateScalar<T>(value)`
- tuple：`ShapeArray{{}}` + `ValuePtrList{...}`
- None：`kMetaTypeNone` + `kNone`
- unknown：`kValueAny`

参照同类算子的已有 C++ UT 文件确认测试宏和参数结构。

### Step 2：Python ST（`reference.md` §8.2）—— 必须注册 OpInfo

> 当前 ST 使用**测试框架 2.0**（`tests/st/ops/share/`），核心操作是在 `op_database.py` 中注册 OpInfo，禁止另外手写独立测试文件。框架原理详见 `reference.md` §8.2。

- 优先先做**形状/类型覆盖**，再补数值验证。
- 严格对齐时按算子特性收紧 `atol/rtol`；`bfloat16` 对比前先升精到 `float32`。
- **必须覆盖**：Pynative、按需 KBK、前向、反向（如有）、动态 shape/rank。

**两种接入方式**：

| 方式 | 适用场景 | 操作 |
| --- | --- | --- |
| **方式 A（通用）** | Unary/Binary/Reduction 等常规算子 | 在 `op_database.py` 添加 OpInfo → 加入对应 `xxx_op_db` → 自动纳入前端参数化用例 |
| **方式 B（特化）** | 需要自定义测试逻辑的算子 | 继承 OpsFactory 写自定义测试套 + 新建前端测试文件 |

**方式 A 操作步骤**（绝大多数算子）：

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

#### other 类算子的 `xxx_reference_inputs_func` 场景覆盖要求

Unary/Binary/Reduction 类算子在 `op_info.py` 中已提供丰富的通用输入生成函数（各种 shape 组合、
广播、非连续、特殊值、极端值等），注册 OpInfo 后自动覆盖。

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

#### 数值验证标准（与 torch 对比时）

- **与 PTA 的对比**放在 Step 3 的精度零偏差验证中完成；本步 ST 的数值基线优先使用 torch CPU 或用户提供的 reference。
- **若存在对应 torch CPU 大算子**：优先直接调用该大算子作为 ST 基线，与 MS 输出做 allclose 或单/双精度 compare。
- **若无对应 torch CPU 大算子、只能用小算子拼接**：
  1. 先按通用数学定义实现 reference，保证形状和梯度结构正确，先把 ST 框架跑通。
  2. 在实现处注明该 reference 为占位实现，并提醒用户向 PTA/验收方索取验证交付时使用的 torch CPU 拼接实现与验收标准。
  3. 拿到 PTA 侧 reference 或标准后，再替换 `op_database.py` 中的参考实现并收紧对比条件。

### Step 3：精度零偏差验证（`reference.md` §14.1，按需）

- 固定随机种子，保存输出为 `.npy`
- `md5sum` 对比 MS/PTA 输出哈希
- Agent 须产出可执行验证脚本，并写明运行环境、运行命令、以及需要用户回传的结果。

### Step 4：显存对齐验证（`reference.md` §14.2，按需）

- MS：`mindspore.runtime.max_memory_allocated()`
- PTA：`torch_npu.npu.max_memory_allocated()`
- 在相同阶段统计

### Step 5：组合场景分层验证（`reference.md` §23.5）

| 阶段 | 验证内容 |
| --- | --- |
| 子算子级 | 每个子算子独立 UT/ST |
| 组合级-中间值 | 临时 dump 中间 tensor 与 PTA 对比 |
| 组合级-最终输出 | 标准 ST 对齐 |
| 反向级 | 反向 ST + 数值梯度检查 |

---

## 需要用户配合的环节

| 环节 | 原因 | 向用户说明 |
| --- | --- | --- |
| Ascend ST 执行 | 需要 Ascend 设备 | "ST 测试需要在 Ascend 设备上运行。请执行我给出的 pytest 命令，并将完整终端输出记录到 Feature 文档 §14 对应表格或指定日志文件，便于我回读判断是否通过。" |
| 精度零偏差验证 | 需同时跑 MS 和 PTA | "请运行我产出的验证脚本，并将 md5 对比结果或输出文件路径记录到 Feature 文档 §14 对应行，或写入我指定的结果文件。" |
| 性能/显存对比 | 需要真实设备 | "请在 Ascend 设备上运行性能脚本，并把耗时和显存数据记录到 Feature 文档 §14 对应表格，或写入我指定的结果文件。" |
| 稳定性 100 次验证 | 耗时较长 | "请在设备上执行 100 次循环脚本，并把通过/失败与简要日志记录到 Feature 文档 §14 或我指定的结果文件。" |

> agent 可以**生成测试脚本和验证命令**，但若无法直接访问 Ascend 设备，必须将脚本和运行指令交给用户执行，**等用户回传结果后再判断是否通过**。
> **禁止**：只写"请跑脚本并回传结果"而不提供脚本路径、运行命令、以及结果记录位置。

---

## 🔒 Step 8 完成前强制检查（不可跳过）

**在标记 Step 8 为完成之前，必须逐项确认以下清单：**

```text
测试产出检查清单：

C++ UT 文件：
  - 文件路径：tests/ut/cpp/ops/test_ops_{op_name}.cc
  - 状态：✅已新建 / ❌未写（原因：___）

Python ST（OpInfo 注册）：
  - 注册文件：tests/st/ops/share/_op_info/op_database.py
  - OpInfo 已注册？ ✅是（算子名：___）/ ❌否（原因：___）
  - 已加入对应 xxx_op_db 列表？ ✅是 / ❌否
  - 前端参数化用例已覆盖？ ✅是（测试文件：___）/ ❌否
  - 若需自定义输入：inputs_func 已编写？ ✅是 / ⏭不需要
  - 🚫 是否新建了独立测试脚本？ 必须为否（如误建需删除并迁移到 OpInfo）

精度零偏差验证脚本（按需）：
  - 文件路径：tests/st/ops/ascend/verify_{op_name}_pta_md5sum.py（或同目录等价命名）
  - 状态：✅已产出 / ⏭不适用（原因：___）
  - 脚本内是否含：固定种子、MS/PTA 分别跑、保存 `.npy/.npz`、md5 对比或输出？ 是/否
```

> 如果 C++ UT 或 Python ST 的状态为 ❌，**必须说明原因并暂停等用户确认后再继续**。
> 不允许静默跳过。

## 成功标准

- [ ] **C++ UT 文件已产出**（Infer 推导覆盖 unknown/None/动态shape）
- [ ] **Python ST OpInfo 已注册且纳入前端参数化用例**（自动覆盖多模式 + 前向精度 + 动态 shape）
- [ ] 稳定性验证：100 次运行无偶现失败（需用户在设备上验证）
- [ ] 覆盖场景：动态 shape / 静态 shape / 非连续 tensor / 空 tensor / 特殊值
- [ ] **用户/场景视角**：典型使用场景和 PTA 对标场景已在样例或输入集里覆盖
- [ ] （精度零偏差）hash 对比通过（按需）
- [ ] （组合场景）分层验证通过（按需）

---

## 下一步

测试完成后，进入 **[Workflow 9: 文档](./09-docs.md)**
