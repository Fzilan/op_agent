# Workflow 8: 测试

## 目标

完成 C++ UT + Python ST（+ 可选 Python UT），确保功能、精度、动态 shape 全覆盖。

## 输入

- **算子实现**：YAML / Infer / PyBoost / KBK / BPROP
- **PTA 对标实现**：用于 ST 数值对齐

## 输出（三类测试，逐项确认）

> **⚠️ 以下三类测试文件是 Step 8 的必须产出，每一类都要明确标注状态。**
> 不允许只写其中一类就认为"测试步骤完成"。

| 类型 | 文件位置 | 必须程度 | 状态标注 |
| --- | --- | --- | --- |
| **C++ UT** | `tests/ut/cpp/ops/test_ops_{op_name}.cc` | `[MUST]` 必须新建 | ✅已写 / ❌未写（说明原因） |
| **Python ST** | `tests/st/ops/ascend/test_{op_name}.py` | `[MUST]` 新建或确认已有 | ✅已写 / ✅已有（标明路径） / ❌未写 |
| **Python UT** | `tests/ut/python/ops/test_{op_name}.py` | `[SHOULD]` 推荐 | ✅已写 / ⏭跳过（说明原因） |

### 关于"已存在"的判断

搜到已有测试文件时，**必须确认它是否覆盖新算子路径**：
- 已有测试调用的是新接口（如 `mint.acos` → `acos_ext`）→ 确认覆盖，标注路径
- 已有测试只调用旧接口（如 `ops.acos` → 旧 `ACos`）→ **不算覆盖**，必须新建
- 已有测试覆盖不完整（如只测了前向没测反向）→ 需要补充

---

## 执行步骤

### Step 1：C++ UT（`reference.md` §8.2）—— 必须新建

> agent 可以完全自主完成，不需要设备。**没有理由跳过。**

典型构造：
- 标量：`ShapeVector{}` + `CreateScalar<T>(value)`
- tuple：`ShapeArray{{}}` + `ValuePtrList{...}`
- None：`kMetaTypeNone` + `kNone`
- unknown：`kValueAny`

参照同类算子的已有 C++ UT 文件确认测试宏和参数结构。

### Step 2：Python ST（`reference.md` §8.3）—— 必须新建或确认已有

> **这是最容易遗漏的一类。** agent 必须生成完整的 ST 测试文件（即使无法在设备上运行）。

- 优先"形状/类型"再比数值
- 严格对齐时 `atol/rtol=0`（按算子特性）
- 避免引入额外算子导致误差累积
- bfloat16 比较前升精到 float32
- **必须覆盖**：Pynative + Graph 双模式、前向 + 反向（如有）、动态 shape

#### 数值验证标准（与 torch 对比时）

- **与 PTA 的对比**：即 **Step 4：精度零偏差验证**（见下），不在此步重复；本步 ST 的数值基线为 torch CPU（allclose）或由 PTA 仓库/用户脚本生成的 reference。
- **Step 2 的基线**：torch CPU 等价算子用 **allclose_nparray** 或单/双精度 atol/rtol；无 torch 等价时用 Step 4 或用户提供的 PTA 脚本产出 reference，ST 内与 reference 对比（如 md5sum）。

#### ST 数值基线策略：torch CPU 大算子 vs 小算子拼接（必读）

- **若存在对应 torch CPU 大算子**（如 `torch.xxx`、`torch_npu` 在 CPU 上的等价接口）：优先直接调用该大算子作为 ST 数值基线，与 MS 输出做 allclose / single_golden_compare。
- **若无对应 torch CPU 大算子、需小算子拼接时**：
  1. **先**用**业界通用或数学定义**实现参考（例如：KL 散度梯度用标准 KL 公式 + autograd、softmax 用标准定义等），仅使用 torch 小算子（sum/softmax/log/matmul 等），保证**形状与梯度结构**正确，使 ST 流程可跑通。
  2. 在实现处添加**注释**，并**提醒用户**：
     - 「本参考为占位实现，基于通用/数学定义，与 ACLNN/PTA 的 block-wise 或融合实现可能存在数值差异。」
     - 「**请向 PTA 或验收方索取**：该算子在**验证交付时使用的 torch CPU 小算子拼接实现**以及**验收标准**（如 rtol/atol、是否 0 偏差）。拿到后可将实现与验收标准提供给 AI，用于替换或增强当前参考并收紧对比条件。」
  3. 验收时以 PTA 验证交付用的实现与验收标准为准；拿到后可替换 ST 内 `_torch_cpu_xxx_ref` 等函数并更新对比逻辑。

### Step 3：Python UT（`reference.md` §8.1）—— 推荐

- 推导正确性：shape/type、动态/边界
- 错误路径：非法参数、None 语义覆盖
- 固定随机种子：`np.random.default_rng(seed)`

### Step 4：精度零偏差验证（`reference.md` §17.1，按需）—— **与 PTA 的对比即本步**

- **与 PTA 的对比**在此步完成：固定随机种子，MS 与 PTA 分别跑出输出，保存为 `.npy`（或 .npz），**md5sum 对比** MS/PTA 输出哈希，保证结果一致即可。
- **Agent 须产出可执行脚本供用户运行**：在 `tests/st/ops/ascend/` 下（或与 ST 同目录）提供 **精度零偏差验证脚本**（如 `verify_{op_name}_pta_md5sum.py`），脚本内完成：固定随机种子 → 分别调用 MS 与 PTA → 将各输出保存为 .npy/.npz → 计算并对比 md5（或打印 md5 供用户比对）。脚本头或 README 中注明运行环境（MS + torch_npu）、运行命令及需要用户回传的内容（输出文件或 md5 结果）。
- 交付 Step 4 时，向用户说明：「请在本仓库 tests/st/ops/ascend/ 下运行脚本 xxx，按脚本内说明回传结果」（见下方「需要用户配合的环节」）。

### Step 5：显存对齐验证（`reference.md` §17.2，按需）

- MS：`mindspore.runtime.max_memory_allocated()`
- PTA：`torch_npu.npu.max_memory_allocated()`
- 在相同阶段统计

### Step 6：组合场景分层验证（`reference.md` §29.4）

| 阶段 | 验证内容 |
| --- | --- |
| 子算子级 | 每个子算子独立 UT/ST |
| 组合级-中间值 | 临时 dump 中间 tensor 与 PTA 对比 |
| 组合级-最终输出 | 标准 ST 对齐 |
| 反向级 | 反向 ST + 数值梯度检查 |

### Step 7：按需补充（来自源文档 3/4/7）

- **反向调试**：若需查看 bprop 图，可设 `context.set_context(save_graphs=True, save_graphs_path='./')`，13_execute_*.ir 为后端图（见 3. 算子开发 3.4.3）。
- **动态 shape 自验**：用 `net.set_inputs(Tensor(shape=[3, None], dtype=...))` 等设定编译期动态维度，再传实际输入运行（见 4. 算子关键特性 4.2）。
- **性能自验（按需）**：整网打点用 `start_hook_net(hook_inside)`（须在网络执行前）；单 API 打点用 `start_analysis()`/`end_analysis()` 包住循环。见 `reference.md` §12、7. 接口性能自验工具。
- **vmap（按需）**：若算子需 vmap，见 `reference.md` §23（注册 VmapRule、结果/IR/效率自验、性能自测表）。

---

## 需要用户配合的环节

凡需用户跑脚本或执行命令的，Agent 必须同时做到：(1) **提供可执行脚本**（或生成并写出到仓库）；(2) 给出**运行命令**；(3) 明确**结果记录位置**（文件 + 章节/表格/段落），便于用户回填、Agent 后续直接读取。

| 环节 | 原因 | 向用户说明（含结果记录位置） |
| --- | --- | --- |
| Ascend ST 执行 | 需要 Ascend 设备 | "ST 测试需要在 Ascend 设备上运行。请执行：`cd tests/st/ops/ascend && pytest test_{op_name}.py -v`。请将**完整终端输出**记录到 **Feature 文档 §14 验收报告 → 功能验证表** 对应行备注，或粘贴到对话/指定文件 `docs/st_run_{op_name}.log`，以便我根据结果判断是否通过。" |
| 精度零偏差验证（Step 4） | 需同时跑 MS 和 PTA | "请运行我产出的验证脚本：`python tests/st/ops/ascend/verify_{op_name}_pta_md5sum.py`（脚本头有运行环境与命令说明）。请将 **md5 对比结果或通过/不通过** 记录到 **Feature 文档 §14 验收报告 → 功能验证表 →「是否与 PTA 计算结果 0 偏差」行** 的「自测结果」「备注」列，我后续会读取该文件确认。" |
| （仅当 ST 用小算子拼接且无大算子时）PTA 验证用 torch CPU 参考与验收标准 | 当前参考为通用/数学占位，与 ACL 可能存差 | "当前 ST 的 torch CPU 参考基于业界通用/数学定义实现，与 ACLNN/PTA 可能存在数值差异。若您能向 PTA 或验收方拿到**该算子在验证交付时使用的 torch CPU 小算子拼接实现**以及**验收标准**（rtol/atol 或 0 偏差等），请提供给我，我会据此替换或增强 ST 内参考并收紧对比条件。" |
| 性能/显存对比 | 需要真实设备 | "请在 Ascend 设备上运行性能脚本（我产出后给出路径与命令）。请将**耗时和显存数据**记录到 **Feature 文档 §14 验收报告 → 性能验证表** 对应行，或 `docs/perf_{op_name}.txt` 并在 Feature §14 注明路径。" |
| 稳定性 100 次验证 | 耗时较长 | "请在设备上执行 100 次循环脚本。请将**通过/失败与简要日志**记录到 **Feature 文档 §14** 或指定文件，并注明路径供我读取。" |

> agent 可以**生成测试脚本和验证命令**，但若无法直接访问 Ascend 设备，必须将脚本和运行指令交给用户执行，**等用户回传结果后再判断是否通过**。  
> **禁止**：只写「请跑脚本并回传结果」而不提供脚本路径、运行命令、以及**结果记录到哪个文件哪一段**。

---

## 🔒 Step 8 完成前强制检查（不可跳过）

**在标记 Step 8 为完成之前，必须逐项确认以下清单：**

```text
测试产出检查清单：

C++ UT 文件：
  - 文件路径：tests/ut/cpp/ops/test_ops_{op_name}.cc
  - 状态：✅已新建 / ❌未写（原因：___）

Python ST 文件：
  - 文件路径：tests/st/ops/ascend/test_{op_name}.py
  - 状态：✅已新建 / ✅已有且确认覆盖新路径（路径：___）/ ❌未写（原因：___）
  - 若"已有"：确认调用的是新接口而非旧接口？ 是/否

Python UT 文件（推荐）：
  - 文件路径：tests/ut/python/ops/test_{op_name}.py
  - 状态：✅已新建 / ⏭跳过（原因：___）

精度零偏差验证脚本（Step 4，与 PTA 对比）：
  - 文件路径：tests/st/ops/ascend/verify_{op_name}_pta_md5sum.py（或同目录等价命名）
  - 状态：✅已产出 / ⏭不适用（原因：___）
  - 脚本内是否含：固定种子、MS/PTA 分别跑、保存 .npy/.npz、md5 对比或输出？ 是/否
```

> 如果 C++ UT 或 Python ST 的状态为 ❌，**必须说明原因并暂停等用户确认后再继续**。
> 不允许静默跳过。

## 成功标准

- [ ] **C++ UT 文件已产出**（Infer 推导覆盖 unknown/None/动态shape）
- [ ] **Python ST 文件已产出或确认已有覆盖**（Pynative + Graph、前向 + 反向、动态 shape）
- [ ] Python UT 已产出或有理由跳过
- [ ] 稳定性验证：100 次运行无偶现失败（需用户在设备上验证）
- [ ] 覆盖场景：动态 shape / 静态 shape / 非连续 tensor / 空 tensor / 特殊值
- [ ] **用户/场景视角**：典型使用场景、业界论文或大模型中的调用方式与输入规格已在用例中覆盖；当前接入要求为对标 PTA，与 PTA 对标通过即视为满足（在验收/测试报告中记录结论）
- [ ] （精度零偏差）hash 对比通过（按需）
- [ ] （组合场景）分层验证通过（按需）

---

## 下一步

测试完成后，进入 **[Workflow 9: 文档](./09-docs.md)**
