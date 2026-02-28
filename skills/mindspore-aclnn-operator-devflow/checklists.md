# ACLNN 算子开发 Checklists（可复制）

> 目标：把"全流程文档"里的高频要求变成可执行清单，便于自检/转测/验收。
> 说明：清单偏"必须做什么"，细节说明见 `reference.md`。
>
> **优先级标记**：`[MUST]` = 必须通过；`[SHOULD]` = 强烈建议；`[NICE]` = 锦上添花。
>
> **本文件 `§` 编号与 SKILL.md workflow 步骤的对应关系**：
>
> | 本文件章节 | 对应 workflow 步骤 |
> | --- | --- |
> | §0a 对标分析, §0b ACLNN 能力 | Pre-A / Pre-B |
> | §0c 子算子盘点 | Pre-C（组合场景） |
> | §0d 版本/环境, §0e 方案评审 | Pre-B |
> | §0g Feature 文档 | Pre-B 初始化 → Step 10 补齐 |
> | §1 YAML/生成 | Step 1 / Step 2 |
> | §2 Infer | Step 3 |
> | §3 PyBoost, §3b 接口层 | Step 4 / Step 7 |
> | §4 KBK | Step 5 |
> | §5 BPROP | Step 6 |
> | §6 测试 | Step 8 |
> | §7 精度, §8 性能/显存 | 验收自验 |
> | §9 文档 | Step 9 |
> | §10 安全编码, §11 质量门禁 | 通用质量 |
> | §12 最终文件清单 | 提交前验证 |

## 0. 方案设计与对标（需求到落地）

### 0a. 对标分析
- [ ] `[MUST]` **对标对象明确**：PyTorch/PTA 的接口名、参数名/顺序/默认值、行为与约束（dtype/shape/layout/边界）。
- [ ] `[MUST]` **对标文档到位**：PTA 接口文档/示例齐全（包含约束、异常、边界）。
- [ ] `[MUST]` **对标代码到位**：定位 PTA 接入实现（torch_npu/op-plugin）并确认其实际调用的 ACLNN/aten 组合与参数预处理。
- [ ] `[MUST]` **PTA 源码三件套审查**（详见 `reference.md` §19）：
  - `op_plugin_functions.yaml`：前向/反向精确签名、参数类型/默认值、返回值个数
  - `derivatives.yaml`：可微输入、grad 函数参数传递顺序、`output_differentiability`
  - `XxxKernelNpuOpApi.cpp`（含 Grad）：实际 ACLNN 调用、None 处理、隐藏默认值、输出构造
- [ ] `[MUST]` **前向/反向参数差异已记录**：参数名/顺序/类型在前向与反向间的差异已识别并记入方案。
- [ ] `[MUST]` **代码与文档不一致已确认**：若发现 PTA 代码与文档矛盾，已整理差异清单交给用户，用户已找 ACLNN/PTA 接口人确认结论并记录。
- [ ] `[MUST]` **PTA 对标七要素**：接口名、参数名/顺序/默认值、算法行为、输入支持范围、隐式类型转换、同名重载、性能基线。
- [ ] `[MUST]` **ACLNN 调用链已提取**：若 PTA C++ 中有多个 ACLNN 调用，已提取完整调用链（前向+反向），画出依赖图（详见 `reference.md` §22）。

### 0b. CANN/ACLNN 能力
- [ ] `[MUST]` **CANN/ACLNN 能力确认**：正向/反向是否都有 ACLNN 大算子；不支持点形成书面结论。
- [ ] `[MUST]` **ACLNN 接口实证（反幻觉）**：已通过 grep 或查阅头文件确认 `aclnnXxx` 接口真实存在，且参数签名与代码一致（禁止仅凭经验臆造）。
- [ ] `[MUST]` **ACLNN 文档到位**：确认具体 `aclnnXxx`/`aclnnXxxGrad` 及其参数约束、layout/dtype/shape、workspace 接口。
- [ ] `[SHOULD]` **反向路径算子完整性**：反向函数中所用到的所有算子都已接入 ACLNN（避免退回非 ACLNN 路径）。
- [ ] `[SHOULD]` **aclnn 接口映射（仅路径 1）**：若走路径 1（自动生成）且算子名与 ACLNN 不一致，在 `aclnn_config.yaml`（或等价配置）中添加映射。路径 2（Customize / 显式编码接入）**不需要**修改 `aclnn_config.yaml`。

### 0c. 子算子盘点与补缺（组合场景）
- [ ] `[MUST]` **覆盖盘点表已产出**：调用链中每个子算子在 MS 中的接入状态已逐个确认（YAML/Infer/PyBoost/KBK）。
- [ ] `[MUST]` **C++ API / Prim 可用性确认**：PyBoost 拼接所需的 C++ API 函数已存在于 `functions.h`；KBK 拼接所需的 Prim 原语已在 `gen_ops_primitive_*.h` 中生成。
- [ ] `[MUST]` **缺失子算子已补齐**：盘点为"❌ 未接入"的子算子已走完步骤 1-8 并通过 UT。
- [ ] `[MUST]` **实施顺序正确**：叶子算子先于组合算子实现；有依赖关系的按拓扑序。
- [ ] `[SHOULD]` **子算子级验证通过**：每个子算子独立 UT/ST 通过后，再进入组合层开发。
- [ ] `[SHOULD]` **中间 tensor 对齐验证**：组合实现中的中间 tensor 与 PTA 逐阶段对比过。

### 0d. 版本/环境/影响面
- [ ] `[SHOULD]` **版本矩阵记录**：torch/torch_npu/CANN/芯片信息（支持范围会随版本变化，必须固定证据）。
- [ ] `[SHOULD]` **文档不足则跑探测**：生成并运行 PTA 支持范围探测脚本，回传 JSON 结果，再固化约束。
- [ ] `[SHOULD]` **影响面评估**：复用存量原语/接口时确保 CPU/GPU 不回退；全新算子不需要新增 CPU/GPU 支持。GE/Lite 有影响要给出 Pass/Expander/占位方案。

### 0e. 方案评审与交付范围
- [ ] `[MUST]` **交付范围写清楚**：支持的平台、模式（Pynative/KBK/GE）、动态 shape/rank、反向、性能/显存目标、遗留问题。
- [ ] `[MUST]` **接口设计决策明确**：接口分析五要素已完成——是否新增原语 / 复用原有原语；是否新增接口 / 复用原有接口；YAML 策略已确定（加 dispatch / 新建 `_ext` / 新建 / `ops.extend`）。详见 `reference.md` §15.4。
- [ ] `[MUST]` **接口/原语变更评审**：按评审规则确认（新增→重点评审；功能扩展→需评审；非兼容修改→原则不允许）。涉及修改已有原语参数签名时，参考 MS 仓库相似算子处理方式，确保已有 UT/ST 回归通过、跨后端（GE/CPU/GPU/Lite）不受影响。
- [ ] `[MUST]` **CPU/GPU 不回退（存量场景）**：若复用存量原语或适配存量算子 API，必须确保原有 CPU/GPU 流程正常、功能不退化；全新算子不需要新增 CPU/GPU 支持。
- [ ] `[SHOULD]` **问题归因与结论**：CANN 问题要有正式书面记录；框架/方案限制要有会议纪要（含议题/时间/人员/背景/结论）。
- [ ] `[SHOULD]` **差异收敛**：若无法对标，明确需要评审/新增接口。

### 0f. Feature 文档（评审与交付必须产物，`reference.md` §24）
- [ ] `[MUST]` **Pre-B 阶段初始化**：从 `templates/feature-document.md` 复制，填写 §1-§4、§6、§8。
- [ ] `[MUST]` **任务清单（13 大类）已逐项填写**：Primitive/functional/nn/tensor、后端/dtype、vmap、动态Shape、反向、资料、性能、功能、门禁、MS Adapter、并行、AMP、安全异常。
- [ ] `[MUST]` **开发章节同步更新**：每完成一个 Workflow Step，回填 Feature 文档对应章节（§5/§7/§9/§10/§11/§12）。
- [ ] `[MUST]` **代码改动说明完整**（§13）：列出所有新增/修改文件的完整路径。
- [ ] `[MUST]` **验收报告 — 资料验证表**（§14，17 项逐项自测标注）：
  1. 新增接口列表
  2. 典型场景 UT/ST 用例
  3. 中文 RST 与英文注释对应
  4. 接口描述详细准确
  5. 与 PyTorch 接口一致
  6. summary 提供公式
  7. 属性描述完整正确
  8. input 描述完整正确
  9. 输出描述完整正确
  10. 输出尺寸与 input 关系说明
  11. Raises 项完整正确
  12. 支持的平台已填写
  13. 资料格式（含样例格式）检查
  14. 样例已提供
  15. 样例有打印结果
  16. 样例执行情况是否 ok
  17. 算子与 API 能力沙盘是否补齐
- [ ] `[MUST]` **验收报告 — 功能验证表**（§14，26 项逐项自测标注）：
  1. 默认参数场景是否验证
  2. 空 Tensor 输入的正反向是否验证
  3. inf 和 nan 是否验证
  4. 算子支持数据类型是否与标杆对齐（pytorch npu/gpu/cpu）
  5. 输入取值范围是否有验证
  6. 输入维度是否有覆盖 0D-8D
  7. 输入支持的 dtype 是否全覆盖
  8. 输入是否支持隐式类型转换
  9. 输入是否支持广播
  10. 输入之间的约束是否有验证
  11. 正向的精度验证是否通过
  12. 反向是否支持
  13. 反向是否是单算子实现
  14. 异常用例是否校验具体报错信息
  15. 是否提供报错白名单
  16. 动态 shape/rank/属性是否都支持
  17. 是否关闭退避功能验证 `MS_DISABLE_KERNEL_BACKOFF=1`
  18. 测试仓接口相关用例是否全部 PASS，无遗留问题单
  19. 是否支持 bf16
  20. 多输入算子的 bprop 函数是否有考虑反向按需求导
  21. 算子输出 shape 是否依赖于算子的计算结果
  22. 是否支持非连续输入
  23. 是否与 PTA 计算结果 0 偏差（MD5 对比）
  24. 是否会使得运算符或存量 ops 接口调用到新增的原语
  25. 是否已支持 amp（混合精度）特性
  26. 若多 Tensor 输入，是否支持各 Tensor 数据类型不一致
- [ ] `[MUST]` **验收报告 — 性能验证表**（§14，4 项）：
  1. 性能测试覆盖 ≥3 种规格
  2. 反向显存优化（SetUnusedInputs）
  3. 性能不低于友商（波动 ≤10%）
  4. 显存持平 PTA
- [ ] `[MUST]` **验收报告 — 安全编码检视表**（§14，12 项）：
  指针判空/先用后校/越界/除零/内存泄露/异常路径释放/nothrow/安全函数库/类型转换溢出/冗余代码/敏感信息/弱随机数
- [ ] `[SHOULD]` **与 PTA 差异说明清晰**（§8）：明确列出差异及原因。
- [ ] `[SHOULD]` **Feature 文档随代码 PR 一起提交**。

## 1. YAML / 接口生成
- [ ] `[MUST]` **接入路径已明确**：路径 1（自动生成，参数直通）或路径 2（Customize，参数需预处理），记录决策依据。
- [ ] `[MUST]` **前向/反向 YAML 各 1 份**（如需要反向单算子）。
- [ ] `[MUST]` **dispatch 配置正确**：路径 1 → `enable: True` 不写 `Ascend`；路径 2 → `enable: True` + `Ascend: XxxAscend`。
- [ ] `[MUST]` **None 语义一致**：若参数允许 `default: None`，必须定义 None 的行为并在 Infer/PyBoost/KBK/文档/测试同步。
- [ ] `[MUST]` **gen_ops.py 可跑通**：常见报错（keys/`py_method`/function_doc）已解决。
- [ ] `[SHOULD]` **若需自定义**：使用"先自动生成骨架→拷贝到 customize→改造→恢复 YAML"的套路。
- [ ] `[SHOULD]` **英文 YAML 不混入中文**：Windows GBK 编码可能导致生成/编译问题。

## 2. Infer（GeneralInfer 优先）
- [ ] `[MUST]` **只做 shape/type 推导**：不做运行期合法性校验（交给 ACLNN/运行时）。
- [ ] `[MUST]` **动态约定**：局部动态用 -1，动态秩用 -2（或项目等价常量）。
- [ ] `[SHOULD]` **尽量精确推导**：关键参数未知时回退动态维；已知时给精确维度。
- [ ] `[MUST]` **不要在 Infer 里修改属性**（禁止 AddAttr/改 prim attr）。
- [ ] `[MUST]` **sequence/none 先判定再取 shape/type**（避免 GetShape/GetType 报错）。
- [ ] `[MUST]` **Infer 注册宏完整**：按项目新接口注册（以仓库现有模式为准）。
- [ ] `[SHOULD]` **InferInfo API 规范**：使用 `GetScalarValueWithCheck`/`GetArrayValue`/`HasUnknownValue`/`IsNone`，不臆造 API。

## 3. PyBoost（Pynative）
- [ ] `[MUST]` **路径 1**：确认 gen_ops.py 自动生成的 PyBoost 调用代码存在且参数正确，编译通过。
- [ ] `[MUST]` **路径 2 输入转换规范**：tuple/list → `std::vector<int64_t>`；标量按框架工具函数取值。
- [ ] `[MUST]` **路径 2 调用 ACLNN 两段式**：workspace size + launch（以项目封装宏为准）。
- [ ] `[MUST]` **非 Ascend 占位**：明确报错（RuntimeError/NotImplementedError）且文档同步说明。
- [ ] `[SHOULD]` **View 算子**（如涉及纯 shape/strides 变换，详见 `reference.md` §26）：
  - 原始算子 YAML 已加 `view: True`
  - `ops/view/{op_name}_view_strides_calc.cc` + 头文件已实现 strides 计算
  - `REG_VIEW_STRIDES_CALC_FUN` 注册完成

## 3b. 接口层（functional / nn / Tensor）
- [ ] `[MUST]` **functional 使用 `_get_cache_prim`**：避免反复 `__init__` 造成性能问题。
- [ ] `[MUST]` **nn.Cell construct 不直接 raise**：编译期校验/抛错使用 `@constexpr` 辅助函数。
- [ ] `[MUST]` **`__init__.py` 与 `__all__` 导出**：新增接口必须在对应包的 `__init__.py` 中导出。
- [ ] `[SHOULD]` **Tensor 方法覆盖多模式**：PyNative/KBK 与 GE（若项目要求）。
- [ ] `[SHOULD]` **GE 映射已注册**：`resource.cc` + `standard_method.py`（若涉及 Tensor 方法）。
- [ ] `[SHOULD]` **mint 接口注册**：若需要 mint 接口，在对应 `__init__.py` 中注册。
- [ ] `[SHOULD]` **复杂接口的"一对多映射"**：按参数分支选择不同 Primitive 或组合实现。

## 4. KBK（Graph / KernelByKernel）
- [ ] `[MUST]` **路径 1**：确认自动注册（`MS_ACLNN_COMMON_KERNEL_FACTORY_REG`）存在且算子名正确，编译通过。
- [ ] `[MUST]` **路径 2 标准结构**：`GetWorkSpaceInfo()` + `Launch()` + 工厂注册宏完整。
- [ ] `[MUST]` **路径 2 前向/反向分文件、分注册**：头/实现命名空间保持一致。
- [ ] `[MUST]` **避免 Launch 里申请显存**：不允许 `cudaMalloc/cudaFree`（GPU 场景）等运行期分配；统一在 Resize/workspace 申请。
- [ ] `[MUST]` **Init/Resize/Launch 职责分离**：能在 Init 确定的放 Init；与 shape 相关的放 Resize；Launch 只做发射。
- [ ] `[SHOULD]` **无意义输出处理**：如有预留输出，覆写 `GetUseLessOutputIdx()`（或项目等价接口）避免 dump/溢出误检。
- [ ] `[SHOULD]` **Compute-depend 输出**：最大 size 分配 + `Sync`/更新输出 shape（按框架要求）。
- [ ] `[SHOULD]` **输入转属性优化**：若有输入实质是属性，覆写 `GetLaunchIgnoredInputAddressIdx()`。
- [ ] `[SHOULD]` **View Host Kernel**（当 YAML 标记 `graph_view: True` 时，详见 `reference.md` §26.4）：
  - View 专用 YAML 已配置 `graph_view: True` + `labels`
  - `ops/kernel/host/view/kernel_mod_impl/{op_name}_view.cc/.h` 已实现
  - `MS_HOST_REG_KERNEL` 注册完成

## 5. BPROP（Expander）
- [ ] `[MUST]` **反向输入/输出个数**：反向输入 = 正向输入 + 2（out, dout），反向输出 = 正向输入个数。
- [ ] `[MUST]` **不可微分入参**：用 `ib->OutZeros(x)` 返回零梯度（MS 要求输出个数对齐）。
- [ ] `[MUST]` **按需求导**：所有可微输入先 `need_compute_grad_out()`，不需要则 `OutZeros()`。
- [ ] `[MUST]` **多输出正向的反向处理**：`out` 为 tuple 时，通过 `TupleGetItem` 取对应输出。
- [ ] `[MUST]` **反向大算子 mask 参数**：若反向 ACLNN 需要 grad_mask，通过 `need_compute_grad_out()` 构造。
- [ ] `[SHOULD]` **梯度确实为 0**：用 `ib->ZerosLikeExt()`（避免退回非 ACLNN 路径）。
- [ ] `[SHOULD]` **SetUnusedInputs**：反向不依赖的输入标记为 unused，便于 Pynative 更早释放内存。
- [ ] `[SHOULD]` **inplace 反向**：如反向需要用到更新前的 self，注册 `CloneInplaceInput(...)`。
- [ ] `[SHOULD]` **KBK 动态 shape + inplace**：必要时用 `ib->Depend(target, inplace_call)` 保序。
- [ ] `[SHOULD]` **反向实现简洁性**：不要引入额外不必要的算子，避免误差累积。

## 6. 测试（UT/ST/自验）

### 6a. 测试文件产出（逐项确认，不允许遗漏）
- [ ] `[MUST]` **C++ UT 文件已产出**：`tests/ut/cpp/ops/test_ops_{op_name}.cc`，覆盖动态维/动态秩/unknown/None。
- [ ] `[MUST]` **Python ST 文件已产出或确认已有覆盖**：`tests/st/ops/share/_op_info/op_database.py`。

### 6b. 场景覆盖
- [ ] `[MUST]` **默认参数场景验证**：使用所有默认参数值调用前向+反向，确认基本路径可通。
- [ ] `[MUST]` **动态 shape 自验**：前端测试用例文件调用 `OpsFactory` 的 `test_op_dynamic` 方法。
- [ ] `[MUST]` **空张量输入**：验证空张量的前向/反向是否支持或正确报错。
- [ ] `[MUST]` **输入 dtype 全覆盖**：算子声明支持的所有 dtype 都有对应用例（含不支持类型的异常用例）。
- [ ] `[MUST]` **输入维度覆盖**：合法维度（输入维度覆盖 0D/8D和一个中间大小的维度（如果支持））和非法维度都有用例。
- [ ] `[MUST]` **输入取值范围验证**：边界值、极端值（极大/极小）、margin/reduction 等枚举参数全覆盖。
- [ ] `[MUST]` **输入间约束验证**：形状匹配/不匹配、dtype 一致/不一致、rank 一致/不一致。
- [ ] `[MUST]` **异常用例校验具体报错信息**：异常场景需断言 TypeError/ValueError/RuntimeError 的具体 message。
- [ ] `[MUST]` **多布局覆盖**：若算子支持多种 layout（如 BSND/TND/PA_BSND），覆盖所有布局组合的前后向。
- [ ] `[MUST]` **非连续张量**：通过 transpose/permute 构造非连续输入，验证正确性。
- [ ] `[MUST]` **特殊值健壮性**：inf/-inf/nan 场景验证（至少不 crash，形状/流程正确）。
- [ ] `[SHOULD]` **多 batch 变长序列**：若涉及 actual_seq_len 类参数，覆盖多 batch + 变长场景。
- [ ] `[MUST]` **bf16 场景**：bf16 支持情况确认（支持则测精度，不支持则有异常用例）；比较前升精到 float32。
- [ ] `[MUST]` **隐式类型转换**：确认是否支持输入间 dtype 不同时的自动提升；不支持则有异常用例。
- [ ] `[MUST]` **广播**：确认是否支持输入间 shape 广播；不支持则有异常用例。
- [ ] `[MUST]` **多 Tensor 输入 dtype 不一致场景**：确认是否支持多 Tensor 输入 dtype 不一致场景；不支持则有异常用例。非多Tensor输入的算子不需要。

### 6c. 测试组合与稳定性
- [ ] `[MUST]` **测试组合覆盖**：接口形态（functional/nn/Tensor）× 后端 × 模式（Pynative/KBK）× shape 类型（静态/动态）。
- [ ] `[MUST]` **关闭退避验证**：`export MS_DISABLE_KERNEL_BACKOFF=1` 环境下用例全部通过（防止退回非 ACLNN 路径）。
- [ ] `[MUST]` **稳定性验证**：关键用例多次运行（例如 100 次）无偶现再上库。
- [ ] `[MUST]` **Pass 用例截图/日志**：提供通过证明（截图或日志）。
- [ ] `[SHOULD]` **测试仓存量用例回归**：若存在测试仓 ST，确认全部 PASS 无遗留问题单。

### 6d. 功能合规性确认
- [ ] `[MUST]` **不影响存量接口**：新增算子/原语不会使运算符或存量 ops 接口调用到新增原语（除非设计如此）。
- [ ] `[SHOULD]` **AMP 混合精度**：确认是否已支持或不涉及（新增 Primitive 需关注 amp_white/black_list）。
- [ ] `[SHOULD]` **多 Tensor 输入 dtype 不一致**：若多输入算子，确认是否支持各输入 dtype 不同。
- [ ] `[SHOULD]` **输出 shape 是否依赖计算结果**：若是 compute-depend 输出，需要 SyncOutputShape 机制。

## 7. 精度 0 偏差（对标 PTA）
- [ ] `[SHOULD]` **MS 与 PTA 使用相同种子和输入**（§6a 的种子要求是通用可复现；此处强调 MS/PTA 双侧输入完全一致）。
- [ ] `[SHOULD]` **二进制一致性**：用 `md5sum`（或等价方式）对比输出 `.npy`，确保 bitwise 一致（若目标要求如此）。
- [ ] `[SHOULD]` **Kernel 一致性**：通过 profiling 验证 MS 与 PTA 调用了相同的底层 kernel。

## 8. 性能与显存（对标 PTA）

### 8a. 性能
- [ ] `[MUST]` **性能达标**：Ascend 算子性能与 PTA 对比 ≤1.1 倍（或按项目门槛）。
- [ ] `[SHOULD]` **性能自验覆盖**：≥3 种规格（小/中/大），含 kernel 性能和端到端性能。
- [ ] `[SHOULD]` **性能工具**：使用 `apitimewrapper`（或等价工具）打点；排除框架首次启动/拷贝耗时。

### 8b. 显存
- [ ] `[SHOULD]` **MS 显存统计**：`mindspore.runtime.max_memory_allocated()`
- [ ] `[SHOULD]` **PTA 显存统计**：`torch_npu.npu.max_memory_allocated()`
- [ ] `[SHOULD]` **对比位置一致**：正向/反向在相同阶段统计，避免把初始化/编译混入。

## 9. 文档（中英文一致）

### 9a. 文档产出
- [ ] `[MUST]` **英文 function_doc 完整**：`doc/{op}_doc.yaml` 中 desc/args/returns/examples 齐全。
- [ ] `[MUST]` **中文 RST 已创建**（公开 API）：`docs/api/api_python/ops/` 或对应 mint/nn 目录。
  英文 doc YAML ≠ 文档完成——中文 RST 是独立产物，**最容易遗漏**。
  仅"内部算子（不在 `__all__` 导出）"允许跳过，且须标注原因。

### 9b. 文档内容完整性（资料验证）
- [ ] `[MUST]` **接口描述详细准确**：功能、参数、返回值、约束、异常均有说明。
- [ ] `[MUST]` **summary 提供公式**：涉及数学运算的算子必须有公式说明。
- [ ] `[MUST]` **参数（属性）描述完整正确**：类型、取值范围、默认值、含义。
- [ ] `[MUST]` **输入描述完整正确**：shape 约束、dtype 约束、是否可选。
- [ ] `[MUST]` **输出描述完整正确**：shape 与输入的关系（特别是 reduction 等参数影响 shape 的场景）。
- [ ] `[MUST]` **Raises 项描述完整正确**：列出所有可能的异常类型和触发条件。
- [ ] `[MUST]` **示例可运行**：包含完整 import，**有打印结果**。
- [ ] `[MUST]` **与 PyTorch 接口一致性说明**：参数、语义、行为是否一致；差异处需说明。
- [ ] `[MUST]` **支持平台已填写**：`Ascend` / `GPU` / `CPU`。
- [ ] `[SHOULD]` **算子与 API 能力沙盘已补齐**：在能力矩阵/索引中登记新接口。

### 9c. 文档格式
- [ ] `[MUST]` **中英文严格一致**：参数/默认值/必选可选/约束/示例一致。
- [ ] `[MUST]` **接口列表按字母序**：避免冲突与重复。
- [ ] `[MUST]` **文件名/标题/API 名三者一致**：否则页面生成失败。
- [ ] `[SHOULD]` **公式/论文引用**：涉及数学公式使用 `:math:` 标记；相关论文给出引用。
- [ ] `[SHOULD]` **交叉引用**：相关接口间使用 `:class:`/`:func:` 链接。
- [ ] `[NICE]` **中文标点与反引号空格规范**：中文里 ` 或 `` 与前后保持 1 个空格。

## 10. 安全编码检视
- [ ] `[MUST]` **空指针判空**：Infer 中的 Primitive、输入输出、GetShape、GetType、GetValue 不需要判空外，其他情况需要判空。
- [ ] `[MUST]` **指针先校验后使用**：不允许先使用后校验。
- [ ] `[MUST]` **数组/指针访存越界**：确认无越界访问。
- [ ] `[MUST]` **除零保护**：涉及除法时有 EPSILON 或显式判断。
- [ ] `[MUST]` **内存泄露**：new/malloc 内存须释放；RAII 或框架资源管理优先。
- [ ] `[MUST]` **异常路径资源释放**：异常/错误处理分支中文件句柄、内存等资源须释放。
- [ ] `[MUST]` **C++ 异常处理规范**：不使用 try-catch 做逻辑流控；错误用框架异常宏。
- [ ] `[MUST]` **检视范围完整**：Python 原语 + NN/functional/Tensor + C++ Infer + bprop + 后端 kernel + Grad 单算子。
- [ ] `[MUST]` **new 声明 nothrow**：使用 new 创建对象须声明为 nothrow（或按项目规范处理分配失败）。
- [ ] `[MUST]` **数据类型转换安全**：无上溢或下溢的隐式转换（如 int64→int32 截断）。
- [ ] `[MUST]` **冗余代码清理**：无冗余校验、不可达代码、未使用变量。
- [ ] `[SHOULD]` 内存操作用安全函数库（securec 等，项目要求时）。
- [ ] `[SHOULD]` 避免弱随机数、敏感信息硬编码、日志泄露敏感信息。

## 11. 质量门禁（项目要求摘要）
- [ ] `[MUST]` 行长 ≤120；无 Tab；UTF-8；无多余空格；文件末尾保留一行空行。
- [ ] `[MUST]` 通过项目要求的 Check_*（Notebooklint/Pylint/Rstlint/Scanoss/Shellcheck/Tab/Utf8 等）。

## 11b. 跨文件变更一致性（D1）

- [ ] `[MUST]` 若修改了参数名/类型/默认值/约束，已对 8 类文件（op_def/api_def YAML、GeneralInfer C++、PyBoost C++、KBK kernel C++、bprop C++、Python 接口层、中英文文档）做全量检索并同步更新，旧值零残留。

## 12. 最终文件清单验证（提交前必做）

> 根据接入路径，逐项确认所有必要文件都已就位。
> 标记 `[自动]` 的文件由 gen_ops.py 生成（不入库，但运行时必需，须确认已生成）。
> 标记 `[手写]` 的文件需要开发者创建/修改。

### A. YAML 定义文件（新建）
- [ ] `[MUST]` `ops/op_def/yaml/{op_name}_op.yaml`（前向 op_def）
- [ ] `[MUST]` `ops/op_def/yaml/{op_name}_grad_op.yaml`（反向 op_def，如需要）
- [ ] `[MUST]` `ops/api_def/{op_name}.yaml`（api_def）
- [ ] `[MUST]` `ops/op_def/yaml/doc/{op_name}_doc.yaml`（英文 function_doc，`_ext` 风格）或 `ops/api_def/function_doc/`（旧风格）——以仓库中同类算子路径为准
- [ ] `[SHOULD]` `ops/api_def/method_doc/{op_name}_doc.yaml`（Tensor 方法文档，如有 Tensor 接口）
- [ ] `[SHOULD]` `ops/op_def/deprecated/{op_name}_method.yaml`（旧接口兼容，如需向后兼容）

### B. Infer 文件（新建）
- [ ] `[MUST]` `ops/infer/ops_func_impl/{op_name}.h` + `.cc`（GeneralInfer）

### C. ACLNN 映射（修改已有文件，**仅路径 1**）
- [ ] `[MUST]`（路径 1）`python/mindspore/ops_generate/pyboost/aclnn_config.yaml`：新增 `{OpName}: 'aclnn{Op}'` 映射
- ⏭️ 路径 2（Customize / 显式编码接入 ACLNN）**跳过此项**——ACLNN 调用在手写 customize 代码中直接指定，不需要 `aclnn_config.yaml` 映射

### D. Python 接口导出（修改已有文件，按需逐项确认）
> 以下按接口形态分类。不是所有接口都需要——取决于算子暴露为 functional/mint/Tensor 的哪些。
> **必须先确认仓库中同类算子暴露了哪些接口**，然后逐个对照。

- [ ] `[MUST]` `python/mindspore/ops/function/math_func.py`（或对应 `*_func.py`）：
  - import `{op_name}` from `auto_generate`
  - 如有别名（如 `arccos_ext`），新增别名函数
  - 更新 `__all__` 列表
- [ ] `[MUST]` `python/mindspore/mint/__init__.py`（如需 mint 接口）：
  - import `{op_name}_ext as {op_name}`
  - 更新 `__all__` 列表
- [ ] `[SHOULD]` `python/mindspore/ops/tensor_method.py`（如需 Tensor 方法）：
  - import `{op_name}_ext`
  - 新增 `tensor_{op_name}` 包装函数
- [ ] `[SHOULD]` 接口重载（如涉及同名多签名，详见 `reference.md` §25）：
  - `ops/api_def/{op_name}.yaml` 多条目配置正确
  - `ops/op_def/deprecated/{op_name}_method.yaml` 参数与 py_method 一致（如需新旧兼容）
  - alias YAML 声明正确（如有符号别名）

### E. 测试文件
- [ ] `[MUST]` `tests/ut/cpp/ops/test_ops_{op_name}.cc`（C++ UT，必须新建）
- [ ] `[MUST]` `tests/st/ops/share/_op_info/op_database.py`(须存在对应接口名的注册)

### F. 中文 RST 文档（按接口形态，公开 API 必须）
> 每个接口形态可能需要独立的中文 RST 文件，不要只写一份。

- [ ] `[MUST]` `docs/api/api_python/ops/mindspore.ops.func_{op_name}.rst`（functional 接口）
- [ ] `[MUST]` `docs/api/api_python/mint/mindspore.mint.func_{op_name}.rst`（mint 接口，如有）
- [ ] `[SHOULD]` `docs/api/api_python/mindspore/Tensor/mindspore.Tensor.{op_name}.rst`（Tensor 方法，如有）
- [ ] 如有别名（如 `arccos`），别名也需要独立 RST

### G. Feature 文档
- [ ] `[MUST]` `{op_name}_Feature.md`（评审与交付必须产物）

### H. 自动生成文件（须确认存在，不入库）
- [ ] `[MUST]` `functional_overload.py` 中包含新算子 Python 入口
- [ ] `[MUST]` 路径 1：PyBoost 自动调用代码 + `aclnn_kernel_register_auto.cc` 注册

### I. 路径 2 额外需要的文件（手写）
- [ ] `[MUST]` `kernel/.../pyboost_impl/customize/{op_name}.h` + `.cc`（PyBoost 前向）
- [ ] `[MUST]` `kernel/.../pyboost_impl/customize/{op_name}_grad.h` + `.cc`（PyBoost 反向，如需要）
- [ ] `[MUST]` `kernel/.../kernel_mod_impl/customize/{op_name}_aclnn_kernel.h` + `.cc`（KBK 前向）
- [ ] `[MUST]` `kernel/.../kernel_mod_impl/customize/{op_name}_grad_aclnn_kernel.h` + `.cc`（KBK 反向，如需要）

### J. View 算子文件（如涉及 View 特性，详见 `reference.md` §26）
- [ ] `[SHOULD]` `ops/op_def/yaml/{op_name}_view_op.yaml`（View 专用 YAML：`view: True` + `graph_view: True` + `labels`）
- [ ] `[SHOULD]` `ops/view/{op_name}_view_strides_calc.cc` + `ops/include/view/{op_name}_view_strides_calc.h`（strides 计算 + `REG_VIEW_STRIDES_CALC_FUN` 注册）
- [ ] `[SHOULD]` `ops/kernel/host/view/kernel_mod_impl/{op_name}_view.cc` + `.h`（KBK host kernel + `MS_HOST_REG_KERNEL` 注册）

### K. BPROP（如需要反向）
- [ ] `[MUST]` `ccsrc/frontend/expander/grad/grad_*_ops.cc` 中 `REG_BPROP_BUILDER` 注册

### L. 构建系统
- [ ] `[MUST]` 路径 2 的新 `.cc` 文件在 `CMakeLists.txt` 的 `GLOB_RECURSE` 范围内

---

## 提交前必检 Top-25（精简版）

> 从以上完整清单中提炼最易遗漏的 25 项，提交 PR 前快速过一遍。

| # | 检查项 | 来源章节 |
| --- | --- | --- |
| 1 | **接入路径已明确**（路径 1 自动 / 路径 2 Customize）且 YAML dispatch 配置正确 | §1 YAML |
| 2 | PTA 源码三件套已审查（functions.yaml / derivatives.yaml / C++） | §0a 对标分析 |
| 3 | **Feature 文档 14 个章节已全部填写** | §0f Feature 文档 |
| 4 | **Feature 文档验收报告四张表已逐项自测**（资料验证 17 项 + 功能验证 26 项 + 性能验证 4 项 + 安全编码 12 项） | §0f Feature 文档 |
| 5 | **复用存量原语/接口时，CPU/GPU 流程不回退**（已有 UT/ST 回归通过） | §0e 方案评审 |
| 6 | gen_ops.py 跑通，functional_overload.py 已生成 | §1 YAML/生成 |
| 7 | **`aclnn_config.yaml` 已添加映射**（`{OpName}: 'aclnn{Op}'`）——**仅路径 1**；路径 2 跳过 | §12-C |
| 8 | Infer 里没有修改属性（禁止 AddAttr） | §2 Infer |
| 9 | **Python 接口导出完整**：`math_func.py` import + 别名 + `__all__`；`mint/__init__.py`；`tensor_method.py` | §12-D |
| 10 | 前向/反向分文件、分注册，命名空间一致（路径 2） | §4 KBK |
| 11 | 反向输入/输出个数正确（+2 / 对齐） | §5 BPROP |
| 12 | 不可微分入参返回了 OutZeros | §5 BPROP |
| 13 | 若为手写接口，functional 使用了 `_get_cache_prim` | §3b 接口层 |
| 14 | 非 Ascend 设备有占位 RuntimeError | §3 PyBoost |
| 15 | **中文 RST 已创建**（每个接口形态一份：ops/mint/Tensor，别名也要） | §12-F |
| 16 | **最终文件清单 §12 全量验证通过**（A-L 逐项确认） | §12 |
| 17 | 中英文文档参数/默认值/示例一致，**样例有打印结果** | §9b/9c 文档 |
| 18 | 测试覆盖：动态 shape + 空张量 + 多布局 + None + dtype 全覆盖 | §6b 场景覆盖 |
| 19 | **`MS_DISABLE_KERNEL_BACKOFF=1` 退避关闭验证**通过 | §6c 测试组合 |
| 20 | 稳定性：关键用例跑 100 次无偶现 | §6c 稳定性 |
| 21 | **异常用例校验具体报错信息**（TypeError/ValueError/RuntimeError message 断言） | §6b 场景覆盖 |
| 22 | **文档 summary 提供公式**（涉及数学运算的算子）；支持平台已填写 | §9b 文档内容 |
| 23 | **不影响存量 ops 接口**（新增原语不会被存量接口意外调用） | §6d 功能合规 |
| 24 | **安全编码 12 项全过**（指针/越界/除零/内存/nothrow/类型转换/冗余） | §10 安全编码 |
| 25 | 行长 ≤120，无 Tab，UTF-8，文件末尾空行 | §11 质量门禁 |
