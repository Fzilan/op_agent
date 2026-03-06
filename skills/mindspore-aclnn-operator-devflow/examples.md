# 使用示例（给 agent 的触发样例）

下面这些用户话术出现时，应该自动应用 `mindspore-aclnn-operator-devflow`。

## 示例 1：新增一个 ACLNN 算子
**用户**：我想在 MindSpore 里适配一个 ACLNN 算子 `foo`，要支持前向和反向、动态 shape。

**期望 agent 行为**：
- 先对齐需求：算子名、输入输出、layout、dtype、是否允许 None、动态策略、对标实现（torch_npu/参考文档）。
- **确定接入路径**：分析 MS API 参数能否原样透传给 ACLNN——路径 1（自动生成）或路径 2（Customize）。
- 分析目标算子的类别特征（如：单 ACLNN 直连 vs 组合算子、是否有反向、输入布局 TND/BSND 等），在仓库中搜索已接入的同类算子作为参照，定位 YAML/Infer/PyBoost/KBK/bprop/测试/文档的位置。
- 按"落地顺序"逐步实现。路径 1 跳过 Step 4/5 手写（只验证自动生成产物）；路径 2 必须手写 PyBoost + KBK。每一步给出改动文件清单与验证方式。

## 示例 2：gen_ops.py 报错排障
**用户**：我写了 `*_op.yaml`，跑 `gen_ops.py` 报 `py_method is empty`。

**期望 agent 行为**：
- 解释这是 YAML keys/字段缺失导致的生成失败。
- 用已有基础算子 YAML 对照，补齐 `py_method` 与必要字段。
- 再次运行生成并把日志中的关键证据贴出来（成功/失败原因）。

## 示例 3：动态 shape 推导不准
**用户**：GeneralInfer 里 `actual_seq_len` 编译期未知时，我输出 shape 应该怎么写？

**期望 agent 行为**：
- 说明"关键参数 unknown 时输出维度回退动态维"的策略（其余维沿用输入）。
- 给出当前工程中相同策略的代码证据（引用相似算子实现）。
- 补齐 UT 覆盖 unknown value 场景。

## 示例 4：KBK kernel 编译报未声明/未定义
**用户**：我加了 `*_aclnn_kernel.cc`，编译报类未声明。

**期望 agent 行为**：
- 优先检查头/实现命名空间是否一致、类名/文件名/注册宏是否一致。
- 确保前向/反向分文件且各自注册。
- 给出修复前后差异，并重新编译验证。

## 示例 5：ST 对齐 torch_npu 数值不一致
**用户**：ST 对齐 torch_npu，bf16 比较老是不过。

**期望 agent 行为**：
- 先对齐比较策略：先 shape/type，再数值；bf16 先升精到 fp32 再比。
- 检查是否引入了额外算子造成误差累积。
- 输出对比证据（最大误差、位置、输入种子复现方式）。

## 示例 6：我要做 0 偏差（bitwise）验收
**用户**：这个算子要做到和 PTA 输出 0 偏差，怎么自验？

**期望 agent 行为**：
- 固定随机种子，确保 MS/PTA 输入一致。
- 保存输出为 `.npy`，用 `md5sum`（或等价方式）对比哈希给出"二进制一致"证据。

## 示例 7：我要对齐显存占用
**用户**：显存要和 PTA 持平/不劣化，怎么测？

**期望 agent 行为**：
- 给出 MS/PTA 分别统计 max memory 的位置与 API（`mindspore.runtime.max_memory_allocated()` /
  `torch_npu.npu.max_memory_allocated()`）。
- 强调统计阶段要一致（避免把初始化/编译混进去），并输出对比数值证据。

## 示例 8：我要写 RFC/转测材料，要求"有书面结论"
**用户**：有些限制没法修，怎么写结论才能转测？

**期望 agent 行为**：
- 先判定问题来源：CANN 问题 vs 框架/方案限制。
- CANN 问题要求正式书面记录；框架/方案限制给会议纪要模板（议题/时间/人员/背景/结论）。
- 将结论与遗留问题写进 RFC/转测说明，并标注影响范围与规避方案。

## 示例 9：我要做性能自验/打点
**用户**：我想看接口端到端耗时和内部耗时，怎么打点？

**期望 agent 行为**：
- 推荐 `apitimewrapper` 的用法：整网打点与单 API 打点两种。
- 提醒 `hook_inside` 的含义，并建议先排除框架首次启动/拷贝影响再测。
- 给出具体操作步骤：
  1. 生成性能测试脚本（包含 `start_hook_net`/`start_analysis`/`end_analysis` 调用）。
  2. 运行脚本，收集 CSV/JSON 输出。
  3. 解析结果：提取关键 API 的 wall-time、对比 MS vs PTA 同接口耗时。
  4. 若耗时劣化，定位到具体子 API/kernel 并给出优化建议。
- 输出证据：运行日志 + 耗时表格 + 与 PTA 的对比结论。

---

## 示例 10：ACLNN 没有 Grad 大算子，反向怎么办
**用户**：我查了 ACLNN 文档，这个算子没有 Grad 接口，但业务需要反向。

**期望 agent 行为**：
- **不要假设存在未文档化的 ACLNN Grad 接口**，先确认事实：
  1. 搜索 ACLNN 头文件 / PTA op-plugin 确认是否有 `aclnnXxxGrad`。
  2. 检查 PTA 侧反向实现：是否用了多个小算子组合（aten 组合）来做反向。
- 给出可选方案：
  - **方案 A**：用现有 ACLNN 小算子组合实现反向（参考 PTA 的 autograd 实现拆解）。
  - **方案 B**：在 bprop 中用 MindSpore 现有算子组合反向图（可能性能不如大算子）。
  - **方案 C**：如果 PTA 也不支持反向，书面记录后标注"不支持反向"并在文档中说明。
- 向用户说明各方案的**精度/性能/维护成本**权衡，让用户决策。

## 示例 11：CANN 版本不支持某 feature，需要降级
**用户**：在 CANN 8.0 上跑报 `unsupported`，但 CANN 8.5 可以，怎么处理？

**期望 agent 行为**：
- **先确认版本矩阵**：记录目标 CANN 版本与实际支持范围。
- **查找 PTA 的处理方式**：PTA 是否有版本分支（`#if CANN_VERSION >= ...`）或运行时降级。
- 给出处理方案：
  - 在 YAML/文档中明确标注最低 CANN 版本要求。
  - 在 Infer/PyBoost/KBK 中加版本检查，低版本给清晰报错（而非 segfault）。
  - 若需要兼容低版本，给出降级实现方案（小算子组合替代或功能裁剪）。
- 输出版本兼容性矩阵表格作为证据。

## 示例 12：运行时崩溃（不是编译报错）
**用户**：算子编译通过了，但运行时在 Launch 里 coredump / 报 ACLNN 内部错误。

**期望 agent 行为**：
- **不要盲目改代码**，先按以下步骤排障：
  1. **收集崩溃信息**：错误码/堆栈/ACLNN 日志（`ASCEND_GLOBAL_LOG_LEVEL`）。
  2. **构造最小复现用例**：缩小输入 shape/dtype 到最简形态。
  3. **排查常见原因**：
     - workspace 大小不足（GetWorkSpaceInfo 参数有误）
     - 输入 tensor 的 format/dtype 与 ACLNN 期望不匹配
     - None 输入未正确处理（空指针传给了 ACLNN）
     - 输出 tensor 未正确分配（shape 计算有误）
  4. **对照 PTA 侧**：在相同输入下 PTA 是否也崩溃（区分 CANN bug vs 框架适配问题）。
- 给出排查结论与修复方案，修复后用最小用例 + 原始用例双重验证。

## 示例 13：多输出算子的反向实现（TupleGetItem）
**用户**：我的算子前向返回 `(output, softmax_max, softmax_sum)` 三个 tensor，反向 bprop 里怎么取？

**期望 agent 行为**：
- 说明反向输入规则：`out` 为 tuple 时用 `ib->TupleGetItem(out, index)` 取对应输出。
- `dout` 也是 tuple，用 `ib->TupleGetItem(dout, index)` 取对应梯度。
- 给出同类算子的 bprop 实现证据（按目标算子特征在仓库中搜索同类已接入算子）。
- 强调 Infer 需返回 `TupleShape`/`TupleType`，输出个数与前向一致。

## 示例 14：可选 None 输入的全链路处理
**用户**：我的算子 `actual_seq_lengths` 参数可以是 None，但 Infer/PyBoost/KBK 里不知道怎么处理。

**期望 agent 行为**：
- 先确认 None 语义：是"不传"还是"传空 tensor/零张量"。
- 检查 PTA 侧 C++ 代码：None 时是否构造 `at::Tensor()` 或 `at::empty({0}, ...)`。
- 给出全链路处理方案：
  - YAML：`default: None`
  - Infer：`IsNone()` 检测 → 不支持时抛错，支持时给回退逻辑
  - PyBoost/KBK：运行时 None 转为空 tensor 传给 ACLNN
  - 测试：覆盖 None 和非 None 两条路径
- 输出 PTA 对照代码片段作为证据。

## 示例 15：不同 layout 的 shape 推导差异
**用户**：算子支持 `layout="BSND"` 和 `layout="TND"`，Infer 里输出 shape 怎么按 layout 区分？

**期望 agent 行为**：
- 说明 layout 为编译期已知时可按分支推导；为 unknown 时回退动态秩/动态维。
- 给出相似算子的 Infer 实现证据（BSND vs TND 的分支处理）。
- 强调 UT 需覆盖所有 layout 组合 + unknown layout 场景。
- 提醒 PyBoost/KBK 也需按 layout 调整参数传递。

## 示例 16：反向显存优化（SetUnusedInputs）
**用户**：反向里有些输入只用 shape/type，不用 tensor 值，怎么优化显存？

**期望 agent 行为**：
- 说明 `SetUnusedInputs({i0, i2})` 的作用：Pynative 可提前释放不需要的正向 tensor。
- 给出判断规则：仅依赖 shape/type 或完全未使用的输入可标记。
- 注意事项：inplace 算子若反向用到更新前的 self，不能标记为 unused。
- 给出相似算子的 bprop 注册代码证据。

## 示例 17：按需求导（need_compute_grad_out + grad mask）
**用户**：算子有 5 个输入但用户可能只对其中 2 个求导，bprop 怎么优化？

**期望 agent 行为**：
- 说明 `need_compute_grad_out(index)` 的用法：逐个判断是否需要计算梯度。
- 需要时计算，不需要时返回 `ib->OutZeros(input)`。
- 若反向 ACLNN 大算子需要 grad_mask 参数，通过 `need_compute_grad_out()` 构造 mask 数组传入。
- 给出相似算子的实现证据。

## 示例 18：KBK GetWorkSpaceInfo 报错
**用户**：KBK kernel 在 `GetWorkSpaceInfo` 里报错，workspace 大小算不对。

**期望 agent 行为**：
- 说明 `GetWorkSpaceInfo` 在编译期调用，需从 `KernelTensor` 正确提取参数。
- 排查常见错误：
  - 参数类型转换错误（`int64_t` vs `int32_t`）
  - 参数顺序与 ACLNN 接口不一致
  - 可选参数为 None 时未正确处理
- 给出修复方案，输出修复前后代码对比。

## 示例 19：PTA 用多个 ACLNN 小算子组合实现，MS 怎么对标
**用户**：我看 PTA 的 `npu_foo` C++ 代码里调了 `aclnnA`、`aclnnB`、`aclnnC` 三个算子
串联实现前向，不是一个大算子。MS 这边怎么做？

**期望 agent 行为**：
- **不要假设可以找到一个对应的 aclnnFoo 大算子**，先确认 PTA 的调用链是事实。
- 按 `reference.md` §28 的方法：
  1. 从 PTA C++ 代码中提取完整调用链，标注每个 `aclnnXxx` 的用途和中间 tensor。
  2. 产出**覆盖盘点表**：逐个搜索 MS 仓库确认 aclnnA/B/C 的接入状态。
  3. 对缺失的子算子，评估工作量并规划实施顺序（叶子先、组合后）。
- 给出实施计划：
  - 阶段 1：补齐缺失子算子（各走 YAML→Infer→PyBoost→KBK→UT）。
  - 阶段 2：在 foo 的 customize 中串联多个 ACLNN 调用（参考 §29 组合模式）。
  - 阶段 3：分层验证（子算子级 → 中间 tensor 对齐 → 最终输出对齐）。
- 向用户输出盘点表和实施计划，确认后再动手。

## 示例 20：组合算子的中间结果与 PTA 不一致，怎么排查
**用户**：我按 PTA 的方式串联了 3 个 ACLNN 调用，最终输出和 PTA 对不上。

**期望 agent 行为**：
- **不要直接调参数**，先定位是哪个阶段出了问题：
  1. 在 PyBoost customize 中，每个子算子调用后临时 dump 中间 tensor。
  2. 在 PTA 侧同样位置 dump 中间 tensor（可在 C++ 中加临时打印或用 hook）。
  3. 逐阶段对比：找到第一个出现偏差的子算子调用。
- 常见原因：
  - 中间 tensor 的 shape/dtype 与 PTA 不一致（检查分配逻辑）。
  - 子算子的参数传递顺序或默认值与 PTA 不同。
  - 某个子算子在 MS 中的接入版本与 PTA 使用的 ACLNN 版本有差异。
  - None 参数在中间步骤的处理方式不同。
- 给出逐阶段对比的结果表格，定位到具体子算子后按单算子排障流程修复。

## 示例 21：复用原有原语，只需补 ACLNN kernel
**用户**：MS 已有 `Eye` 算子，我想给它加上 Ascend ACLNN 的支持。

**期望 agent 行为**：
- 搜索 MS 仓库确认 `Eye` Primitive 已存在，检查其 op_def YAML、Infer、functional/nn 接口。
- 完成接口分析五要素，确认功能/参数/数据类型一致，结论：复用原有原语。
- 告知用户：
  - 在现有 YAML 上加 `dispatch` 字段（`dispatch.Ascend: EyeAscend`），不新建 YAML
  - Infer/functional/nn/Tensor 接口不改
  - 只做 PyBoost customize + KBK kernel 注册 + aclnn_config 映射 + ST 测试
  - 可跳过 Step 3（Infer）、Step 7（导出/接口）
- 若签名不一致 → 转入示例 22/23 决策。

## 示例 22：已有原语但参数不兼容，需新增 `_ext`
**用户**：MS 已有 `ZerosLike` 算子，但 PTA 多了一个 `dtype` 参数，要怎么处理？

**期望 agent 行为**：
- 完成接口分析五要素，发现参数不兼容，判断不能对存量 `ops.zeros_like` 做兼容性修改。
- 告知用户走**新增原语 + `_ext` 后缀**策略：
  1. 新建 YAML `zeros_like_ext`（加 `_ext` 与原有区分）
  2. 新增 Infer → gen_ops.py → PyBoost → KBK → 接口层 → 测试/文档
  3. 新增接口需**重点评审**，评审后发接口变更邮件
- 若功能完全不一致且不能兼容 → 走 `ops.extend` 命名空间。

## 示例 23：已有原语需扩展参数，判断复用还是新建
**用户**：MS 有 `XxxAttention`，但 PTA 多了 `sparse_mode` 和 `block_table` 参数，该怎么办？

**期望 agent 行为**：
- 对比已有 `XxxAttention` 与 PTA 接口，产出参数差异表格。
- 给出决策分析（参考 `reference.md` §19.4）：
  - **若可兼容修改**（新参数可加默认值、不影响已有调用方和其他后端）→ 直接修改现有 YAML + Infer + 接口层。搜索 MS 仓库中相似算子的扩展方式作为参考。需走"功能扩展"评审。
  - **若不可兼容**（改了会破坏已有行为）→ 新增原语加 `_ext` 后缀。需走"新增接口"重点评审。
  - **若功能完全不一致** → 走 `ops.extend`。
- 明确说明各方案的评审要求和影响面，等用户确认后再实施。

## 示例 24：生成 Feature 文档
**用户**：帮我生成 nsa_compress_attention 的 Feature 文档。

**期望 agent 行为**：
- 从 `templates/feature-document.md` 复制模板，替换算子名。
- 根据已有的方案设计（Pre-B 产出）和实现代码，自动填充：
  - §1 背景描述、§2 标杆接口、§3 任务清单（逐项标注状态）、§4 功能与接口说明
  - §5 YAML 定义（从实际 YAML 文件中提取）、§6 约束与类型
  - §7-§12 的各实现/测试/异常章节
- §13 代码改动说明：自动列出所有新增/修改的文件路径。
- §14 验收报告：生成四张表框架（资料/功能/性能/安全编码），标注"待测试"状态。
- 提示用户：验收报告需在实际测试通过后更新自测结果。

## 示例 25：Feature 文档已有部分内容，需要补齐
**用户**：我之前方案评审时写了 Feature 文档的前半部分，现在开发完了帮我补齐后半部分。

**期望 agent 行为**：
- 读取已有的 Feature 文档，识别哪些章节已填写、哪些空缺。
- 根据实际代码实现，补齐 §5 YAML、§7 执行模式、§9-§12 等开发章节。
- 生成 §13 代码改动列表、§14 验收报告。
- 更新 §3 任务清单中每项的最终状态。
- 不覆盖用户已填写的内容，仅填充空缺部分。
