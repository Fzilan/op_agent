---
name: mindspore-aclnn-operator-devflow
description: Guides end-to-end ACLNN custom operator development and adaptation in MindSpore (PyBoost/Pynative + KBK/Graph paths), including YAML definitions, code generation, GeneralInfer, kernel registration, bprop wiring, tests (UT/ST), and docs. Use when the user mentions ACLNN, Ascend, 算子适配/算子开发, PyBoost, KBK, op_def YAML, gen_ops.py, bprop, or Ascend operator adaptation tasks.
---

# ACLNN 算子开发全流程（MindSpore 适配）

## 目标
把一个 Ascend 平台 ACLNN 算子在 MindSpore 里**端到端落地**：前向/反向、PyBoost(Pynative) 与 KBK(Graph) 双路径、动态 shape/rank、UT/ST、文档与导出，并完成必要的质量检查与验证。

## 使用方式（你要怎么用这份 skill）
- 当用户说"给MindSpore接入/适配一个 ACLNN 算子""给MS添加一个xxx接口，实现对标torch_npu""请通过skill帮我补一个npu算子""新增 xxx_op.yaml""PyBoost/KBK 怎么写""bprop 怎么注册""UT/ST 怎么补"等，直接按本 skill 的步骤推进。
- 输出时遵循：**明确说明检查了哪些文件/目录、给出代码证据片段、说明验证方式与结果**（与项目 `.cursorrules` 保持一致）。

> **⚠️ 必读规则：执行每个 Step 前，必须先用 Read 工具读取对应的 workflow 文件**
> （`workflows/XX-xxx.md`）获取详细步骤、约束和成功标准。
> 仅看 SKILL.md 的摘要**不足以确保正确执行**——workflow 文件中包含关键禁令和
> 反模式说明，跳过阅读会导致交付件缺失或实现错误。

## 执行流程总览

ACLNN 算子开发遵循 **Pre + 10 步流程**，从前置分析到转测交付。
**核心决策：Pre-B 阶段确定走"路径 1（自动生成）"还是"路径 2（Customize）"，决定后续工作量。**

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ Pre-A/B/C         │ ──▶ │ 1. YAML 定义      │ ──▶ │ 2. 代码生成       │
│ 前置检查/方案/    │     │ op_def/api_def    │     │ gen_ops.py        │
│ 调用链盘点        │     │ function_doc      │     │ ★路径决策影响      │
│ ★路径决策         │     │ ★dispatch 配置    │     │   生成范围         │
└───────────────────┘     └───────────────────┘     └───────────────────┘
                                                             │
           ┌─────────────────────────────────────────────────┘
           ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ 3. GeneralInfer   │ ──▶ │ 4. PyBoost        │ ──▶ │ 5. KBK            │
│ 形状/类型推导     │     │ Pynative 路径     │     │ Graph 路径        │
│ +InferValue 可选  │     │ 路径1:自动生成    │     │ 路径1:自动注册    │
│ 两条路径都需要    │     │ 路径2:手写customize│     │ 路径2:手写kernel  │
└───────────────────┘     └───────────────────┘     └───────────────────┘
                                                             │
           ┌─────────────────────────────────────────────────┘
           ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│ 6. BPROP 注册     │ ──▶ │ 7. 导出与占位     │ ──▶ │ 8. 测试           │
│ 反向图接线        │     │ ops 包导出        │     │ UT + ST           │
│ 梯度路径          │     │ 非 Ascend 占位    │     │ 精度/显存对齐     │
└───────────────────┘     └───────────────────┘     └───────────────────┘
                                                             │
           ┌─────────────────────────────────────────────────┘
           ▼
┌───────────────────┐     ┌───────────────────┐
│ 9. 文档           │ ──▶ │ 10. 转测交付      │
│ EN + CN 一致      │     │ Feature + PR      │
└───────────────────┘     └───────────────────┘
```

### 两条接入路径（Pre-B 必须决策，详见 `reference.md` §2.3）

| | **路径 1：自动生成** | **路径 2：Customize** |
| --- | --- | --- |
| **判断条件** | API 参数能原样透传给 ACLNN | 调用 ACLNN 前需要参数预处理 |
| **YAML dispatch** | `enable: True`，**不写** `Ascend` | `enable: True` + `Ascend: XxxAscend` |
| **PyBoost/KBK** | 编译自动生成（跳过 Step 4/5 手写） | 开发者手写 customize 文件 |
| **Step 4/5** | **验证**即可（确认自动生成产物正确） | **必须手写**实现 |
| **典型算子** | abs、mul、trunc、xlogy | dense_lightning_indexer、conv2d_ext |
| **gen_ops.py 作用** | 生成完整 PyBoost/KBK 调用+注册代码 | 生成包装代码，调用 Customize 类 |

### Workflow 执行清单

使用此 skill 开发 ACLNN 算子时，**创建 TODOLIST**，按顺序执行以下 workflow。
**注意：标记 `🔒不可跳过` 的步骤无论什么场景都必须执行，不能通过条件跳步裁剪。**
**注意：标记 `⛔ HARD GATE` 的地方必须完成前置产出后才能继续，否则停下等用户确认。**

- [ ] **[Pre](workflows/00-pre-checks.md)**`🔒不可跳过`：前置检查（Pre-A 存量检查 + Pre-B 方案设计 + Pre-C 调用链盘点）
  - 输入：算子名称、PTA 对标接口
  - 输出：存量检查结果、接口分析结论、方案文档、调用链盘点表（组合场景）
  - **必须产出**：使用 `templates/pta-analysis-report.md` 模板生成 PTA 源码审查报告，输出给用户
  - 组合场景还需产出：`templates/aclnn-callchain-inventory.md` 调用链盘点表
- [ ] **Feature 文档初始化** `🔒不可跳过`：从 `templates/feature-document.md` 复制模板，填写 §1-§4、§6、§8
  - 这是评审和转测交付的**必须产物**，必须在 Pre-B 完成后、Step 1 开始前生成并输出给用户
  - 后续每完成一个 Step，回填 Feature 文档对应章节

> **⛔ HARD GATE 1**：进入 Step 1 之前，必须确认以下**文件已生成到工作区**：
> 1. PTA 源码审查报告 .md 文件（按模板填充，Write 生成，告知用户路径）
> 2. Feature 文档 .md 文件（§1-§4、§6、§8 已填写，Write 生成，告知用户路径）
>
> **"交付给用户" = 生成 .md 文件 + 告知路径。做了分析但没生成文件 = 未产出。**
> **缺少任何一项则停下，不得继续。**

- [ ] **[Step 1](workflows/01-yaml-definition.md)**：YAML 定义 → 回填 Feature §5
  - 输入：Pre-B 方案设计、PTA 源码审查结果
  - 输出：op_def + api_def + function_doc YAML 文件
- [ ] **[Step 2](workflows/02-code-generation.md)**：代码生成
  - 输入：YAML 文件
  - 输出：gen_ops.py 运行成功、骨架代码
- [ ] **[Step 3](workflows/03-general-infer.md)**：GeneralInfer + InferValue（可选） → 回填 Feature §9/§10
  - 输入：YAML、PTA 输出 shape 逻辑
  - 输出：Infer 实现文件
- [ ] **[Step 4](workflows/04-pyboost.md)**：PyBoost（Pynative） → 回填 Feature §7
  - **路径 1**：跳过手写，验证自动生成产物正确即可
  - **路径 2**：手写 customize 实现文件（参数转换 + ACLNN 调用）
  - 输入：YAML、ACLNN 调用细节
  - 输出：customize 实现文件（路径 2）/ 验证通过（路径 1）
- [ ] **[Step 5](workflows/05-kbk.md)**：KBK（Graph） → 回填 Feature §7
  - **路径 1**：跳过手写，验证自动注册正确即可
  - **路径 2**：手写 kernel 文件（GetWorkSpaceInfo + Launch + 注册）
  - 输入：YAML、PyBoost 参考
  - 输出：kernel 实现文件 + 注册（路径 2）/ 验证通过（路径 1）
- [ ] **[Step 6](workflows/06-bprop.md)**：BPROP 注册 → 回填 Feature §11
  - 输入：derivatives.yaml 分析、反向 kernel
  - 输出：bprop builder 注册代码
- [ ] **[Step 7](workflows/07-export.md)**：导出与占位
  - 输入：算子实现
  - 输出：ops 包导出、接口文件、非 Ascend 占位；如涉及接口重载见 `reference.md` §25
- [ ] **[Step 8](workflows/08-testing.md)**：测试 → 回填 Feature §12
  - 输入：全部实现、PTA 对标
  - 输出：C++ UT（必须新建） + Python ST（**必须注册 OpInfo**）
  - 🚫 **Python ST 禁令**：禁止在 `tests/st/` 下新建独立 `test_xxx.py` 测试文件。
    所有新增算子的 ST **必须**通过在 `op_database.py` 注册 OpInfo 完成。
    详见 `workflows/08-testing.md` Step 2。
  - **⚠️ 完成前必须逐项确认两类测试的产出状态，不允许只写一类就跳过**
- [ ] **[Step 9](workflows/09-docs.md)**：文档
  - 输入：接口实现
  - 输出：英文 function_doc（Step 1 已创建，此处完善）+ **中文 RST（公开 API 必须）**
  - **⚠️ 英文 doc YAML 不等于文档步骤完成——中文 RST 是独立产物，最容易遗漏**
  - **⚠️ mint/ops/nn/Tensor 公开接口不得跳过此步骤**（仅内部算子可跳过，见条件跳步表）

> **⛔ HARD GATE 2**：代码开发全部完成后（Step 1-9），必须完成以下产出才能进入 Step 10：
> 1. Feature 文档定稿（§13 代码改动 + §14 验收报告 + §3 任务状态更新）→ **回填到 Feature.md 文件**
> 2. Step 9 文档产出确认（公开 API 必须有中文 RST）
>
> **Feature 定稿 = Feature.md 文件全部 14 章节已填写完毕，告知用户"Feature 文件已更新完毕"。**
> **缺少任何一项则停下，不得继续。**

- [ ] **Feature 文档定稿** `🔒不可跳过`：补齐 §13（代码改动）、§14（验收报告）、更新 §3 任务状态
  - 即使 Step 9/10 被跳过或推迟，Feature 文档也必须在代码开发完成后补齐并输出给用户
- [ ] **[Step 10](workflows/10-delivery.md)**：转测交付
  - 输入：全部代码/测试/文档 + **完整 Feature 文档**
  - 输出：Feature 文档 + PR + 验收 checklist

### 条件跳步表（按实际场景裁剪步骤）
> "路径 1/2"定义见 `reference.md` §2.3 和上方对比表。

| 场景 | 可跳过/精简的步骤 | 说明 |
| --- | --- | --- |
| **路径 1（参数直通、自动生成）** | Step 4/5 **跳过手写** | 编译自动生成 PyBoost/KBK，只需验证产物正确 |
| **路径 2（参数需预处理）** | Step 4/5 必须手写 | 手写 customize PyBoost + KBK kernel 文件 |
| 仅前向，无反向需求 | 跳过 6(bprop) | YAML 也只需前向一份 |
| **仅限内部算子**（不在 `__all__` 导出、无 mint/Tensor 接口） | 跳过 9(文档) | **mint/ops/nn/Tensor 公开接口禁止跳过 Step 9** |
| 不需要对标 PTA 精度 0 偏差 | checklists 7/8 降级 | 只做 rtol/atol 级别对比即可 |
| 不涉及 functional/nn/Tensor 导出 | 步骤 7 精简 | 只做 Primitive 层导出 |
| PTA 直连单个 aclnnXxx 大算子 | 跳过 Pre-C | 无需调用链分析与子算子盘点 |
| PTA 用多个小算子拼接组合 | Pre-C 必做，4 用 C++ API 拼接，5 用 Meta DSL | 详见 `reference.md` §22/§23 |
| PTA 为 View 算子（纯 shape/strides 变换） | 4 走 View strides calc，5 走 host kernel | 详见 `reference.md` §26 |

> **🔒 不可跳过项**：无论什么场景，以下步骤**绝对不能跳过**：
> - **Feature 文档初始化**（Pre-B 完成后）
> - **Feature 文档定稿**（代码开发完成后）
>
> Feature 文档是算子评审和转测交付的必须产物，即使跳过 Step 9（文档）或推迟 Step 10（交付），
> Feature 文档本身也必须生成。

## 关键约束（必须遵守）

> **以仓库实际代码为准，不要盲从文档流程。**
> 本 skill 的流程、模板、命名约定都可能因 MindSpore 版本迭代而过时。
> 发现文档描述与仓库现状不一致时，**以仓库现状为准**。

- **仓库现状 > 文档描述**：目录结构、注册宏、Infer API、生成脚本行为、CMake 配置等
  都可能随版本变化。确认当前版本的真实用法，不要照搬文档中可能已过时的示例。
- **不要**在 `~/.cursor/skills-cursor/` 创建 skill（这是内置技能目录）；项目内用 `.cursor/skills/`。
- **SKILL.md 控制在 500 行内**：把细节放到 `reference.md` / `examples.md`。
- **不使用 Windows 风格路径**：文档里路径统一用 `a/b/c`。
- **行长不超过 120**；推导/错误处理遵循项目规范（见根目录 `.cursorrules`）。

### 开发纪律（内嵌，不依赖外部配置）

> 以下三条纪律直接影响算子交付质量，已内嵌至本 skill 中，
> 无需依赖项目级 `.cursorrules` 或其他外部配置文件。

**D1. 跨文件变更一致性（Change Propagation）**

算子实现涉及 8 类代码文件（op_def YAML、api_def YAML、GeneralInfer C++、PyBoost C++、
KBK kernel C++、bprop C++、Python 接口层、中英文文档），同一参数名/类型/默认值/约束
会在上述文件中多处引用。任何单点修改必须执行以下流程：
1. **定位变更项**：明确本次修改的具体修改点（如参数名 `actual_seq_len`、dtype 约束 `float16/bfloat16`、dispatch 配置 `Ascend: XxxAscend`）
2. **全量检索**：在同一批操作中，对整个算子相关目录执行关键词搜索，定位所有引用点
3. **同步更新**：逐一修改所有引用点，确保一致
4. **残留验证**：修改完成后再次搜索旧值，确认零残留

**D2. 问题定位与经验沉淀（Defect Root-Cause & Codification）**

当出现以下情形之一时，必须在修复后执行定位与沉淀：
- 同类问题重复出现（≥2 次）
- 修复过程中发现连锁遗漏
- 问题根因位于流程或规范层面而非代码层面

沉淀流程：
1. **定位环节**：确认问题发生在哪个开发环节——YAML 定义 / Infer 推导 / kernel 实现 / 接口导出 / 测试覆盖 / 文档同步
2. **逆向复盘**：确认修复所依赖的关键前置信息，评估该信息是否已在 skill 文档中显式记录
3. **经验规则化**：将教训转化为 `checklists.md` 中可独立判定"通过/不通过"的检查项

**D3. 独立视角交叉验证（Cross-Perspective Review）**

每个开发步骤完成后，必须从至少两个独立视角进行验证：
- **实现视角**：代码逻辑是否正确、与相似算子实现是否一致
- **评审视角**：对照 `checklists.md` 对应章节逐项核对，每项须可判定"是/否"
- 两个视角的验证结果均需在"验证闭环"模板中显式记录

## 开发前的"信息收集"清单（缺一不可）
在动手改代码前，先收集/确认（能从上下文推断就直接推断）：
- **算子名**：对外 API 名、内部 Primitive 名、Ascend kernel 调度名。
- **接入路径决策**：分析 MS API 参数是否可直通 ACLNN——**路径 1（自动生成）** vs **路径 2（Customize）**
  （详见 `reference.md` §2.3）。此决策直接影响 YAML 的 `dispatch.Ascend` 配置和 Step 4/5 的工作量。
- **ACLNN 对接对象**：具体接入哪个 `aclnnXxx`（含可能的变体/Grad）以及对应 ACLNN 文档/约束。
  **注意**：PTA 可能不是直连单个大算子，而是用多个小算子拼接组合；需从 C++ 代码中提取完整调用链
  并盘点 MS 侧覆盖情况（详见 Pre-C 与 `reference.md` §22），组合实现采用 C++ API 拼接（PyBoost）+ Meta DSL（KBK）（详见 `reference.md` §23）。
- **PTA 对标与源码审查（必做）**：`torch_npu` 对标接口的文档/行为，**同时**直接审查 op-plugin
  仓库中的三类关键文件（详见 `reference.md` §19），从代码中提取真实的接口签名、反向接线、ACLNN 调用细节：
  - `op_plugin/config/op_plugin_functions.yaml`：函数签名、参数类型/默认值、返回值结构
  - `op_plugin/config/derivatives.yaml`：反向注册、哪些输入可微、grad 函数参数传递
  - `op_plugin/ops/opapi/XxxKernelNpuOpApi.cpp`（含 Grad）：实际 ACLNN 调用、参数预处理、输出构造
  - **文档与源码一致时**：结合两者参考，高效推进。
  - **发现代码与文档不一致时**：不要猜，整理差异清单交给用户，
    让用户找 ACLNN/PTA 算子开发接口人确认以哪边为准，拿到结论后再继续。
- **版本矩阵**：torch/torch_npu/CANN 版本（不同版本支持范围会变动，必须先记录再对齐）。
- **文档不足时的补证**：必要时生成"PTA 探测脚本"跑出真实支持范围，再回填约束到 YAML/Infer/测试/文档。
- **输入/输出**：rank、layout（BSND/TND/PA_BSND 等）、dtype、可选输入/关键字参数、返回值个数。
- **是否需要反向**：梯度输入/输出定义、哪些输入需要梯度、是否有非张量参数。
- **动态支持**：动态 shape / 动态 rank / unknown value 的推导策略。
- **对标实现**：torch_npu / 参考 feature 文档 / 仓库中已接入的同类算子（见下方"相似算子查找策略"）。

### 相似算子查找策略（`reference.md` §2.4）
不要硬编码某几个算子名作为参照。正确做法：
1. **判断功能/算法类别**：目标算子属于哪个族？（Attention / Loss / Norm / Optimizer / 激活 / 逐元素算术 / Reduce / 矩阵运算 / 索引gather / 卷积池化 / 通信并行 等）
2. **确定技术特征**：在同族内用输入布局、ACLNN 对接方式、是否有反向、接口形态、参数特殊性进一步筛选。
3. **在仓库中搜索同类**：按族名 + 技术特征在 `op_def/yaml/`、`customize/`、`api_def/` 中 grep。
4. **选 2-3 个最匹配算子**，对照其各目录的代码风格和结构。
5. 相似算子只用于确认**目录结构、宏名、注册方式、测试写法**；功能逻辑以 PTA 源码 + ACLNN 文档为准。

## 验证闭环（每一步都要给证据）`🔒不可跳过`

每完成一个 Step，**必须**使用以下模板向用户展示执行报告（不可省略、不可合并、不可延后）。
**这是对用户的强制交付物，不是内部记录——必须在消息中直接输出给用户看到。**

```text
━━━ Step X 执行报告 ━━━

执行依据（我依据 skill 的哪条要求来执行）：
- workflow 文件：workflows/XX-xxx.md
- 对应的 skill 要求：（引用 SKILL.md / workflow 中的具体条目）
- 本步骤的成功标准：（从 workflow 成功标准中摘录）

我做了什么（产出清单）：
- ...

关键证据（代码片段/文件路径/搜索结果）：
- ...
- 对照了哪个已有算子的实际代码：...

验证结果：
- ...

成功标准逐项核对：
- [ ] 标准1：✅/❌
- [ ] 标准2：✅/❌
- ...

遗留问题/风险与下一步：
- ...
```

> **为什么必须展示执行依据？**
> 历史教训表明，不追溯执行依据时 agent 容易"凭感觉"跳过关键约束。
> 强制引用 skill 条目可以确保每一步都有据可查，用户可以及时发现偏差。

## 卡住时的排障升级路径
当某个步骤反复失败或无法推进时，按以下优先级逐步升级：
1. **回退确认前提**：重新检查上一步的输出是否正确（YAML 是否合规、Infer 是否编译通过等）。
2. **对照相似算子**：找仓库中最相似的已有算子实现，逐行对比差异点。
3. **检查版本/环境**：确认 CANN/torch_npu/MindSpore 版本是否匹配，API 是否存在于当前版本。
4. **缩小复现范围**：构造最小可复现用例，排除无关因素。
5. **向用户确认**：明确告知卡在哪、已尝试了什么、需要用户提供什么信息（日志/环境/权限/设备运行结果）。**不要默默跳过，必须暂停等用户回复后再继续。**
6. **建议提 issue**：如果判断是 CANN/框架 bug，建议用户在社区提交 issue 记录。

> **禁止**：不要在卡住时反复尝试相同的修改方式，不要跳过失败步骤继续后续步骤。

## 常见坑快速规避

### 🚫 高频致命反模式（必须逐条确认未触犯）

> 以下反模式在历史开发中反复出现，每个都直接导致交付被打回。
> **每次开发完成后，逐条自查；触犯任何一条即为不合格。**

1. **Pre 报告未输出**：Pre-B 完成后没有用模板生成 PTA 分析报告给用户。
   → 必须用 `templates/pta-analysis-report.md` 模板产出报告。
2. **Feature 文档未生成**：整个开发过程中从未初始化或定稿 Feature 文档。
   → Feature 初始化和定稿都是 🔒不可跳过项，必须生成并输出给用户。
3. **bprop 标量 unknown 直接报错退出**：在 `ContainsValueAny()` 为 true 时
   用 `MS_EXCEPTION` 抛异常而非构建运行时分支。
   → 必须用 `ib->Conditional(cond, true_branch, false_branch)` 处理。
   编译期值已知时走 C++ if/else 分支；unknown 时走 `Conditional` 运行时分支。
   参考 `ReduceStd` bprop 的写法。详见 `workflows/06-bprop.md` Step 5。
4. **Python ST 写成独立测试文件**：在 `tests/st/` 下新建了 `test_xxx.py` 独立脚本。
   → 新增算子的 ST **必须**通过在 `op_database.py` 中注册 OpInfo 完成，
   不允许新建独立测试文件（旧框架写法，新增算子不接受）。
   详见 `workflows/08-testing.md` Step 2。
5. **公开 API 跳过 Step 9 文档**：mint/ops/nn/Tensor 接口未创建中文 RST。
   → 公开 API 的中文 RST 是独立于英文 doc YAML 的必须产物。
   仅内部算子（不在 `__all__` 导出）可跳过。
6. **未读 workflow 文件就执行**：只看 SKILL.md 摘要，没有 Read 对应的
   `workflows/XX-xxx.md` 文件就开始写代码。
   → 每个 Step 开始前必须先读对应 workflow 文件，获取完整约束和成功标准。

### 一般注意事项
- **YAML/代码生成**：缺 `py_method`、keys 结构不符会导致 `gen_ops.py` 报错。
- **InferInfo API**：按项目已有用法写（如 `GetScalarValueWithCheck` / `GetArrayValue` / `HasUnknownValue` / `IsNone`），不要臆造 API。
- **Infer 职责**：只做推导，不做运行时合法性校验（合法性让 ACLNN/运行时处理）。
- **前后向拆分**：前向/反向、PyBoost/KBK 通常都要分文件，注册也各自独立。
- **Windows 编码**：英文 YAML 文档避免夹杂中文（可能触发编码问题）；中文放 RST。

## 额外资料（按需读取）

### skill 自带
- **Workflow 详细步骤**：`workflows/` 目录（每步的 Goal/Input/Output/Steps/成功标准）
- **模板文件**：`templates/` 目录
  - PTA 源码审查报告：`templates/pta-analysis-report.md`
  - ACLNN 调用链盘点表：`templates/aclnn-callchain-inventory.md`
  - AI 操作全过程记录模板：`templates/ai-operation-record.md`
- **知识参考与代码骨架**：`reference.md`
- **可复制清单**：`checklists.md`
- **触发样例**：`examples.md`
- **溯源映射**：`traceability.md`
- **workflow 核对报告**：`workflows/reconciliation-report.md`
- **PTA 探测脚本模板**：`scripts/probe_pta_sparse_flash_attention.py`

### 原始来源文档（不随 skill 分发，如项目中有则可参考）
> 以下文档是本 skill 的知识来源，其内容已提炼到上述 skill 文件中。
> 如果你的工作区中有 `算子流程/` 目录，可以按需查阅原始文档获取更多上下文。

- 开发经验与排障：`算子流程/ACLNN_nsa_compress_适配开发经验.md`
- 方案/交付/评审全流程：`算子流程/算子知识库_脱敏版/.../Aclnn算子对接开发整体流程.md`
- 接入示例：`算子流程/算子知识库_脱敏版/.../aclnn开发示例.md`
- 需求/接口参考：`算子流程/参考feature.md`
