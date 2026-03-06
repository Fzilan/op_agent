# ACLNN 算子开发 Skill 解耦重构方案

> **文档状态**：提案讨论稿
> **日期**：2026-03-06
> **目的**：解决当前 skill 耦合度深、难量化评估、步骤无法独立操作的问题

---

## 一、问题背景

### 同事反馈的核心痛点

1. **耦合度深**：想做某一步（如只写 YAML 或只补测试），必须先理解整个 SKILL.md 的全局上下文
2. **难以量化评估**：每步做完后不知道"做对了没有"，验证靠主观叙述而非可检测的标准
3. **步骤无法独立操作**：不能单独执行或评估某一步，必须从头到尾走完整流程

### 当前架构的结构性原因

```
当前文件结构:
SKILL.md (343行, 同时承担6种职责)
├── 核心行为准则 D1-D4 (给AI的执行约束)
├── 流程编排 (Pre + 10步的串行依赖)
├── 回复与执行规范 (AI输出格式要求)
├── 信息收集清单 (开发前准备)
├── 验证闭环模板 (主观叙述式)
└── 排障指南

reference.md (1535行, 31章节, 被各workflow零散引用)
checklists.md (400行, 与workflow手工映射, 一改容易漏)
workflows/ (11个文件, 每个都隐式依赖SKILL.md全局上下文)
```

**根因**：`SKILL.md` 是一个"上帝对象"——它同时是行为准则、流程编排器、知识库入口和执行规范。所有 workflow 都隐式依赖它的全局上下文，导致无法单独读取和执行任何一个步骤。

---

## 二、方案选型

我们评估了两种模块化路径：

### 方案 A：拆成多个独立 skill（未采纳）

把每个步骤拆成独立 skill（如 `aclnn-yaml/SKILL.md`、`aclnn-infer/SKILL.md` 等）。

**放弃原因**：
- 共享知识（路径 1/2 决策、命名规范、验证模式）会在 11 个 skill 中重复，改一处漏一处
- 端到端流程碎片化，用户需要知道"下一步用哪个 skill"
- 维护成本高：11+ 个 SKILL.md 文件

### 方案 B 增强版：一个 skill 内部模块化 + 单步入口（采纳）

保持一个 skill，但做三件事：
1. **SKILL.md 瘦身**为路由器（343 行 -> 约 120 行）
2. **每个 workflow 自包含**（有独立的输入/输出契约和验收门）
3. **三层验收门**提供可量化的评估标准

**采纳原因**：
- 共享知识集中在 `shared/` 目录，不重复
- 每步可独立执行，也可端到端串联
- 向后兼容：旧流程仍可用

---

## 三、目标架构

### 3.1 文件结构

```
mindspore-aclnn-operator-devflow/
├── SKILL.md                     # 瘦身为路由器 (~120行)
│                                #   - 单步入口路由表
│                                #   - AI 加载协议
│                                #   - 流程总览图
│                                #   - 路径1/2对比表
│
├── shared/                      # 共享知识（提取自SKILL.md）
│   ├── conventions.md           # AI 完整版：行为准则 + 路径决策 + 命名规范 + 验证模式
│   ├── dev-guide.md             # 人类精简版：开发规范摘要，评审时参考
│   └── gate-template.md         # 验收门通用模板定义
│
├── workflows/                   # 每个 workflow 自包含（改造后）
│   ├── 00-pre-checks.md         #   统一结构：
│   ├── 01-yaml-definition.md    #     前置条件（接口契约）
│   ├── 02-code-generation.md    #     目标与产出契约
│   ├── 03-general-infer.md      #     执行步骤
│   ├── 04-pyboost.md            #     验收门精简版（Gate 1+2）
│   ├── 05-kbk.md                #     下一步建议
│   ├── 06-bprop.md
│   ├── 07-export.md
│   ├── 08-testing.md
│   ├── 09-docs.md
│   └── 10-delivery.md
│
├── checklists/                  # 详细验收清单（按步骤1:1拆分）
│   ├── pre-checks.md            #   每个文件包含完整的 Gate 1/2/3
│   ├── yaml-definition.md
│   ├── general-infer.md
│   ├── ...
│   └── delivery.md
│
├── reference.md                 # 保留，新增反向索引
├── templates/                   # 保留不变
├── scripts/                     # 保留 + 新增 Gate 1 自动验证脚本
└── docs/                        # 保留不变
```

### 3.2 架构变化对比

**当前**：SKILL.md 是中心节点，所有 workflow 依赖它的全局上下文

```
SKILL.md (343行, 6种职责)
    ↓ 全局依赖
workflows/ (每个都要先读 SKILL.md 全文)
    ↔ 手工映射
checklists.md (400行大文件)
```

**改造后**：SKILL.md 是薄路由器，workflow 通过接口契约解耦

```
SKILL.md (~120行, 路由器)
    ↓ 指向
shared/ (共享知识, 按需加载)
    ↓ 引用
workflows/ (每个自包含, 可独立执行)
    ↓ 对应
checklists/ (1:1拆分, Gate 1/2/3 三层)
    ↗ 按需引用 section
reference.md (+ 反向索引)
```

### 3.3 单步入口机制

这是解决"步骤无法独立操作"的关键设计。

**SKILL.md 中的路由表**：

| 你想做什么 | 先读 | 再读 |
| --- | --- | --- |
| 完整流程（端到端） | SKILL.md + shared/conventions.md | workflows/00 开始依次推进 |
| 只做 YAML 定义 | shared/conventions.md | workflows/01-yaml-definition.md |
| 只做 Infer 推导 | shared/conventions.md | workflows/03-general-infer.md |
| 只做 PyBoost | shared/conventions.md | workflows/04-pyboost.md |
| 只做测试 | shared/conventions.md | workflows/08-testing.md |
| 检查某步验收标准 | — | checklists/{step}.md |

**AI 加载协议**：
- **完整流程**：读 SKILL.md + shared/conventions.md，然后按步骤逐个读 workflow
- **单步执行**：只读 shared/conventions.md + 对应 workflow，不读其他步骤
- **评估/评审**：只读对应 checklists/{step}.md

---

## 四、Workflow 自包含改造

### 4.1 统一结构

每个 workflow 文件改造为以下结构：

```markdown
# Step X: {名称}

## 前置条件
> 本步骤可独立执行。执行前确认以下条件已满足：
- [ ] 已读 shared/conventions.md（了解路径决策和命名规范）
- [ ] 来自 Step Y 的产出已就绪：{具体文件路径}
- [ ] 路径决策已确定：路径 1 -> {本步做什么} / 路径 2 -> {本步做什么}

## 目标与产出契约
- 目标：{一句话}
- 产出文件：
  - {具体路径模式 1}
  - {具体路径模式 2}
- 完成标志：{什么条件算这步做完了}

## 执行步骤
（当前 workflow 内容，去掉对 SKILL.md 全局上下文的引用，
  reference.md 引用精确到节："详见 reference.md §X.Y"）

## 验收门（精简版）
### Gate 1: 产物存在性 [自动]
- [ ] ...
### Gate 2: 结构正确性 [半自动]
- [ ] ...
> Gate 3（语义完整性）见 checklists/{step}.md

## 下一步（建议）
```

### 4.2 核心改动点

| 改动 | 当前 | 改造后 |
| --- | --- | --- |
| 前置条件 | "你需要理解 SKILL.md 全文" | "你需要 Step Y 的这个具体产出文件"（接口契约） |
| 产出定义 | "完成 Step X" | "产出这些文件、满足这些条件"（产出契约） |
| 验收标准 | 分散在 SKILL.md 和 checklists.md | workflow 内嵌精简版 + 独立 checklist 详细版 |
| reference 引用 | "见 reference.md" | "详见 reference.md §4.2"（精确到节） |

---

## 五、评估体系：三层验收门

### 5.1 设计理念

当前的验收标准有两个问题：
1. 验证闭环模板是主观叙述（"我检查了…""关键证据…"），无法判断"是否真的做对了"
2. checklists.md 的 `[MUST]/[SHOULD]` 标记混合了"检测方式"和"重要程度"两个维度

改造后按**检测方式**分三层，每层有明确的通过标准：

| 层级 | 检测方式 | 内容 | 通过标准 |
| --- | --- | --- | --- |
| **Gate 1** | 自动检测 | 产物是否存在、关键字段是否非空 | 全部通过 |
| **Gate 2** | 半自动检测 | 跨文件一致性、编译/导入通过 | 全部通过 |
| **Gate 3** | 人工判定（Yes/No） | 与 PTA 对齐、边界覆盖、语义正确 | 所有 [MUST] 项为 Yes |

### 5.2 各步骤 Gate 1 示例

Gate 1 是最容易自动化的，可以写脚本一键检查：

| 步骤 | Gate 1 检查项 | 验证方式 |
| --- | --- | --- |
| Step 1 YAML | op_def yaml 文件存在、dispatch 字段存在、function_doc 非空 | 脚本检查文件和字段 |
| Step 3 Infer | .h/.cc 文件存在、注册宏存在 | grep `REGISTER_PRIMITIVE_OP_INFER_IMPL` |
| Step 4 PyBoost | 路径 1: 自动生成文件存在；路径 2: customize 文件存在 | 脚本检查目录 |
| Step 8 Testing | C++ UT 文件存在、Python ST 文件存在、含固定种子 | 脚本检查文件 + grep `seed` |

### 5.3 验收门模板

每个 checklist 文件使用统一格式：

```markdown
## 验收门 — Step X: {名称}

### Gate 1: 产物存在性 [自动检测]
> 全部通过才进入 Gate 2。
- [ ] {检查项} — 验证: {命令}
通过标准: 全部勾选

### Gate 2: 结构正确性 [半自动检测]
> 检查跨文件一致性、编译/导入是否通过。
- [ ] {检查项} — 验证: {命令}
通过标准: 全部勾选

### Gate 3: 语义完整性 [人工判定, Yes/No]
> 每项回答 Yes 或 No，不允许"部分是"。
- [ ] [MUST] {检查项}
- [ ] [SHOULD] {检查项}
通过标准: 所有 [MUST] 项为 Yes

### 步骤评估摘要
| Gate | 项数 | 通过数 | 状态 |
| --- | --- | --- | --- |
| Gate 1 (自动) | N | ? | PASS/FAIL |
| Gate 2 (半自动) | N | ? | PASS/FAIL |
| Gate 3 MUST (人工) | N | ? | PASS/FAIL |
| Gate 3 SHOULD (人工) | N | ? | 参考 |
总体判定: Gate 1 + Gate 2 + Gate 3 MUST 全过 = PASS
```

---

## 六、shared/ 目录设计

### 6.1 conventions.md（AI 完整版）

从 SKILL.md 迁入以下内容，供 AI 执行时加载：

- 核心行为准则 D1-D4（认知红线、执行协议、交付标准、进化策略）
- 回复与执行规范（TodoWrite、验证闭环、操作记录产出要求）
- 路径 1/2 决策详细逻辑（判断条件、YAML dispatch 配置差异、对后续步骤的影响）
- 跨文件变更一致性（D1）的 8 类文件列表
- 排障升级路径
- 相似算子查找策略
- 当发现用户采用了与 skill 不同做法时的处理方式

### 6.2 dev-guide.md（人类精简版）

同事做人工评审或自己开发时看的精简版（约 50 行），不包含 AI 执行细节：

- 路径 1/2 判断方法（一段话 + 对比表）
- 命名规范（PascalCase/snake_case 对应关系）
- 跨文件一致性要点（参数改了哪 8 处要同步）
- 评审重点清单（告诉评审者"关注什么"）
- 常见遗漏 Top 5

---

## 七、SKILL.md 瘦身细节

### 7.1 从 SKILL.md 移出的内容及去向

| 内容 | 当前行数(约) | 移动到 |
| --- | --- | --- |
| 核心行为准则 D1-D4 | 80 行 | shared/conventions.md |
| 回复与执行规范 | 20 行 | shared/conventions.md |
| 信息收集清单 | 25 行 | workflows/00-pre-checks.md |
| 验证闭环模板 | 25 行 | shared/gate-template.md（改造为三层验收门） |
| 排障升级路径 | 30 行 | shared/conventions.md |
| 常见坑快速规避 | 6 行 | 各 workflow 内嵌到对应步骤 |
| 额外资料列表 | 22 行 | shared/conventions.md 尾部 |
| Workflow 执行清单 | 45 行 | 精简保留在 SKILL.md（作为流程总览） |
| 条件跳步表 | 20 行 | 精简保留在 SKILL.md |

### 7.2 SKILL.md 保留的内容（约 120 行）

1. **元信息**（frontmatter，5 行）
2. **单步入口路由表 + AI 加载协议**（30 行）
3. **流程总览图**（ASCII 图，30 行）
4. **共享契约摘要**（路径 1/2 对比表 + 条件跳步表，40 行）
5. **额外资料索引**（15 行）

---

## 八、reference.md 处理

**不拆散**——reference.md 作为"百科全书"有独立存在价值。

**改进**：
1. 在文件开头添加**反向索引**：从 workflow step 到需要读的 sections

```markdown
## 反向索引：按步骤查阅

| Workflow Step | 需要读的章节 |
| --- | --- |
| Pre-A/B/C | §1, §2.3, §2.4, §19.4, §25, §28 |
| Step 1 YAML | §2.1, §2.2, §2.3, §24.1 |
| Step 3 Infer | §4, §26, §27 |
| Step 4 PyBoost | §5, §29.1 |
| Step 5 KBK | §6, §29.2 |
| Step 6 BPROP | §7, §14 |
| Step 8 Testing | §8, §17, §22, §29.4 |
| Step 9 Docs | §9, §11 |
```

2. 各 workflow 内的引用精确到节："详见 reference.md §4.2"，不再写"见 reference.md"

---

## 九、实施计划

### 两阶段策略

**为什么分两阶段**：一次性全改 11 个步骤容易遗漏；但如果只改骨架不改 workflow，会出现新旧混合的混乱。折中方案：先做骨架 + 一个试点步骤验证可行性，确认后再系统转换。

### Phase 1: 骨架 + 试点

| 步骤 | 内容 | 产出 |
| --- | --- | --- |
| 1.1 | 创建 shared/ 目录和三个文件 | conventions.md, dev-guide.md, gate-template.md |
| 1.2 | 瘦身 SKILL.md | ~120 行的路由器版本 |
| 1.3 | 试点改造 Step 1 YAML | 新结构的 workflow + 独立 checklist |
| 1.4 | 验证试点 | 确认单步可独立、Gate 可操作、旧步骤不受影响 |

**试点选择 Step 1 YAML 的理由**：
- 在流程最前面，依赖最少
- 改造后效果最直观（YAML 定义是所有后续步骤的基础）
- 当前的 checklists.md §1 内容明确，容易按 Gate 1/2/3 重构

### Phase 2: 全量转换

| 步骤 | 内容 |
| --- | --- |
| 2.1 | 按试点模板转换剩余 10 个步骤的 workflow + checklist |
| 2.2 | 为 reference.md 添加反向索引 |
| 2.3 | 删除旧 checklists.md，更新 README.md |
| 2.4 | 全量回归验证（用已完成的算子案例走一遍新流程） |

---

## 十、预期效果

### 解决的问题

| 痛点 | 当前 | 改造后 |
| --- | --- | --- |
| 耦合度深 | 每步隐式依赖 SKILL.md 全局上下文 | 每步通过接口契约解耦，只依赖上一步的具体产出 |
| 难量化评估 | 验证靠主观叙述 | 三层验收门：自动/半自动/人工 Yes-No |
| 步骤无法独立 | 必须从头读 SKILL.md | 单步入口：只读 shared/ + 对应 workflow |
| Checklist 维护难 | 400 行大文件，与 workflow 手工映射 | 按步骤 1:1 拆分，与 workflow 同步维护 |
| 人/AI 需求不同 | 同一份文件服务两种读者 | AI 完整版 + 人类精简版 |

### 不变的部分

- reference.md 保留（加反向索引）
- templates/ 目录保留
- scripts/ 目录保留
- 端到端流程仍可用（SKILL.md 流程总览图保留）
- 行为准则 D1-D4 内容不变（只是从 SKILL.md 迁移到 shared/）

---

## 附录：待讨论的开放问题

1. **试点步骤选择**：建议 Step 1 YAML，是否有更好的选择？
2. **Gate 1 自动验证脚本**：Phase 1 是否需要同步产出，还是 Phase 2 再做？
3. **旧 checklists.md 的过渡期**：Phase 1 试点期间，旧文件保留还是标注"部分已迁移"？
4. **dev-guide.md 的内容范围**：人类精简版需要包含哪些内容？是否需要同事参与评审？
