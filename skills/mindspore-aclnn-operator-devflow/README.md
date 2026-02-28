# MindSpore ACLNN Operator Devflow (Agent Skill)

**MindSpore Ascend ACLNN 算子适配开发流程**（以 Agent Skill 形式组织，兼容 Cursor 与 Claude Code），覆盖前向/反向、
PyBoost/KBK、动态 shape、测试与文档，帮助开发者更稳定地完成对接与验证闭环。
支持**两条接入路径**：参数直通的自动生成路径（路径 1）和参数需预处理的 Customize 路径（路径 2）。

## 结构一句话

**SKILL** 看流程 → **workflows/** 按 00～10 做 → 某步细节查 **reference.md §X** → 自检用 **checklists.md**，交活用 **templates/**，溯源看 **traceability.md**。

## 文件结构

| 文件/目录 | 用途 | 说明 |
| --- | --- | --- |
| `SKILL.md` | **主入口** | 流程图、TODOLIST 执行清单、条件跳步、关键约束、排障路径 |
| `workflows/` | **步骤详解** | 每步独立 md 文件，含 Goal/Input/Output/Steps/成功标准/下一步；步骤内会引用 `reference.md` §X |
| `templates/` | **中间产物模板** | Feature 文档模板、PTA 源码审查报告、ACLNN 调用链盘点表、AI 操作记录模板 |
| `reference.md` | 知识参考 | 技术细节、代码骨架模板、接口重载与 View 等专题 |
| `checklists.md` | 可复制清单 | 带优先级标记的 12 大类检查项 + 最终文件清单 + 提交前 Top-25 |
| `examples.md` | 触发样例 | "用户说 X → agent 做 Y" 示例 |
| `traceability.md` | 溯源映射 | 原始来源文档 → skill 落点对应表 |
| `workflows/reconciliation-report.md` | 核对报告（维护用） | workflow 与源文档逐节核对结果；维护时用 |
| `scripts/` | 工具脚本 | PTA 支持范围探测脚本模板 |

### workflows/ 目录详情

| 文件 | 对应步骤 |
| --- | --- |
| `00-pre-checks.md` | Pre-A 存量检查 + Pre-B 方案设计 + Pre-C 调用链盘点 |
| `01-yaml-definition.md` | Step 1: YAML 定义（op_def/api_def/function_doc） |
| `02-code-generation.md` | Step 2: 代码生成（gen_ops.py） |
| `03-general-infer.md` | Step 3: GeneralInfer + InferValue |
| `04-pyboost.md` | Step 4: PyBoost（Pynative） |
| `05-kbk.md` | Step 5: KBK（Graph） |
| `06-bprop.md` | Step 6: BPROP 注册 |
| `07-export.md` | Step 7: 导出与占位 |
| `08-testing.md` | Step 8: 测试（UT + ST） |
| `09-docs.md` | Step 9: 文档（EN + CN） |
| `10-delivery.md` | Step 10: 转测交付 |

## 如何使用

本 skill 同时兼容 **Cursor** 和 **Claude Code**（SKILL.md 格式通用）。

### Cursor 用户
把 `mindspore-aclnn-operator-devflow/` 放到项目的 `.cursor/skills/` 下，Cursor 启动后自动发现。

### Claude Code 用户
把同一目录放到项目的 `.claude/skills/` 下，Claude Code 自动发现；
也可用 `/mindspore-aclnn-operator-devflow` 斜杠命令手动调用。

### 团队同时使用两个工具
只需维护一份文件，用符号链接让两边都能读到：

```bash
# Linux / Mac
mkdir -p .claude/skills
ln -s ../../.cursor/skills/mindspore-aclnn-operator-devflow .claude/skills/

# Windows（管理员 cmd）
mkdir .claude\skills
mklink /D .claude\skills\mindspore-aclnn-operator-devflow .cursor\skills\mindspore-aclnn-operator-devflow
```

如果 Windows 下不方便建符号链接，也可以用脚本同步：

```powershell
# sync-skills.ps1
Copy-Item -Recurse -Force .cursor\skills\* .claude\skills\
```

建议在 `.gitignore` 中加入 `.claude/skills/`，只把 `.cursor/skills/` 作为 git 中的唯一来源。

## 快速上手

1. 阅读 `SKILL.md` 了解整体流程图与 TODOLIST 执行清单。
2. 按 `workflows/` 中的步骤文件逐步执行（每步有明确的输入/输出/成功标准）。
3. 使用 `templates/` 中的模板产出中间产物（**Feature 文档**、PTA 分析报告、调用链盘点表）。
4. 开发过程中，按 `checklists.md` 逐项自检。
5. 遇到具体实现问题，查阅 `reference.md` 对应章节（有目录索引）。
6. 提交前，过一遍 `checklists.md` 末尾的"提交前必检 Top-25"。

## 已知局限与优化建议

本 skill 基于有限的算子案例和文档构建，使用前建议了解以下局限并按需补强：

| 局限 | 建议的补强方式 |
| --- | --- |
| **算子类型覆盖有限**：主要基于路径 2（Customize）复杂算子经验，路径 1、符号重载、纯 Python 组合等场景覆盖较浅 | 拿不同类型的算子实际跑一遍 workflow，记录不适用的步骤并调整 |
| **框架实现细节可能滞后**：`gen_ops.py`、注册宏、GeneralInfer API 等已基于源码写入 `reference.md`（§3-§7），但框架持续演进 | 实现时对照同目录已有算子的最新代码；skill 已内置"仓库现状 > 文档描述"原则 |
| **AI 幻觉风险**：可能编造不存在的接口或误判参数属性，尤其在缺乏上下文时 | skill 已内置核心行为准则（三位一体校验、无证不推、疑点上报）；见 SKILL.md 核心行为准则 |
| **隐性知识依赖人工补充**：组内口头惯例、评审偏好、已废弃流程无法从文档或代码中获取 | 让有经验的开发者审阅 `checklists.md`，标注过时/缺失的条目 |
| **排障案例不够丰富**：常见编译报错、ST 失败的典型原因覆盖不全 | 收集组内 Top 10 常见问题，补充到 `reference.md` 对应章节 |
| **PTA/CANN 版本差异**：不同版本间接口有差异，skill 未针对特定版本适配 | 使用前确认版本矩阵，遇到差异以实际头文件/源码为准 |
| **不同 AI 模型效果有差异**：指令跟随精度、验证自觉性因模型而异 | 首次用新模型时先拿简单算子跑完整 workflow 验证；skill 的验证闭环和 checklist 机制用于兜底模型差异 |

> **核心原则**：本 skill 提供的是经过验证的流程框架，但不能替代对目标仓库实际代码的阅读。
> 每一步的具体写法以仓库里同类算子的最新实现为准。

## 版本说明

MindSpore/ACLNN 相关目录结构、注册宏、生成脚本在不同版本可能变动；遇到差异以目标仓库现状为准，
建议用 tag 记录已验证版本。

## License

建议使用 Apache-2.0（或按团队规范配置）。
