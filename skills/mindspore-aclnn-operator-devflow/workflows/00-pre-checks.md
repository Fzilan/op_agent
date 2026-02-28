# Workflow 0: 前置检查（Pre-A / Pre-B / Pre-C）

## 目标

在动手写代码前，完成存量检查、方案设计与对标分析、组合场景调用链盘点。

## 输入

- **算子名称**：用户提供的算子名（API 名、Primitive 名、ACLNN 名）
- **PTA 对标接口**：`torch_npu.npu_xxx` 或 `torch.xxx`

## 输出

- **存量检查结果**：该算子在 MS 仓库中已有 / 缺失的部分
- **方案设计文档**：接口类型、对接分类、影响面评估, 以md文件形式输出
- **ACLNN 调用链盘点表**（组合场景）：子算子覆盖状态与实施计划

## 约束

- **本地代码优先**：PTA（op-plugin）、PyTorch、MindSpore 的源码查阅**必须在本地工作区目录中搜索**，禁止通过网络搜索或 WebFetch 获取这些仓库的代码内容。只有在本地确实不存在对应仓库、且用户明确同意时，才可使用网络作为兜底手段。

---

## Pre-A：存量检查

当用户让你"新增 / 适配某算子"时，**先搜索确认**该算子在仓库中是否已存在。

### 执行步骤

1. **搜索 YAML**：在 `mindspore/ops/op_def/yaml/` 中搜索算子名
2. **搜索 Infer**：在 `ops_func_impl` / `ops/infer` 目录搜索对应 FuncImpl
3. **搜索 PyBoost**：搜索 `LAUNCH_ACLNN` + 算子名
4. **搜索 KBK**：搜索 `MS_ACLNN_KERNEL_FACTORY_REG` + 算子名
5. **搜索 BPROP**：搜索 `REG_BPROP_BUILDER("OpName")`
6. **搜索测试**：在 `tests/` 下搜索算子名
7. **搜索文档**：在 `docs/api/` 下搜索算子名

### 输出模板

```text
算子存量检查：{OpName}

| 组件 | 状态 | 文件路径 | 备注 |
| ---- | ---- | -------- | ---- |
| YAML (op_def) | ✅/❌ | ... | |
| YAML (api_def) | ✅/❌ | ... | |
| GeneralInfer | ✅/❌ | ... | |
| PyBoost | ✅/❌ | ... | |
| KBK kernel | ✅/❌ | ... | |
| BPROP | ✅/❌ | ... | |
| 测试 (UT) | ✅/❌ | ... | |
| 测试 (ST) | ✅/❌ | ... | |
| 文档 (EN) | ✅/❌ | ... | |
| 文档 (CN) | ✅/❌ | ... | |

结论：{全新开发 / 需补齐xxx部分}
```

如果已存在，**不要直接改代码**，先问用户是否要继续做"完善"。

---

## Pre-B：方案设计与对标分析

对齐 MS/PTA/CANN 的接口差异，决定原语/接口接入策略，**确定接入路径（路径 1 自动 / 路径 2 Customize）**，并初始化 Feature 文档。

### 执行步骤

1. **PTA 源码审查（必做）**：审查 op-plugin 三类关键文件（详见 `reference.md` §19）
   - `op_plugin_functions.yaml`：函数签名、参数类型/默认值
   - `derivatives.yaml`：反向注册、可微输入
   - `XxxKernelNpuOpApi.cpp`：实际 ACLNN 调用、参数预处理
   - 注意 PTA 是否有**同名接口重载**（同函数名、不同参数签名）
2. **接口分析五要素（必做）**（`reference.md` §15.4.1）：
   - 功能 / 参数定义 / 数据类型是否一致
   - **是否要新增原语**；**是新增接口还是复用原有接口**
3. **确定 YAML 策略**（`reference.md` §15.4.2）：
   - 已有 YAML + 复用原有原语 → 加 `dispatch` 字段
   - 已有 YAML + 新增原语 → 新建 YAML 加 `_ext` 后缀
   - 没有 YAML → 新建
   - 若不兼容且不能改存量接口 → `ops.extend`（`reference.md` §15.4.3）
   - 若需修改已有原语参数签名 → 参考 MS 仓库相似算子处理方式，具体分析兼容性（`reference.md` §15.4.4）
4. **确定接入路径（核心决策）**（`reference.md` §2.3）：
   - 分析 MindSpore API 参数能否**原样透传**给 ACLNN 接口
   - **路径 1（自动生成）**：参数直通 → YAML 不写 `Ascend` 字段 → Step 4/5 跳过手写
   - **路径 2（Customize）**：参数需预处理 → YAML 写 `Ascend: XxxAscend` → Step 4/5 必须手写
   - 常见需预处理的情况：tuple→vector、Optional None 处理、str→enum、标量提取、参数重排、输出手动分配
   - **此决策直接决定后续整个开发工作量，必须在 Pre-B 阶段明确**
5. **评估影响面（含 CPU/GPU 不回退）**：
   - 若**复用存量原语或适配存量算子 API**，必须确保原有 CPU/GPU 流程不回退、功能不退化
   - 若**全新算子（新原语 + 新接口）**，不需要新增 CPU/GPU 支持，仅做 Ascend
   - 是否影响 GE/Lite 现有流程；有影响需 Pass/Expander 消除
6. **版本矩阵记录**：torch / torch_npu / CANN 版本
7. **产出 PTA 差异记录**（使用 `templates/pta-analysis-report.md` 模板）

---

## 🔒 Feature 文档初始化（Pre-B 完成后必须执行，不可跳过）

> **这是评审和转测交付的必须产物。** 无论什么场景（前向/反向、单算子/组合、内部/公开），
> 都必须生成 Feature 文档。如果跳过此步，后续将无法通过评审。

### 执行步骤

1. 从 `templates/feature-document.md` 复制一份，命名为 `{算子名}_Feature.md`
2. 基于 Pre-B 的分析结果，填写以下章节：
   - §1（背景描述）
   - §2（标杆与接口）
   - §3（任务清单——标准 13 大类表格，初始化每项状态）
   - §4（功能与接口说明——接口签名、参数说明）
   - §6（约束与类型——设备、dtype、shape 约束）
   - §8（与 PTA 的差异与对齐——初始化版）
3. **后续每完成一个 Workflow Step，必须回填 Feature 文档对应章节**：
   - Step 1 → §5（YAML）
   - Step 3 → §9（动态 Shape）、§10（异常）
   - Step 4/5 → §7（执行模式）
   - Step 6 → §11（反向）
   - Step 8 → §12（测试方案）
   - 代码完成后 → §13（代码改动）、§14（验收报告）

### 检查点

> **⛔ HARD GATE：在进入 Step 1 之前，以下两项必须完成并交付给用户：**
> 1. ✅ PTA 源码审查报告（Pre-B 产出，使用 `templates/pta-analysis-report.md` 模板）
> 2. ✅ Feature 文档初始化版（§1-§4、§6、§8 已填写）
>
> **缺少任何一项则停下，禁止继续。不可默默跳过。**

> **⚠️ "交付给用户"的含义：生成实际的 .md 文件到工作区，并告知用户文件路径。**
>
> - PTA 源码审查报告：按 `templates/pta-analysis-report.md` 模板填充后，用 Write 工具
>   **生成文件**（如 `{op_name}_pta_analysis.md`），在消息中告知用户文件路径。
> - Feature 文档：按 `templates/feature-document.md` 模板填写 §1-§4、§6、§8 后，
>   **生成文件**（如 `{op_name}_feature.md`），在消息中告知用户文件路径。
> - 后续每个 Step 完成时，用 StrReplace 工具**回填 Feature 文件的对应章节**，
>   在执行报告中告知用户"已更新 Feature 文件 §X"。
>
> agent 常见错误：做了分析但只在脑中/消息中记录，没有生成文件。
> 文件不存在 = 文档未产出。

---

## Pre-C：ACLNN 调用链分析与子算子盘点（组合场景必做）

> 仅当 PTA C++ 实现中使用**多个 ACLNN 小算子串联**时执行。
> 单个 `aclnnXxx` 直连时跳过此步。

### 执行步骤

1. **提取 ACLNN 调用链**：从 PTA C++ 代码中提取前向+反向的全部
   `EXEC_NPU_CMD` / `aclnnXxx` 调用（详见 `reference.md` §22.2）
2. **盘点 MS 覆盖情况**：逐个搜索确认子算子是否已接入（`reference.md` §22.3）
3. **产出覆盖盘点表**（使用 `templates/aclnn-callchain-inventory.md` 模板）
4. **规划实施顺序**：叶子算子先、组合算子后；按拓扑序（`reference.md` §22.5）

---

## 需要用户提供/确认的信息

> 以下信息 agent 无法自行获取，必须主动向用户索取。**缺少任何一项时暂停并明确告知用户。**

| 信息 | 何时需要 | 如何向用户表述 |
| --- | --- | --- |
| 算子名称（API 名 / Primitive 名 / ACLNN 名） | Pre-A 开始前 | "请确认要适配的算子名称" |
| ACLNN 文档/头文件 | Pre-B 步骤 1 | "请提供 aclnnXxx 的文档或头文件路径" |
| PTA 源码路径（若不在工作区） | Pre-B 步骤 1 | "我需要 op-plugin 源码路径来审查 PTA 实现" |
| 版本矩阵（torch/torch_npu/CANN/芯片） | Pre-B 步骤 6 | "请提供当前使用的 torch/torch_npu/CANN 版本和芯片型号" |
| 代码与文档不一致时的确认结论 | Pre-B 发现差异时 | "PTA 代码与文档在 XX 处不一致，请找接口人确认以哪边为准" |
| 是否需要反向 / 性能目标 / 精度要求 | Pre-B 方案设计 | "请确认：是否需要反向？性能目标？是否要求精度 0 偏差？" |
| 运行环境（是否有 Ascend 设备） | Pre-B 方案设计 | "后续 ST 测试需要 Ascend 设备，请确认是否可用" |

---

## 成功标准

- [ ] Pre-A：已完成存量检查，明确告知用户当前状态
- [ ] Pre-B：已审查 PTA 三类关键文件，完成接口分析五要素，确定 YAML 策略
- [ ] Pre-B：**已明确接入路径**（路径 1 自动生成 / 路径 2 Customize），并记录决策依据
- [ ] Pre-B：若涉及新增/修改接口或原语，已确认评审要求和跨后端影响
- [ ] Pre-B：已记录版本矩阵
- [ ] Pre-B：如发现代码与文档不一致，已整理差异清单交给用户确认
- [ ] **🔒 Feature 文档初始化版已生成**（§1-§4、§6、§8 已填写）——此项不可跳过
- [ ] Pre-C（组合场景）：已提取完整 ACLNN 调用链
- [ ] Pre-C（组合场景）：已产出覆盖盘点表
- [ ] Pre-C（组合场景）：已规划实施顺序

---

## 下一步

前置检查完成后，进入 **[Workflow 1: YAML 定义](./01-yaml-definition.md)**
