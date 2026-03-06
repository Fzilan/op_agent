# Workflow 9: 文档

## 目标

完成英文 function_doc（YAML）和**中文 RST 文档**，中英文**内容**严格一致。

> **⚠️ 常见遗漏**：英文 function_doc 通常在 Step 1 的 YAML 里已经写了，
> agent 容易误以为"文档步骤已完成"从而**跳过中文 RST**。
> **英文 doc YAML ≠ 文档步骤完成**——中文 RST 是独立产物，必须单独确认。

**中英文一致的含义**：**内容**一致（参数名、默认值、必选/可选、语义、示例可运行）；**格式**按各自规范，不追求刻板一致——英文需写支持平台与样例，中文不写（生成时从英文提取）；Note/Warning 等按《5. 资料开发指导》中英文格式要求分别写。

**完整规范来源**：本 workflow 为步骤与摘要；**详细要求（开始前事项、六种场景表、mint 特例、常见问题）见 `reference.md` §11.3～§11.6**。  
写作前应阅读参考资料《5. 资料开发指导》中的中英文 API 内容/格式要求链接。

---

## 输入

- **YAML 定义**：function_doc 部分（Step 1 已创建）
- **算子接口实现**：参数、默认值、示例

## 输出（两类文档 + 接口列表，逐项确认）

| 类型 | 文件位置 | 必须程度 | 状态标注 |
| --- | --- | --- | --- |
| **英文 function_doc** | `ops/op_def/yaml/doc/{op}_doc.yaml` | `[MUST]` | ✅已在 Step 1 创建 / 需完善 |
| **中文 RST** | 见 `reference.md` §11.4「按场景落点」表 | `[MUST]` 公开 API 必须 | ✅已写 / ✅已有 / ❌未写 |
| **接口列表** | 见 `reference.md` §11.4「按场景落点」表 | `[MUST]` | 英文列表 + 中文列表均需按字母序添加 |

**开始前事项、六种场景表、mint 特例、常见问题**：见 **`reference.md` §11.3～§11.6**。

---

## 执行步骤

### Step 1：英文 function_doc 完善

确保 Step 1 创建的 YAML function_doc 完整：
- `desc`：算子功能简述（公开 API 建议含**原理、公式、论文出处**或配图，见常见问题-内容）
- `args`：每个参数的描述
- `returns`：返回值描述
- `examples`：可运行的完整示例（含 import）

### Step 2：中文 RST（公开 API 必须）

> ⚠️ **这是最容易被遗漏的步骤。** 先搜索仓库中是否已有对应的中文 RST。

按 `reference.md` §11（§11.4 按场景落点、§11.6 常见问题）与参考资料 5.2/5.3：
- 文件位置：按 **`reference.md` §11.4**「按场景落点」表对应目录（ops/mint/nn/Tensor）
- **先看仓库里同类算子的中文 RST** 确认格式和目录结构
- **文件名、文件内标题、文件内接口定义三者严格一致**（function 仅文件名多 func_ 前缀）；接口名下方 `=` 长度 ≥ 标题名
- 接口列表按**字母序**添加

**如果已有旧版中文 RST**（如 `acos` 已有但 `acos_ext` 没有），需要确认：
- 旧文档是否需要更新指向新接口
- 是否需要为新接口（如 `mint.acos`）新增独立的中文 RST

### Step 3：一致性检查（`reference.md` §11.1）

| 检查项 | 英文 | 中文 |
| --- | --- | --- |
| 参数名 | ✅ 一致 | ✅ 一致 |
| 默认值 | ✅ 一致 | ✅ 一致 |
| 必选/可选 | ✅ 一致 | ✅ 一致 |
| 示例 | ✅ 可运行 | ✅ 可运行 |

### Step 4：落点确认（`reference.md` §11.2）

与「按场景落点」表一致：functional → ops 下 func_*.rst + mindspore.ops.rst；mint → mint 目录 + mindspore.mint.rst；nn → nn 目录 + mindspore.nn.rst；Tensor → Tensor 目录 + mindspore.Tensor.rst；ops Primitive → ops 下无 func_ 的 rst + mindspore.ops.primitive.rst。

---

**常见问题（内容与格式）**：见 **`reference.md` §11.6**（接口描述、实验性接口、反引号/缩进/内部跳转/shape 等）。

---

## 🔒 Step 9 完成前强制检查

```text
文档产出检查清单：

英文 function_doc（YAML）：
  - 文件路径：ops/op_def/yaml/doc/{op}_doc.yaml
  - 状态：✅已在 Step 1 创建且完整 / 需完善（哪些字段缺失：___）

中文 RST：
  - 文件路径：按 reference.md §11.4「按场景落点」表（ops/mint/nn/Tensor/primitive 对应目录）
  - 状态：✅已新建 / ✅已有且覆盖新接口 / ❌未写（原因：___）
  - 若跳过：是否为内部算子（非公开 API）？ 是/否

接口列表（英文 + 中文）：
  - 是否已按字母序添加到对应 mindspore.xxx.rst？ 是/否

中英文一致性：
  - 参数名一致：是/否
  - 默认值一致：是/否
  - 示例一致且可运行：是/否
```

> **公开 API（functional/mint/nn/Tensor）必须有中文 RST。**
> 只有**内部算子**（不在 `__all__` 中导出、不需要公开文档）才允许跳过。
> 跳过时必须明确标注原因，不允许静默跳过。

## 成功标准

- [ ] 英文 function_doc 完整（desc/args/returns/examples）
- [ ] **中文 RST 文件已创建**（公开 API）或明确标注为内部算子可跳过
- [ ] **英文 + 中文接口列表**已按字母序添加到对应 mindspore.xxx.rst
- [ ] 中英文参数名/默认值/示例严格一致
- [ ] 示例可运行，含完整 import
- [ ] 文件名/标题/接口定义三者一致，格式符合 reference.md §11.6

---

## 下一步

文档完成后，进入 **[Workflow 10: 转测交付](./10-delivery.md)**
