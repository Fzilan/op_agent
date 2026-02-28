# Workflow 9: 文档

## 目标

完成英文 function_doc（YAML）和**中文 RST 文档**，中英文严格一致。

> **⚠️ 常见遗漏**：英文 function_doc 通常在 Step 1 的 YAML 里已经写了，
> agent 容易误以为"文档步骤已完成"从而**跳过中文 RST**。
> **英文 doc YAML ≠ 文档步骤完成**——中文 RST 是独立产物，必须单独确认。

**中英文一致的含义**：参数名、默认值、必选/可选、语义与示例保持一致；
格式遵循中英文各自规范，不要求字面完全相同。

## 输入

- **YAML 定义**：function_doc 部分（Step 1 已创建）
- **算子接口实现**：参数、默认值、示例

## 输出（两类文档 + 接口列表，逐项确认）

| 类型 | 文件位置 | 必须程度 | 状态标注 |
| --- | --- | --- | --- |
| **英文 function_doc** | `ops/op_def/yaml/doc/{op}_doc.yaml` | `[MUST]` | ✅已在 Step 1 创建 / 需完善 |
| **中文 RST** | `docs/api/api_python/ops/*.rst`（或对应 mint/nn 目录） | `[MUST]` 公开 API 必须 | ✅已写 / ✅已有 / ❌未写 |
| **接口列表** | 对应 `mindspore.xxx.rst` 列表文件 | `[MUST]` | 英文列表 + 中文列表均需按字母序更新 |

---

## 执行步骤

### Step 1：英文 function_doc 完善

确保 Step 1 创建的 YAML function_doc 完整：
- `desc`：算子功能简述（公开 API 建议包含原理、公式、论文出处或必要背景）
- `args`：每个参数的描述
- `returns`：返回值描述
- `examples`：可运行的完整示例（含 import）

### Step 2：中文 RST（公开 API 必须）

> ⚠️ **这是最容易被遗漏的步骤。** 先搜索仓库中是否已有对应的中文 RST。

按 `reference.md` §10 的规范：
- 文件位置：`docs/api/api_python/ops/` 下（或 mint/nn 对应目录）
- **先看仓库里同类算子的中文 RST 文件**确认格式和目录结构
- **文件名、文件内标题、接口定义三者严格一致**（functional 场景通常只有文件名多 `func_` 前缀）
- 接口名下方 `=` 长度不小于标题名长度
- 接口列表按字母序添加

**如果已有旧版中文 RST**（如 `acos` 已有但 `acos_ext` 没有），需要确认：
- 旧文档是否需要更新指向新接口
- 是否需要为新接口（如 `mint.acos`）新增独立的中文 RST

### Step 3：一致性检查（`reference.md` §10.1）

| 检查项 | 英文 | 中文 |
| --- | --- | --- |
| 参数名 | ✅ 一致 | ✅ 一致 |
| 默认值 | ✅ 一致 | ✅ 一致 |
| 必选/可选 | ✅ 一致 | ✅ 一致 |
| 示例 | ✅ 可运行 | ✅ 可运行 |

### Step 4：落点确认（`reference.md` §10.2）

| 接口类型 | 英文位置 | 中文位置 |
| --- | --- | --- |
| functional | 实现 .py | `docs/api/.../ops/func_*.rst` |
| mint | mint 接口实现/列表 | `docs/api/.../mint/*.rst` |
| nn | `nn/*.py` | `docs/api/.../nn/*.rst` |
| Tensor 方法 | `tensor.py` | `docs/api/.../Tensor/` |
| ops Primitive | Primitive 实现/列表 | `docs/api/.../ops/mindspore.ops.*.rst` |

---

## 🔒 Step 9 完成前强制检查

```text
文档产出检查清单：

英文 function_doc（YAML）：
  - 文件路径：ops/op_def/yaml/doc/{op}_doc.yaml
  - 状态：✅已在 Step 1 创建且完整 / 需完善（哪些字段缺失：___）

中文 RST：
  - 文件路径：docs/api/api_python/ops/mindspore.ops.func_{op}.rst（或对应 mint/Tensor 目录）
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
- [ ] 中英文参数名/默认值/示例严格一致
- [ ] 示例可运行，含完整 import
- [ ] 接口列表已按字母序更新
- [ ] 文件名/标题/接口定义三者一致

---

## 下一步

文档完成后，进入 **[Workflow 10: 转测交付](./10-delivery.md)**
