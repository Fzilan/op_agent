# Workflow 2: 代码生成

## 目标

运行 `gen_ops.py`，基于 YAML 生成算子代码。**gen_ops.py 在两条路径下作用不同：**
- **路径 1（自动生成）**：生成完整的 PyBoost/KBK 调用代码 + 注册代码 + Python 接口包装
- **路径 2（Customize）**：生成包装代码（调用你手写的 Customize 类）+ Python 接口包装

## 输入

- **YAML 文件**：Workflow 1 产出的 op_def / api_def / function_doc
- **接入路径**：Pre-B 确定的路径 1 或路径 2

## 输出

- **gen_ops.py 运行成功**（无报错）
- **自动生成的文件**（以下文件由 gen_ops.py 产出，通常在 `.gitignore` 中，不入库）：
  - `functional_overload.py`：Python 函数式 API 入口
  - `aclnn_kernel_register_auto.cc`：ACLNN kernel 自动注册（路径 1 的 KBK）
  - PyBoost 模板产物（路径 1：完整调用代码；路径 2：调用 Customize 的包装代码）
  - 其他 `auto_generate/` 下的文件

> **重要**：这些自动生成文件虽然不入库（在 `.gitignore` 中），但**运行时必需**。
> 每次修改 YAML 后都必须重新运行 gen_ops.py 更新它们。

---

## 执行步骤

### Step 1：运行 gen_ops.py

```bash
python mindspore/ops/op_def/gen_ops.py
```

### Step 2：确认自动生成产物

运行完成后，**必须确认**以下文件已正确生成：

| 文件 | 路径 1 | 路径 2 | 说明 |
| --- | --- | --- | --- |
| `functional_overload.py` | 生成 | 生成 | Python API 入口，两条路径都需要 |
| PyBoost 调用代码 | **完整生成** | 生成包装 | 路径 1 直接调用 ACLNN；路径 2 调用 Customize 类 |
| KBK 自动注册 | **完整生成** | 不生成 | 路径 2 需手写 kernel 并手动注册 |

### Step 3：处理常见报错

| 报错类型 | 原因 | 修复方式 |
| --- | --- | --- |
| keys 结构不匹配 | 字段层级有误 | 对照已有算子 YAML（如 add）调整 |
| 缺 `py_method` | api_def 不完整 | 补齐 python 暴露字段 |
| function_doc 缺条目 | 文档节点缺失 | 补齐参数文档，保持一致 |
| 编码问题 | 英文 YAML 混入中文 | 移除中文，中文放 RST |

详见 `reference.md` §3。

### Step 4：路径 2 的"先生成后改造"流程

**仅路径 2 需要**（`reference.md` §2.5）：
1. YAML 中打开 `dispatch.enable: True`
2. 临时注释掉 `dispatch.Ascend` 自定义声明
3. 运行 gen_ops.py 生成可编译骨架（此时按路径 1 生成完整代码）
4. 将生成的 PyBoost/KBK 文件拷贝到 `customize` 目录作为参考起点
5. 按 ACLNN 实际签名调整入参、添加参数预处理逻辑
6. 恢复 YAML 中的 `dispatch.Ascend` 声明
7. 重新运行 gen_ops.py（此时生成的是调用 Customize 类的包装代码）

> **路径 1 不需要此步骤**——gen_ops.py 直接生成最终可用的代码。

---

## 成功标准

- [ ] `gen_ops.py` 运行无报错
- [ ] `functional_overload.py` 中包含新算子的 Python 入口
- [ ] `aclnn_config.yaml` 已添加 `{OpName}: 'aclnn{Op}'` 映射（若算子名与 ACLNN 不一致）
- [ ] 路径 1：确认 PyBoost 调用代码和 KBK 注册已自动生成
- [ ] 路径 2：已完成"先生成后改造"流程，Customize 文件就位

---

## 下一步

代码生成完成后，进入 **[Workflow 3: GeneralInfer](./03-general-infer.md)**
