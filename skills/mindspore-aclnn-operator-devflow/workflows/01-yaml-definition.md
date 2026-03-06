# Workflow 1: YAML 定义

## 目标

创建算子的 YAML 定义文件（op_def + api_def + function_doc），前向/反向各一份。

## 输入

- **Pre-B 方案设计**：对接类型、参数列表、输入/输出定义
- **PTA 源码审查结果**：参数名、类型、默认值、返回值结构

## 输出

- **YAML 文件**：`mindspore/ops/op_def/yaml/{op_name}_op.yaml`（前向）
- **YAML 文件**：`mindspore/ops/op_def/yaml/{op_name}_grad_op.yaml`（反向，如需要）

---

## 执行步骤

### Step 1：确定 YAML 结构

参照同目录下**同类算子**的 YAML 文件，确认字段层级（按目标算子特征搜索，见 `reference.md` §2.4）。
> ⚠️ YAML 字段结构可能随版本迭代变化，以 `op_def/yaml/` 目录下最新已有算子为准，不要只看文档模板。

核心字段：
- `op_name`：Primitive 名（通常 PascalCase + 可选的 Customize 后缀）
- `args`：每个参数的 `name`/`type`/`default`/`desc`
- `outputs`：每个输出的 `name`/`type`/`desc`
- `dispatch`：路径 1 仅 `enable: True`；路径 2 加 `Ascend: "XxxAscend"`（详见 Step 3）
- `api`：`py_method` / `module` 等 Python 暴露字段
- `function_doc`：英文文档（desc/args/returns/examples）

### Step 2：一致性校验

同一个参数（如 `actual_seq_len`）必须在以下位置一致（`reference.md` §2.1）：
- YAML（op_def + api_def + function_doc）
- GeneralInfer、PyBoost、KBK、文档、测试

### Step 3：dispatch 配置（根据接入路径）

**这是路径决策在 YAML 中的落地点**（`reference.md` §2.3）：

**路径 1（自动生成）**——参数直通，不需要 Customize：
```yaml
dispatch:
  enable: True
  # 不写 Ascend 字段 → 编译自动生成 PyBoost/KBK 代码
```

**路径 2（Customize）**——参数需预处理：
```yaml
dispatch:
  enable: True
  Ascend: OpNameAscend    # 指定 Customize 类名
```

判断依据：
- 参数个数、顺序、类型与 ACLNN 完全一致 → 路径 1
- 需要 tuple→vector / None 处理 / str→enum / 标量提取 / 参数重排 / 手动分配输出 → 路径 2
- 不确定时，先按路径 2 处理（可以后续简化为路径 1，反之不行）

### Step 4：代码骨架参考

YAML 最小模板见 `reference.md` §24.1。

---

## 成功标准

- [ ] YAML 文件已创建（前向；反向如需要）
- [ ] 参数名/类型/默认值与 PTA 源码审查结论一致
- [ ] `dispatch` 配置正确（路径 1 不写 Ascend / 路径 2 写 `Ascend: XxxAscend`）
- [ ] `function_doc` 英文文档已填写
- [ ] 对照相似算子检查字段完整性（无缺 `py_method`、keys 结构正确）

---

## 下一步

YAML 定义完成后，进入 **[Workflow 2: 代码生成](./02-code-generation.md)**
