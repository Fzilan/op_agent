# Workflow 3: GeneralInfer（C++ 推导）

## 目标

实现算子的形状/类型推导（C++），处理动态 shape/rank 回退。
可选：实现 InferValue 常量折叠。

## 输入

- **YAML 定义**：参数列表、输出结构
- **PTA 源码分析**：输出 shape 推导逻辑

## 输出

- **Infer 实现文件**：`op_name_general_infer.cc`（或项目对应路径）
- **（可选）InferValue 实现**

---

## 执行步骤

### Step 1：实现 InferShape

职责边界（`reference.md` §4.1）：
- **只做推导**，不做运行时合法性校验（交给 ACLNN/运行时）
- 报错使用框架异常宏，包含：参数名、期望、实际

### Step 2：处理动态 shape/rank

三种动态类型及策略（`reference.md` §27）：

| 类型 | Infer 策略 |
| --- | --- |
| InputDynamic | 输出对应维度设为 `kShapeDimAny` |
| Input Value Depend | `GetShapeValue()` 取值；unknown 时回退 |
| Compute Depend | 分配最大可能 size + 运行后 SyncOutputShape |

快速回退策略（`reference.md` §4.2）：
- 动态 rank → 返回 `kShapeRankAny`
- 关键参数 unknown → 对应维度回退 `kShapeDimAny`
- 参数都已知 → 返回精确 shape

### Step 3：实现 InferType

通常输出 dtype 与输入一致或按算子语义确定。

### Step 4：常用 API（`reference.md` §4.3）

以项目已有实现为准：
- `GetScalarValueWithCheck<T>()`
- `GetArrayValue<T>()` + `HasUnknownValue()`
- `IsNone()`

### Step 5（可选）：InferValue 常量折叠

当算子输入编译期全部已知时（`reference.md` §26）：
- C++ 实现（优先）或 Python 回调
- 验证：全常量输入 UT + IR 图中确认 ValueNode

代码骨架见 `reference.md` §24.2。

---

## 成功标准

- [ ] InferShape 实现完成，覆盖精确推导和动态回退
- [ ] InferType 实现完成
- [ ] 编译通过，无链接错误
- [ ] C++ UT 可构造 unknown/None 输入并验证推导结果
- [ ] （可选）InferValue 实现并验证

---

## 下一步

GeneralInfer 完成后，进入 **[Workflow 4: PyBoost](./04-pyboost.md)**
