# Workflow 4: PyBoost（Pynative, C++）

## 目标

实现 Pynative 路径下的 ACLNN 算子调用。
**根据接入路径不同，本步骤工作量差异很大：**
- **路径 1（自动生成）**：gen_ops.py 已生成完整调用代码，此步只需**验证**
- **路径 2（Customize）**：需要在 customize 目录手写实现文件

## 输入

- **接入路径**：Pre-B 确定的路径 1 或路径 2
- **YAML 定义**：参数列表、dispatch 配置
- **PTA 源码分析**：ACLNN 调用细节、参数预处理逻辑
- **（组合场景）ACLNN 调用链**：子算子列表与依赖关系

## 输出

- **路径 1**：验证自动生成的 PyBoost 调用代码正确
- **路径 2**：手写 PyBoost 实现文件 `op_name_ascend_customize.cc/.h`
- **（反向）PyBoost Grad 实现文件（路径 2）**

---

## 执行步骤

### 路径 1 分支：验证自动生成产物

> 如果 Pre-B 确定为路径 1（参数直通），**不需要手写 PyBoost 文件**。
> 只需验证 gen_ops.py 自动生成的调用代码正确：

1. **确认生成代码存在**：检查自动生成目录下有对应的 PyBoost 调用模板产物
2. **确认 ACLNN 调用参数正确**：自动生成的 `LAUNCH_ACLNN(aclnnXxx, ...)` 参数顺序/类型无误
3. **编译验证**：确保自动生成代码编译通过
4. 验证通过后，直接进入 [Workflow 5: KBK](./05-kbk.md)

> **如果验证发现自动生成代码有问题**（参数不匹配等），
> 说明参数无法直通，需要重新评估→改为路径 2。

### 路径 2 分支：手写 Customize 文件

#### Step 0：ACLNN 接口核对（反幻觉）

> 若 Pre-B/Pre-C 阶段已确认接口信息（用户贴过文档、PTA 源码已分析），直接引用之前的结论即可，跳过本步。

当准备调用一个之前未确认过的 ACLNN 接口时，先核实：
1. **grep 搜索**：`grep -r "aclnnXxx" .` 尝试寻找项目内的已有调用或头文件定义。
2. **确认签名**：
   - **若搜到**：提取参数列表（尤其是 workspace 参数位置）。
   - **若未搜到**（新算子/头文件不在工程内/宏拼接）：**禁止猜测**，请用户提供 ACLNN 接口文档或头文件定义片段。
3. **记录证据**：向用户展示查到的代码引用或用户提供的文档内容。

#### Step 1：单算子直连模式

标准三段式（`reference.md` §5）：
1. 输出 tensor 分配
2. 参数转换（tuple→vector / None 处理等）
3. ACLNN 两段式调用（`LAUNCH_ACLNN` 或项目等价宏）

### Step 2：组合算子模式（C++ 小算子 API 拼接）

当目标算子由多个小算子拼接组合实现时（`reference.md` §23.1）：
1. 引入头文件 `#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"`
2. 直接调用 C++ 小算子 API（如 `add()`/`mul()`/`sum_ext()` 等）拼接计算逻辑，**无需手动 `LAUNCH_ACLNN`**
3. YAML 设置 `bprop_expander: False`，由小算子各自负责自动微分
4. 若大算子已有独立 bprop，需用 `RequireGradGuard(false)` 禁止小算子重复做自动微分

### Step 3：View 算子模式（零拷贝，`reference.md` §26）

当算子为纯 shape/strides 变换（如 transpose、reshape、expand_dims、slice 等）时：

1. **不需要** `LAUNCH_ACLNN` / PyBoost customize（框架自动处理 View 路径）
2. **需要实现**：strides 计算函数（`{OpName}ViewBasicTypeCalc`）+ 注册
3. **文件位置**：`ops/view/{op_name}_view_strides_calc.cc` + 头文件 `ops/include/view/{op_name}_view_strides_calc.h`
4. **YAML 配置**：原始算子 YAML 加 `view: True`
5. strides 计算逻辑参考 PyTorch `aten/src/ATen/native/TensorShape.cpp` 中对应算子

> View 专用 YAML（`{op_name}_view_op.yaml`）的 strides calc 通常直接委托给原始算子的 strides 计算函数。

### Step 4：输入参数转换（`reference.md` §5.1）

- tuple/list → `std::vector<int64_t>`
- 可选输入 None → 定义 None 语义，PyBoost/Infer/KBK 同步处理
- 标量参数 → 按项目封装提取

### Step 5：对照相似算子（以仓库现状为准）

**必须**参考同目录下相似算子的现有代码文件，确保宏/工具函数用法一致。
> ⚠️ 宏名、头文件、工具函数可能随版本变化。不要照搬 reference.md 中的示例，
> 以 `customize/` 目录下最新的已有算子代码为准。

代码骨架见 `reference.md` §18.3（单算子）/ §23.1（C++ API 拼接）/ §26.3（View strides calc），但**以仓库实际代码为最终参考**。

---

## 成功标准

**路径 1**：
- [ ] 确认自动生成的 PyBoost 调用代码存在且参数正确
- [ ] 编译通过

**路径 2**：
- [ ] PyBoost 前向 Customize 实现完成，编译通过
- [ ] PyBoost 反向 Customize 实现完成（如需要），编译通过
- [ ] 参数转换正确（tuple/None/标量）
- [ ] 组合场景：中间 tensor 分配正确，调用顺序与 PTA 一致
- [ ] 风格与同目录已有实现一致

**View 算子**：
- [ ] strides 计算函数实现正确（shape/strides/offset）
- [ ] 原始算子 YAML 已加 `view: True`

---

## 下一步

PyBoost 完成后，进入 **[Workflow 5: KBK](./05-kbk.md)**
