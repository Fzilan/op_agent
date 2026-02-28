# Workflow 5: KBK（Graph, C++）

## 目标

实现 Graph 路径下的 ACLNN kernel。
**根据接入路径不同，本步骤工作量差异很大：**
- **路径 1（自动生成）**：gen_ops.py 已通过 `MS_ACLNN_COMMON_KERNEL_FACTORY_REG` 自动注册，此步只需**验证**
- **路径 2（Customize）**：需手写 kernel 文件（GetWorkSpaceInfo + Launch + 注册）

## 输入

- **接入路径**：Pre-B 确定的路径 1 或路径 2
- **YAML 定义**：参数列表
- **PyBoost 实现**：可参考参数处理逻辑（路径 2）
- **（组合场景）ACLNN 调用链**

## 输出

- **路径 1**：验证自动注册代码正确
- **路径 2**：手写 KBK kernel 文件 + 注册
  - `op_name_aclnn_kernel.cc/.h`（前向）
  - `op_name_grad_aclnn_kernel.cc/.h`（反向）
  - `MS_ACLNN_KERNEL_FACTORY_REG` 注册

---

## 执行步骤

### 路径 1 分支：验证自动注册

> 如果 Pre-B 确定为路径 1（参数直通），**不需要手写 KBK kernel 文件**。
> gen_ops.py 已生成 `aclnn_kernel_register_auto.cc` 中的注册代码。

1. **确认注册存在**：在自动生成的注册文件中搜索算子名，确认 `MS_ACLNN_COMMON_KERNEL_FACTORY_REG` 已注册
2. **编译验证**：确保 Graph 模式下能正确调度到 ACLNN kernel
3. 验证通过后，直接进入 [Workflow 6: BPROP](./06-bprop.md)

> **如果验证发现自动注册有问题**，需要重新评估接入路径。

### 路径 2 分支：手写 Kernel 文件

#### Step 0：ACLNN 接口核对（反幻觉）

与 [04-pyboost.md Step 0](./04-pyboost.md#step-0aclnn-接口核对反幻觉) 相同。若 Step 4（PyBoost）已完成核对，直接引用结论即可。

#### Step 1：标准结构（`reference.md` §6）

- `GetWorkSpaceInfo()`：取参 + `GetWorkspaceForResize`
- `Launch()`：调用 `RunOp` 或等价执行路径
- 注册：`MS_ACLNN_KERNEL_FACTORY_REG`

### Step 2：强约束

- 前向/反向**分文件、分注册**
- 头/实现命名空间保持一致
- 不要在 InferShape 中修改属性（`reference.md` §13.1）

### Step 3：Resize/Launch 优化（`reference.md` §13）

- 能在 Init 确定的放 Init
- 与 shape 强相关的放 Resize
- Launch 只做发射/调用
- 无意义输出：覆写 `GetUseLessOutputIdx()`
- 计算依赖输出：分配最大可能 + SyncOutputShape

### Step 4：组合算子模式（Meta DSL 编程范式，`reference.md` §23.2）

Meta DSL 通过 C++ 构图替代手动 `GetWorkSpaceInfo/Launch/RunOp`，框架自动处理类型推导和自动微分：
1. 在 `mindspore/ccsrc/frontend/operator/meta_dsl/func_op/` 下新建 `.cc` 文件
2. 使用 `REGISTER_FUNCTION_OP(OpName)` 注册算子（可选传入校验函数）
3. 在 `BeginFunction(OpName, args...) { ... } EndFunction(OpName)` 中用 `Call(Prim(SubOp), ...)` 拼接小算子
4. 框架自动处理多平台适配，**无需手写 KBK kernel 文件**

代码骨架见 `reference.md` §18.4（单算子）/ §23.2（Meta DSL），但**以仓库实际代码为最终参考**。

### Step 5：View Host Kernel（当 YAML 标记 `graph_view: True` 时，`reference.md` §26.4）

当算子为 View 算子且需要支持 KBK 图模式 View 路径时：

1. 在 YAML 中配置 `graph_view: True`
2. 在 `ops/kernel/host/view/kernel_mod_impl/` 下新建 `{op_name}_view.cc/.h`
3. 继承 `HostKernelMod`，实现 `GetWorkSpaceInfo` → 调用 strides 计算更新输出 `tensor_storage_info`
4. 使用 `MS_HOST_REG_KERNEL({OpName}View, {OpName}View)` 注册
5. **不走 ACLNN**，host kernel 直接操作 strides

> ⚠️ 注册宏名（如 `MS_ACLNN_KERNEL_FACTORY_REG`）、基类、workspace 接口等
> 可能随版本变化。务必先看 `kernel_mod_impl/` 下最新已有算子的写法。

---

## 成功标准

**路径 1**：
- [ ] 确认自动注册代码存在且算子名正确
- [ ] 编译通过，Graph 模式可调度

**路径 2**：
- [ ] KBK 前向 kernel 实现完成，编译通过
- [ ] KBK 反向 kernel 实现完成（如需要），编译通过
- [ ] 注册宏正确，可在 Graph 模式下调度到
- [ ] 组合场景：Meta DSL 拼接逻辑正确，或 workspace 管理正确（旧模式）
- [ ] 前后向分文件

**View 算子**：
- [ ] `graph_view: True` 已配置在 View 专用 YAML 中
- [ ] Host kernel 实现正确（strides 更新 + `MS_HOST_REG_KERNEL` 注册）

---

## 下一步

KBK 完成后，进入 **[Workflow 6: BPROP 注册](./06-bprop.md)**
