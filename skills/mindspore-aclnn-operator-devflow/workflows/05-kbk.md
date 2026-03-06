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
- 不要在 InferShape 中修改属性（`reference.md` §16.1）

### Step 3：Resize/Launch 优化（`reference.md` §16）

- 能在 Init 确定的放 Init
- 与 shape 强相关的放 Resize
- Launch 只做发射/调用
- 无意义输出：覆写 `GetUseLessOutputIdx()`
- 计算依赖输出：分配最大可能 + SyncOutputShape

### Step 4：组合算子模式（`reference.md` §29.2）

- workspace：多个子算子分别计算并累加
- Launch 中按顺序调用多个 `RunOp`
- 任一子算子失败立即返回 false
- 中间 tensor 可能需要通过 workspace 分配

代码骨架见 `reference.md` §24.4（单算子）/ §29.2（组合），但**以仓库实际代码为最终参考**。

> ⚠️ 注册宏名（如 `MS_ACLNN_KERNEL_FACTORY_REG`）、基类、workspace 接口等
> 可能随版本变化。务必先看 `kernel_mod_impl/customize/` 下最新已有算子的写法。

---

## 成功标准

**路径 1**：
- [ ] 确认自动注册代码存在且算子名正确
- [ ] 编译通过，Graph 模式可调度

**路径 2**：
- [ ] KBK 前向 kernel 实现完成，编译通过
- [ ] KBK 反向 kernel 实现完成（如需要），编译通过
- [ ] 注册宏正确，可在 Graph 模式下调度到
- [ ] 组合场景：workspace 管理正确，多 RunOp 顺序正确
- [ ] 前后向分文件

---

## 下一步

KBK 完成后，进入 **[Workflow 6: BPROP 注册](./06-bprop.md)**
