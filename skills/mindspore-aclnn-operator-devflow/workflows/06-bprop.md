# Workflow 6: BPROP 注册

## 目标

在 bprop builder 中把反向图接起来，确保梯度路径正确。

## 输入

- **PTA derivatives.yaml 分析**：哪些输入可微、grad 函数参数传递
- **反向 YAML 定义**：反向算子参数
- **反向 PyBoost/KBK**：已实现的反向 kernel

## 输出

- **BPROP 注册代码**：在合适的 `grad_*ops.cc` 中添加

---

## 执行步骤

### Step 1：基本接线（`reference.md` §7）

- 只为需要梯度的输入构建反向子图
- 非张量/不需要梯度的输入返回零梯度占位
- 使用 `need_compute_grad_out()` 做必要性判断

### Step 2：I/O 个数规则（`reference.md` §7.1）

- 反向输入 = 正向输入个数 + 2（`out` 与 `dout`）
- 反向输出 = 正向输入个数（每个输入一个梯度）
- 多输出正向：`out` 在反向侧通常是 tuple → `TupleGetItem`

### Step 3：进阶注意事项（`reference.md` §12）

| 场景 | 处理方式 |
| --- | --- |
| 不可微分入参 | `ib->OutZeros(x)` |
| 全部不可微分 | `ReturnZeros` |
| 梯度理论为 0 | `ib->ZerosLikeExt()` |
| inplace 反向 | 输入与输出为同一对象时，**只要有一个被用于反向就不能加入 SetUnusedInputs**；反向逻辑需要「更新前的 self」时，注册 **CloneInplaceInput**（见算子反向注意事项 §3） |
| KBK 动态 shape inplace | `ib->Depend(target, inplace_call)` |
| str 类型参数梯度 | 若在 str 位置返回 OutZeros，KBK 反向动态 shape 可能报错，以实际框架行为为准（见算子反向注意事项 §7） |

### Step 4：SetUnusedInputs（`reference.md` §7.2）

反向不依赖某些输入 tensor value 时，标记 unused 以尽早释放内存。

代码骨架见 `reference.md` §18.5。

### Step 5：图模式动态输入处理（`reference.md` §7.3）

> 图模式（KBK）下正向输入的**值或 shape** 在编译态可能未知，bprop builder 中
> 基于正向输入的 ShapeCalc 或控制流分支需要能延迟到运行时执行。
> **不处理此场景会导致图模式下反向编译失败或结果错误。**

必须检查以下场景并采取对应措施：

| 场景 | 检查方法 | 处理方式 |
| --- | --- | --- |
| 标量输入值 unknown | `GetScalarValue<>()->has_value()` | 值已知走编译期分支；unknown 走 `Conditional` 运行时分支 |
| 输入 shape 动态 | `IsDynamicRank()` / `IsDynamicShape()` | shape 依赖的计算用 `DEF_PURE_SHAPE_CALC` + `ib->ShapeCalc` 延迟 |
| 控制流依赖运行时值 | 编译期值可能变化 | 用 `ib->Conditional(cond, true_br, false_br)` 替代 C++ if/else |

> 🚫 **反模式禁令（绝对禁止）**：
>
> 当 `GetScalarValue<>()->has_value()` 返回 false 时，**禁止直接
> `MS_EXCEPTION(ValueError)` 报错退出**。这等于放弃了图模式动态输入的支持。
>
> **错误写法（禁止）**：
> ```cpp
> p = p_node->BuildValue();
> if (!GetScalarValue<float>(p)->has_value()) {
>   MS_EXCEPTION(ValueError) << "p must be constant!";  // ❌ 禁止
> }
> ```
>
> **正确写法（必须）**：
> ```cpp
> p = p_node->BuildValue();
> p_opt = GetScalarValue<float>(p);
> if (p_opt->has_value()) {
>   // 编译期已知 → C++ if/else 分支优化
>   auto p_val = p_opt.value();
>   // ... 按值分支
>   if (p_val...) 
> } else {
>   // 编译期未知 → 用 Conditional 构建运行时分支
>   auto true_branch = [&ib](...) { ... };
>   auto false_branch = [&ib](...) { ... };
>   result = ib->Conditional(cond, true_branch, false_branch);
> }
> ```
>
> 如果算子的反向逻辑确实无法在值 unknown 时推导（极罕见），或者你实在无法处理kbk下的动态输入，
> 必须在验证闭环中**明确记录原因并征求用户确认**，而非默默 throw。

**参考实现**（仓库中的典型写法）：
- `ReduceStd` bprop：`keep_dims` 和 `unbiased` 值 unknown 时用 `Conditional` 做运行时分支
- `MatMulExt` bprop：`IsDynamicRank(x_shape) || IsDynamicShape(w_shape)` 时走独立动态路径

---

## 成功标准

- [ ] BPROP 注册代码已添加
- [ ] 反向 I/O 个数与正向一致
- [ ] 不可微分入参已返回零梯度占位
- [ ] 与 PTA `derivatives.yaml` 的可微输入列表对齐
- [ ] **图模式动态输入场景已处理**（标量 unknown → Conditional；shape 动态 → ShapeCalc/动态路径）
- [ ] 编译通过

---

## 下一步

BPROP 完成后，进入 **[Workflow 7: 导出与占位](./07-export.md)**

> 如果不需要反向，本步骤可跳过，直接进入 Workflow 7。
