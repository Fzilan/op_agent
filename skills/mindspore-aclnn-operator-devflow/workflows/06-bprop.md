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

### Step 3：进阶注意事项（`reference.md` §14）

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

代码骨架见 `reference.md` §24.5。

---

## 成功标准

- [ ] BPROP 注册代码已添加
- [ ] 反向 I/O 个数与正向一致
- [ ] 不可微分入参已返回零梯度占位
- [ ] 与 PTA `derivatives.yaml` 的可微输入列表对齐
- [ ] 编译通过

---

## 下一步

BPROP 完成后，进入 **[Workflow 7: 导出与占位](./07-export.md)**

> 如果不需要反向，本步骤可跳过，直接进入 Workflow 7。
