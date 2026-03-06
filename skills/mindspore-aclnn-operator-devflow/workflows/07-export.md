# Workflow 7: 导出与占位

## 目标

确保算子通过 ops 包正确导出，非 Ascend 设备有清晰错误占位；接口满足对标 PyTorch 要求（见下）。

## 输入

- **算子实现**：前向/反向的完整代码

## 输出

- **ops 包导出**：`__init__.py` / `__all__` 更新
- **接口文件**：functional / nn / Tensor 方法（按需）
- **占位实现**：非 Ascend 设备的 RuntimeError

**接口对标约束**（来自 2. 接口开发 2.1）：接口名、参数名、顺序与默认值、算法与精度、输入范围（dtype）、性能不低于 PyTorch 的 1/2、Parameter 名/name 等需与 PyTorch 一致；无法一致需经 CCB/接口设计评审。详见 `reference.md` §19。

---

## 执行步骤

### Step 1：ops 包显式导出

- 在 `mindspore/ops/` 相关 `__init__.py` 中**对应算子类别**和 **`__all__` 两处**添加算子名（见 2. 接口开发 2.1.3/2.2）。
- 确保 `__all__` 列表包含新算子

### Step 2：接口开发（`reference.md` §19）

| 接口类型 | 要点 |
| --- | --- |
| **functional** | 内部用 `_get_cache_prim` 获取 Primitive（避免反复 __init__） |
| **nn** | Cell 子类；`construct` 中不直接 `raise`，用 `@constexpr` |
| **Tensor 方法** | 覆盖 PyNative/KBK/GE 模式（按项目要求）。**GE 模式**需在 `resource.cc` 注册映射、`standard_method.py` 实现；该校验函数不能接收 Tensor 类型入参（见 2. 接口开发 2.4） |

### Step 3：非 Ascend 设备占位

- 清晰的 RuntimeError 说明该算子仅支持 Ascend
- 不要让用户遇到难以理解的内部错误

---

## 成功标准

- [ ] ops 包中可正常 import 算子
- [ ] functional/nn/Tensor 接口可用（按需）
- [ ] 非 Ascend 设备给出清晰错误信息
- [ ] `_get_cache_prim` 使用正确（functional 接口）

---

## 下一步

导出完成后，进入 **[Workflow 8: 测试](./08-testing.md)**

> 如果不涉及 functional/nn/Tensor 导出，本步骤可精简。
