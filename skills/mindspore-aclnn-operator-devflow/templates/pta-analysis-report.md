# PTA 源码审查报告 - {OpName}

> **文档用途**：记录 PTA（op-plugin）源码审查结果，作为 MS 适配的依据。
> **文档状态**：本地文件，不提交 Git。
> **生成时间**：{generation_time}

---

## 一、基本信息

| 属性 | 值 |
| ---- | -- |
| **PTA 接口名** | `torch_npu.npu_{op_name}` |
| **MS 目标接口** | `mindspore.ops.{op_name}` |
| **对接类型** | 类型 {1/2/3} |
| **torch_npu 版本** | {version} |
| **CANN 版本** | {version} |

---

## 二、前向接口分析

### 2.1 函数签名（来自 op_plugin_functions.yaml）

```yaml
# 摘自 op_plugin/config/op_plugin_functions.yaml
{paste_yaml_entry}
```

### 2.2 参数详情

| 参数名 | 类型 | 必需 | 默认值 | MS 映射 | 备注 |
| ------ | ---- | ---- | ------ | ------- | ---- |
| {param1} | {type} | ✅/❌ | {default} | {ms_name} | |

### 2.3 ACLNN 调用分析（来自 C++ 实现）

**文件**：`op_plugin/ops/opapi/{OpName}KernelNpuOpApi.cpp`

```cpp
// 关键代码摘录
{key_code_snippet}
```

**调用的 ACLNN 接口**：`aclnn{XxxYyy}`
**参数预处理**：{describe_preprocessing}
**输出构造**：{describe_output_construction}
**硬编码参数**：{list_hardcoded_params}

---

## 三、反向接口分析

### 3.1 反向注册（来自 derivatives.yaml）

```yaml
# 摘自 op_plugin/config/derivatives.yaml
{paste_derivatives_entry}
```

### 3.2 可微输入列表

| 输入 | 是否可微 | 备注 |
| ---- | -------- | ---- |
| {input1} | ✅/❌ | |

### 3.3 反向 ACLNN 调用分析

**文件**：`op_plugin/ops/opapi/{OpName}GradKernelNpuOpApi.cpp`

**调用的 ACLNN 接口**：`aclnn{XxxGrad}`
**反向输出**：{list_grad_outputs}
**硬编码参数**：{list_hardcoded_params}

---

## 四、前向 vs 反向差异

| # | 内容 | 前向 | 反向 | MS 适配影响 |
| - | ---- | ---- | ---- | ----------- |
| 1 | {diff_item} | {fwd_value} | {bwd_value} | {impact} |

---

## 五、代码与文档不一致项（如有）

| # | 内容 | 文档描述 | 代码实际行为 | 文件/行号 | 状态 |
| - | ---- | -------- | ------------ | --------- | ---- |
| 1 | {item} | {doc_says} | {code_does} | {file:line} | ⚠️ 待确认 / ✅ 已确认 |

> **处理方式**：不一致项交给用户找接口人确认，拿到结论后继续。

---

## 六、ACLNN 调用链（组合场景填写）

> 如果 PTA 直连单个 aclnnXxx，此节留空。

```text
{OpName} 前向调用链：
  ① aclnn{Sub1}(...) → {intermediate_1}
  ② aclnn{Sub2}(...) → {output}

{OpName} 反向调用链：
  ① aclnn{SubGrad}(...) → {grad_outputs}
```

---

## 七、MS 适配结论

- **对接类型**：{1/2/3}
- **ACLNN 接口**：`aclnn{Xxx}`（前向）/ `aclnn{XxxGrad}`（反向）
- **需要 customize**：{是/否}
- **组合场景**：{是/否}，子算子列表见第六节
- **关键注意事项**：
  - {item1}
  - {item2}
