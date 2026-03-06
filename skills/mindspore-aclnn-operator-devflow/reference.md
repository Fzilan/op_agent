# ACLNN 算子开发全流程参考（细节版）

本文件给 `mindspore-aclnn-operator-devflow` 提供"可按图索骥"的细节与模板。需要时再读取，避免把 `SKILL.md` 撑得过长。

> **关于"来自/来源"标注**：部分章节标题中的 `来自 算子流程/.../xxx.md` 是知识溯源标记，
> 表明该节内容提炼自哪份原始文档。**这些原始文档不随 skill 分发**，不影响本文件的独立使用。

## 目录

- [1. 目录/文件定位](#1-目录文件定位建议用搜索而非硬编码路径)
- [2. YAML 设计模板](#2-yaml-设计模板前向反向各一份)
- [3. gen_ops.py 代码生成机制（基于源码分析）](#3-genopspy-代码生成机制基于源码分析)
- [4. GeneralInfer（C++）推导约定](#4-generalinferc推导约定)
- [5. PyBoost（Pynative）实现要点](#5-pyboostpynative实现要点)
- [6. KBK（Graph）kernel 要点](#6-kbkgraphkernel-要点)
- [7. BPROP 接线要点](#7-bprop-接线要点)
- [8. 测试策略（UT + ST）](#8-测试策略ut--st)
- [9. 文档与导出](#9-文档与导出)
- [10. 交付/转测共识要点](#10-交付转测共识要点)
- [11. 资料开发要点](#11-资料开发要点)
- [12. 性能自验工具 apitimewrapper](#12-性能自验工具-apitimewrapper)
- [13. 开源运作（RFC）流程要点](#13-开源运作rfc流程要点)
- [14. 反向实现注意事项](#14-反向实现注意事项)
- [15. 安全编码与代码检视](#15-安全编码与代码检视)
- [16. Resize/Launch 优化要点](#16-resizelaunch-优化要点)
- [17. 精度零偏差与显存对齐自验](#17-精度零偏差与显存对齐自验)
- [18. 用 Cursor 辅助分析 PyTorch 算子](#18-用-cursor-辅助分析-pytorch-算子)
- [19. 接口开发要点（functional / nn / Tensor）](#19-接口开发要点functional--nn--tensor)
- [20. 问题处理与"书面结论"](#20-问题处理与书面结论)
- [21. 质量门禁与格式要求](#21-质量门禁与格式要求)
- [22. 当 ACLNN/PTA 文档不完善：用"探测脚本"补齐事实范围](#22-当-aclnnpta-文档不完善用探测脚本补齐事实范围)
- [23. vmap 支持（按需）](#23-vmap-支持按需)
- [24. 代码骨架模板（可直接复制改造）](#24-代码骨架模板可直接复制改造)
- [25. PTA 源码审查方法（必做）](#25-pta-源码审查方法必做)
- [26. InferValue 常量折叠（可选优化）](#26-infervalue-常量折叠可选优化)
- [27. 动态 shape 分类与处理策略](#27-动态-shape-分类与处理策略)
- [28. ACLNN 调用链分析与子算子盘点（组合场景）](#28-aclnn-调用链分析与子算子盘点组合场景)
- [29. 组合实现模式（PyBoost/KBK 多 ACLNN 串联）](#29-组合实现模式pyboostkbk-多-aclnn-串联)
- [§30. Feature 文档（评审与交付必须产物）](#30-feature-文档评审与交付必须产物)
- [31. Skill 维护策略](#31-skill-维护策略)

---

## 1. 目录/文件定位（建议用搜索而非硬编码路径）

MindSpore / op-plugin 的目录在不同分支可能不一致，优先用搜索定位：
- 通过字符串搜索：`gen_ops.py`、`LAUNCH_ACLNN`、`MS_ACLNN_KERNEL_FACTORY_REG`、`REG_BPROP_BUILDER`。
- 通过相似算子对照：按目标算子特征分类，在仓库中搜索已接入的同类算子（详见 §2.4）。

常见的"目标区域"（仅作方向提示）：
- **YAML**：`mindspore/ops/op_def/yaml/`
- **推导/元实现**：`mindspore/` 下 `ops` / `infer` / `ops_func_impl` 等目录（以实际仓库为准）
- **Ascend kernel / PyBoost / KBK**：`mindspore/ccsrc/` 与 `op-plugin-*/` 内的 `ascend`/`kernel`/`aclnn`/`customize`
- **bprop**：`mindspore/ccsrc/` 下 `bprop` / `grad_*ops.cc`
- **测试**：`tests/ut/`、`tests/st/`
- **文档**：英文 function_doc 的 YAML + 中文 `docs/api/api_python/ops/*.rst`

### 1.1 CMake 构建（基于源码分析：新增算子无需改 CMake）

MindSpore 的 `ops/` 构建系统使用 `merge_ops_files()` + `file(GLOB_RECURSE ...)` 自动收集源文件，
**新增算子只需把文件放到正确目录，不需要修改任何 CMake 文件**。

| 目录 | 收集方式 | 说明 |
| --- | --- | --- |
| `ops/infer/ops_func_impl/` | `merge_ops_files` 合并 | Infer 实现 |
| `ops/kernel/ascend/aclnn/pyboost_impl/customize/` | `merge_ops_files(customize)` 单独合并 | PyBoost Customize |
| `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/` | 与 kernel_mod_impl 一起被 `merge_ops_files` 合并 | KBK Customize |
| `ops/kernel/ascend/aclnn/pyboost_impl/auto_generate/` | GLOB 收集 | PyBoost 自动生成 |
| `ops/kernel/ascend/aclnn/kernel_mod_impl/aclnn_auto_gen/` | GLOB 收集（.gitignore 中，不提交） | KBK 自动生成 |

关键 CMake 文件（仅供理解，不需要修改）：
- `ops/CMakeLists.txt`：顶层，include `merge_ops.cmake`
- `ops/cmake/merge_ops.cmake`：`merge_ops_files()` 定义
- `ops/kernel/ascend/aclnn/CMakeLists.txt`：`add_subdirectory(pyboost_impl)` + `add_subdirectory(kernel_mod_impl)`

## 2. YAML 设计模板（前向/反向各一份）

### 2.1 最小一致性原则
同一个参数（例如 `actual_seq_len`）必须在以下位置**一致**：
- YAML（op_def + api_def + function_doc）
- GeneralInfer（C++ 推导）
- PyBoost（Pynative 调用）
- KBK（Graph kernel 取参/Launch）
- 文档（中英文）
- UT/ST（覆盖参数边界与异常路径）

### 2.2 Customize 后缀
若你走的是项目默认 ACLNN kernel 机制，**一般不需要在 YAML 手工加 Customize 后缀**（框架会自动处理）。

### 2.3 两条接入路径（核心决策——决定整个开发工作量）

ACLNN 算子接入的**最关键决策**是：MindSpore API 的参数能否原样透传给 ACLNN 接口？
这决定了走**自动生成路径**还是**手动 Customize 路径**，直接影响需要写哪些文件。

#### 路径 1：自动生成（参数直通，不需要 Customize）

**适用条件**：MindSpore API 参数与 ACLNN 接口参数完全一致——
参数个数、顺序、类型、默认值都不需要在调用前做任何转换。

**YAML 配置关键**：`op_def` 中 `dispatch: enable: True`，**不写** `Ascend:` 字段。
框架内部 `Ascend` 默认为 `'default'`，走自动生成路径。

```yaml
# 路径 1 示例（如 abs、mul、trunc）
dispatch:
  enable: True
  # 不写 Ascend 字段 → 自动生成
```

**编译时自动生成**：
- PyBoost 调用代码（`pyboost_ascend_call_template.tpl` → `LAUNCH_ACLNN(aclnnXxx, ...)`）
- KBK 注册（`MS_ACLNN_COMMON_KERNEL_FACTORY_REG` → `aclnn_kernel_register_auto.cc`）
- Python 接口包装（`functional_overload.py` 等）

**开发者需要手写的文件**：
| 文件 | 对应步骤 |
| --- | --- |
| `op_def/yaml/xxx_op.yaml` | Step 1 |
| `api_def/xxx.yaml` | Step 1 |
| `op_def/yaml/doc/xxx_doc.yaml`（`_ext` 风格）或 `api_def/function_doc/xxx_doc.yaml`（旧风格） | Step 1 |
| `infer/ops_func_impl/xxx.h` + `.cc` | Step 3 |
| `aclnn_config.yaml` 添加映射（修改） | Step 2 |
| `math_func.py` / `mint/__init__.py` / `tensor_method.py` 导出（修改） | Step 7 |
| `tests/ut/cpp/ops/test_xxx.cc` | Step 8 |
| `tests/st/ops/ascend/test_xxx.py`（+ `tests/st/mint/`、`tests/st/tensor/overload/`） | Step 8 |
| 英文 function_doc + 中文 RST（每个接口形态一份） | Step 9 |

**不需要写**：PyBoost customize 文件、KBK customize 文件（**跳过 Step 4 和 Step 5**）。

**实例**：`abs`、`mul`、`trunc`、`xlogy`、`div`（基础算术）。

#### 路径 2：手动 Customize（参数需要预处理）

**适用条件**：调用 ACLNN 前需要做参数转换，常见情况：
- `tuple[int]` → `std::vector<int64_t>`（如 `actual_seq_qlen`）
- `Optional[Tensor]` 的 None 语义需特殊处理
- `str` → enum/int 转换（如 `layout: "BSND"` → 整型编码）
- 标量参数提取（从 Value 中取值）
- 多个输入需要重排序/合并后传入 ACLNN
- 输出 Tensor 需要手动分配（shape 与输入不同）

**YAML 配置关键**：`dispatch: enable: True` + `Ascend: XxxAscend`（显式指定 Customize 类名）。

```yaml
# 路径 2 示例（如 dense_lightning_indexer_grad_kl_loss）
dispatch:
  enable: True
  Ascend: DenseLightningIndexerGradKlLossAscend
```

**编译时**：gen_ops.py 生成包装代码，该包装代码调用你手写的 Customize 类
（`pyboost_ascend_customize_call_template.tpl` → `XxxAscendCustomize(...)`）。

**开发者需要额外手写的文件**（在路径 1 基础上）：
| 文件 | 对应步骤 |
| --- | --- |
| `kernel/.../pyboost_impl/customize/xxx.h` + `.cc` | Step 4 |
| `kernel/.../kernel_mod_impl/customize/xxx_aclnn_kernel.h` + `.cc` | Step 5 |
| （如有反向）上述文件的 `_grad` 版本 | Step 4/5 |

**实例**：`dense_lightning_indexer_grad_kl_loss`、`multi_scale_deformable_attn`、`conv2d_ext`、`add`。

#### 路径决策流程图

```
分析 MindSpore API 参数 vs ACLNN 接口参数
                │
      参数能否原样透传？
       ╱              ╲
      是               否
      │                │
  路径 1（自动）    路径 2（Customize）
      │                │
  YAML 不写           YAML 写
  Ascend 字段     Ascend: XxxAscend
      │                │
  跳过 Step 4/5    必须写 Step 4/5
      │                │
  编译自动生成     编译调用你的
  PyBoost/KBK      Customize 类
```

#### "对接类型"三分类与路径的对应关系

| 对接类型 | 描述 | 对应路径 |
| --- | --- | --- |
| **类型 1** | API 定义与 ACLNN 完全一致 | **路径 1**（自动生成） |
| **类型 2** | 名称不同但功能一致 | 通常**路径 1**（通过 YAML 的 `class` 字段做名称映射） |
| **类型 3** | 原型/语义不一致 | **路径 2**（必须手动 Customize） |

> **注意**：类型 2 是否需要 Customize 取决于"名称不同"是否仅限于算子名映射。
> 如果只是名字不同但参数完全一致，路径 1 即可（YAML 的 `class` 字段做映射）；
> 如果还涉及参数顺序/类型差异，仍需走路径 2。

### 2.4 相似算子查找策略（不要硬编码算子名）

开发中需要参照"已接入的相似算子"来确认代码风格、目录结构、宏用法。**不要默认指定某几个算子名**，
而是先分析目标算子的特征，再在仓库中搜索匹配的同类算子。

#### 分类维度（按优先级排列）

#### A. 功能/算法类别（最直觉的分类——同一类算子的实现模式往往高度相似）

| 类别 | 典型算子 | 共性特征 |
| --- | --- | --- |
| **Attention 族** | flash_attention、nsa_compress_attention、paged_attention、incre_flash_attention | TND/BSND 布局、多输出（softmax_max/sum）、带 mask/actual_seq_len、独立 Grad 算子 |
| **Loss 族** | cross_entropy、cosine_embedding_loss、ctc_loss、nll_loss | 前向输出 loss + 中间缓存（log_sum_exp 等）、reduction 参数（none/mean/sum）、反向需中间值 |
| **Norm 族** | layer_norm、group_norm、rms_norm、batch_norm | 输入 + weight + bias 三件套、running_mean/var 状态、rstd 中间输出、反向输出 dx/dw/db |
| **Optimizer 族** | adam、sgd、lamb、adamw | 就地更新（副作用算子）、lr/beta/epsilon 标量参数、多 Tensor 输入（param/grad/m/v）、通常无反向 |
| **激活函数族** | relu、gelu、silu、swish、leaky_relu | 逐元素、单输入单输出、反向简单（乘 mask 或导函数）、通常类型 1 直连 |
| **逐元素算术族** | add、mul、div、eq、ne、gt | 逐元素、支持广播、支持 Tensor-Scalar 重载、符号重载（`__add__`/`__eq__`）、多态分发 |
| **Reduce 族** | sum、mean、prod、amax、argmax | 沿指定 axis 缩减、keepdim 参数、输出 shape 少一个或多个维度、部分有反向（sum/mean）部分无（argmax） |
| **矩阵运算族** | matmul、bmm、linear、baddbmm | 2D/3D 矩阵乘、transpose 参数、alpha/beta 系数、输出 shape 由矩阵乘法规则决定 |
| **索引/gather 族** | index_select、gather、scatter、embedding | 索引 Tensor 输入、不规则 shape 推导、反向是 scatter/zero-fill 模式 |
| **变形/排列族** | reshape、transpose、permute、contiguous | 通常不涉及 ACLNN 计算（纯 shape 变换）、不需要反向或反向是逆变换 |
| **卷积/池化族** | conv2d、avg_pool2d、max_pool2d | kernel_size/stride/padding/dilation 四参数组、NCHW/NHWC 布局、反向有独立 Grad 算子 |
| **通信/并行族** | all_reduce、all_gather、reduce_scatter | 集合通信、group 参数、副作用算子、通常无标准 ACLNN（走 HCCL） |

> **用法**：先判断目标算子属于哪个族，然后在仓库中搜索同族已接入的算子。
> 同族算子的 Infer 推导逻辑、PyBoost/KBK 调用模式、bprop 接线方式、测试覆盖策略往往高度相似，
> 是最有价值的参照对象。

#### B. 技术实现特征（辅助筛选——在同族内进一步缩小范围）

| 维度 | 典型分类 | 搜索关键词/方法 |
| --- | --- | --- |
| **输入布局** | TND / BSND / BNSD / 标准逐元素 | 在 `op_def/yaml/` 中 grep 相同 shape 注释 |
| **ACLNN 对接方式** | 单 ACLNN 直连 / 多 ACLNN 组合 / 无 ACLNN（纯 Python 组合） | grep `LAUNCH_ACLNN` 数量；组合算子看 customize 目录 |
| **是否有反向** | 有独立 Grad 算子 / 自动微分 / 无反向 | grep `REG_BPROP_BUILDER` + grep `_grad` YAML |
| **接口形态** | functional only / functional + nn / functional + tensor / 符号重载 | 看 api_def YAML 的 `interface` 字段 |
| **参数特殊性** | 含 Optional[Tensor] / tuple[int] / 枚举(layout/mode) / 标量 | 看 YAML 的 `default: None` / `type_cast` / `arg_handler` |
| **对接类型** | 类型 1（完全一致）/ 类型 2（名称映射）/ 类型 3（需 customize） | 对照 §2.3 判断 |

#### 查找流程

1. **判断功能/算法类别**：目标算子属于上面哪个族？
   - 例：`nsa_compress_attention` → **Attention 族**
   - 例：`cosine_embedding_loss` → **Loss 族**
   - 例：`eq`（== 重载）→ **逐元素算术族**

2. **确定技术特征标签**：从 B 表中选出 2-3 个显著特征，在同族内进一步筛选。
   - 例：`nsa_compress_attention` → Attention 族 + TND 布局 + 单 ACLNN 直连 + 有独立 Grad + 含 tuple[int]
   - 例：`cosine_embedding_loss` → Loss 族 + 多 ACLNN 组合 + 无单独 Primitive + functional + nn + reduction 参数
   - 例：`adamw` → Optimizer 族 + 就地更新（副作用）+ 多 Tensor 输入 + 无反向

3. **在仓库中搜索同类**：
   ```bash
   # 按功能族名找：搜索同族算子（如 attention 族）
   grep -rl "attention" mindspore/ops/op_def/yaml/ --include="*.yaml"

   # 按布局找：搜索含相同 shape 模式的算子
   grep -r "TND" mindspore/ops/op_def/yaml/ --include="*.yaml" -l

   # 按 ACLNN 组合找：搜索 customize 目录下含多个 LAUNCH_ACLNN 的文件
   grep -rl "LAUNCH_ACLNN" mindspore/ops/kernel/.../customize/

   # 按反向模式找：搜索有 Grad 后缀 YAML 的算子
   ls mindspore/ops/op_def/yaml/*_grad_op.yaml

   # 按接口形态找：搜索同时有 tensor + function 接口的算子
   grep -l "interface:.*tensor.*function" mindspore/ops/api_def/*.yaml

   # 按 reduction 参数找（loss 族常见）
   grep -l "reduction" mindspore/ops/op_def/yaml/*.yaml
   ```

4. **选择 2-3 个最匹配的算子**，逐目录对照其 YAML/Infer/PyBoost/KBK/bprop/测试/文档的写法。
   优先选**同族 + 技术特征最接近**的；其次选**不同族但技术特征（对接类型/参数模式）相似**的。

5. **如果搜不到高度匹配的同类**（全新类型算子），退而选择"对接类型"（§2.3）相同的任意算子作为
   代码风格参考，同时在实现过程中更谨慎地逐步验证。

> **原则**：相似算子是"代码风格和结构的参照"，不是"功能逻辑的抄写对象"。
> 功能逻辑以 PTA 源码 + ACLNN 文档为准，相似算子只用来确认目录结构、宏名、注册方式、测试写法等。

### 2.5 dispatch + "先自动生成，再拷贝改造"的实用套路
当你需要自定义 PyBoost/KBK 时，一个高效做法是：
1. 在 YAML 里打开 `dispatch.enable: True`。
2. **临时注释掉** YAML 中 `dispatch.Ascend: XxxAscend` 这类自定义声明，让 `gen_ops.py` 先生成一份可编译骨架。
3. 将生成目录里的 `.h/.cc` **拷贝**到 `customize` 目录（或对应自定义目录）。
4. 按 ACLNN 实际签名调整入参（例如删除 ACLNN 不需要的 dtype、处理 tuple→vector 等）。
5. 按项目约定重命名入口（常见模式：`OpNameAscendCustomize` / `OpNameGradAscendCustomize`），恢复 YAML 声明。
6. 删除临时自动生成文件，只保留自定义实现。

## 3. gen_ops.py 代码生成机制（基于源码分析）

**脚本位置**：`mindspore/python/mindspore/ops_generate/gen_ops.py`
**调用方式**：CMake 构建时由 `cmake/gencode.cmake` 自动调用，也可手动执行：
```bash
python mindspore/python/mindspore/ops_generate/gen_ops.py  # 在 MindSpore 根目录执行
```

### 3.1 YAML 允许字段（源自 `gen_constants.py`）

| 顶层字段 | 子字段 | 必填 | 说明 |
| --- | --- | --- | --- |
| `args` | `dtype`, `default`, `prim_init`, `type_cast`, `arg_handler`, `disable_tensor_to_scalar` | 是 | 参数定义 |
| `returns` | `dtype`, `inplace`, `type_cast` | 是 | 返回值定义 |
| `dispatch` | `enable`, `is_comm_op`, `Ascend`, `InternalOpAscend`, `GPU`, `CPU` | 否 | 无则跳过 PyBoost/KBK/auto_grad 生成 |
| `function` | `name`, `disable` | 否 | `disable: True` 时不生成 `gen_ops_def` 中的函数 |
| `class` | `name`, `disable` | 否 | `disable: True` 时不生成 `gen_ops_prim` 中的类 |
| `args_signature` | `rw_write`, `rw_read`, `rw_ref`, `dtype_group` | 否 | 参数签名 |
| `view` | - | 否 | `True` 时走 view 算子特殊逻辑 |
| `composite` | - | 否 | `True` 时跳过 PyBoost 生成 |
| `bprop_expander` | - | 否 | 默认 `True`，使用 bprop expander |
| `non-differentiable` | - | 否 | 不可微分 |
| `labels` | - | 否 | 标签 |

### 3.2 路径 1/2 的代码级判断

核心条件：`dispatch.Ascend` 的值（源自 `pyboost_op_cpp_code_generator.py`）。

- **`'default'`（未写 Ascend 字段时的默认值）**→ 路径 1：`AclnnOpCppCodeGenerator.generate_aclnn_op_cpp_code`
- **非 `'default'` 且非 `'None'`**→ 路径 2：`PyboostOpCppGenerator.generate_customize_op_cpp_code`，生成对 `{Ascend}Customize` 的调用
- **`'None'`**→ 该设备不生成任何代码

KBK 注册（`aclnn_kernel_register_auto_cc_generator.py`）：只在 `ascend == 'default'` 时自动生成并注册。

### 3.3 生成文件与目标路径

| 类别 | 生成文件 | 目标路径 |
| --- | --- | --- |
| Python Primitive | `gen_ops_prim.py` | `python/mindspore/ops/auto_generate/` |
| Python 函数接口 | `gen_ops_def.py` | `python/mindspore/ops/auto_generate/` |
| C++ op_def | `gen_ops_def.cc/.h` | `ops/op_def/auto_generate/` |
| C++ primitive | `gen_ops_primitive_*.h` | `ops/include/primitive/auto_generate/` |
| PyBoost 核心 | `pyboost_core.cc` | `ccsrc/pynative/forward/pyboost/auto_generate/` |
| PyBoost Ascend | `pyboost_ascend_ops_*.cc` | `ops/kernel/ascend/aclnn/pyboost_impl/auto_generate/` |
| KBK 注册 | `aclnn_kernel_register_auto.cc` | `ops/kernel/ascend/aclnn/kernel_mod_impl/auto_generate/` |
| KBK KernelMod（路径1） | `{op}_aclnn_kernel.h/.cc` | `ops/kernel/ascend/aclnn/kernel_mod_impl/aclnn_auto_gen/` |

### 3.4 常见报错与方向

- **keys 结构不匹配**：对照 `gen_constants.py` 中的 `OP_KEYS`/`ARG_KEYS`/`DISPATCH_KEYS` 检查字段名
- **function_doc 缺条目**：补齐对应的 doc YAML，保持参数一致
- **Windows 编码**：英文 YAML 文档不要混入中文字符

## 4. GeneralInfer（C++）推导约定（基于源码分析）

### 4.1 职责边界
- 只做**形状/类型推导**；不要做运行时合法性校验（交给 ACLNN/运行时）。
- 报错使用框架异常宏，错误信息要包含：参数名、期望、实际。

### 4.2 框架入口与实现位置
- **入口**：`ops::DoGeneralInfer(prim, abstract_list, frontend_func_impl)`（`core/ops/infer_info/infer_info_utils.cc`）
- **算子实现**：`mindspore/ops/infer/ops_func_impl/{op_name}.cc`
- **InferInfo 基类**：`core/include/ops/infer_info/infer_info.h`

算子需实现 `OpFuncImpl` 的 `InferShape` 和 `InferType`：
```cpp
BaseShapePtr InferShape(const PrimitivePtr &prim,
                        const std::vector<AbstractBasePtr> &input_args) const override;
TypePtr InferType(const PrimitivePtr &prim,
                  const std::vector<AbstractBasePtr> &input_args) const override;
```

### 4.3 常用 InferInfo API（源码确认的签名）

| API | 头文件 | 签名 | 用途 |
| --- | --- | --- | --- |
| `GetScalarValueWithCheck<T>()` | `core/include/ops/infer_info/infer_info.h` | `T GetScalarValueWithCheck()` | 取标量（失败则抛异常） |
| `GetArrayValue<T>()` | `core/include/utils/value_utils.h` | `std::optional<ArrayValue<T>>` | 取 tuple/list |
| `HasUnknownValue()` | 同上（`ArrayValue` 方法） | `bool HasUnknownValue() const` | 判断是否含 unknown 元素 |
| `IsNone()` | `core/include/ops/infer_info/infer_info.h` | `virtual bool IsNone() = 0` | 判断 None |
| `CheckAndConvertUtils::*` | `core/include/utils/check_convert_utils.h` | 静态方法 | `CheckInteger`、`CheckTypeValid` 等 |

### 4.4 动态 shape / 动态 rank
> 完整三分类（InputDynamic / OutputDynamic）见 §27。

关键常量（`abstract::Shape` 中定义）：
- **动态维**：`kShapeDimAny`（-2）——某个维度未知
- **动态秩**：`kShapeRankAny`（-1）——维度数量未知

推荐策略：
- 动态 rank：返回 `ShapeVector{kShapeRankAny}`
- 关键参数（如 block/stride/seq_len）出现 unknown 时，对应维度回退为 `kShapeDimAny`
- 关键参数都已知时，返回精确 shape

典型写法：
```cpp
auto value_opt = GetArrayValue<int64_t>(input_args[idx]);
if (!value_opt.has_value() || value_opt.value().HasUnknownValue()) {
  return std::make_shared<abstract::TensorShape>(ShapeVector{kShapeDimAny});
}
auto vec = value_opt.value().ToVector();
```

## 5. PyBoost（Pynative）实现要点（基于源码分析）

### 5.1 目录结构

| 场景 | 路径 | 命名 |
| --- | --- | --- |
| 路径 1（自动生成） | `ops/kernel/ascend/aclnn/pyboost_impl/auto_generate/` | `pyboost_ascend_ops_*.cc` |
| 路径 2（Customize） | `ops/kernel/ascend/aclnn/pyboost_impl/customize/` | `{op_name}.cc` / `{op_name}.h` |

### 5.2 注册宏
```cpp
// ccsrc/include/pynative/utils/pyboost/op_register.h
MS_REG_PYBOOST_OP(Ascend, OpName);  // 将 OpNameAscend 注册到 OpFactory
```

### 5.3 Customize 函数签名
```cpp
// 单输出
tensor::TensorPtr {OpName}AscendCustomize(
    const std::shared_ptr<OpRunner> &op, const TensorPtr &arg1, ...);
// 多输出
std::vector<tensor::TensorPtr> {OpName}AscendCustomize(
    const std::shared_ptr<OpRunner> &op, ...);
```

### 5.4 标准实现流程（6 步）
```cpp
tensor::TensorPtr XxxAscendCustomize(const std::shared_ptr<OpRunner> &op, ...) {
  // 1. 推断输出 shape/dtype
  OpRunner::InferOpOutput(op, arg1, arg2, ...);
  // 2-3. 准备设备地址
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), ...);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), ...);
  // 4-6. 异步调度：分配显存 + 调用 ACLNN
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, ...]() {
      PyBoostUtils::MallocOpInputs(op->device_context(), op->stream_id(), ...);
      PyBoostUtils::MallocOpOutputs(op->device_context(), op->stream_id(), ...);
      LAUNCH_ACLNN(aclnnXxx, op->device_context(), op->stream_id(), ...);
    }));
  return op->output(0);
}
```

### 5.5 LAUNCH_ACLNN 宏
定义在 `ops/kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h`，内部通过 ACLNN executor 调用对应 `aclnn*` C 接口。

### 5.6 调用链（Python → ACLNN）
```
Python: ops.xxx(...)
  → pybind: xxx_Base → xxx_OP → DispatchOp
    → kernel::pyboost::Xxx → OpFactory<Xxx>::Create(Ascend)
      → XxxAscend::Call()
        → XxxAscendCustomize(op, args...)  [路径2]
        → 或直接 LAUNCH_ACLNN(aclnnXxx)    [路径1]
```

### 5.7 输入参数转换
- tuple/list：统一转为 `std::vector<int64_t>` 再传给 ACLNN
- 可选输入：若允许 None，需定义"None 语义"，并在 PyBoost/Infer/KBK 同步处理

## 6. KBK（Graph）kernel 要点（基于源码分析）

> Init/Resize/Launch 职责分离、无意义输出、compute-depend 输出等优化要点见 §16。

### 6.1 目录结构

| 场景 | 路径 | 命名 |
| --- | --- | --- |
| 路径 1（自动生成） | `ops/kernel/ascend/aclnn/kernel_mod_impl/aclnn_auto_gen/` | `{op}_aclnn_kernel.h/.cc` |
| 路径 2（Customize） | `ops/kernel/ascend/aclnn/kernel_mod_impl/customize/` | `{op}_aclnn_kernel.h/.cc` |
| 基类与宏 | `ops/kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h` | - |

### 6.2 基类与必须重写的方法

基类：`AclnnKernelMod`（定义在 `aclnn_kernel_mod.h`）

```cpp
class XxxAscend : public AclnnKernelMod {
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                        const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs,
              const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs,
              void *stream_ptr) override;
 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()  // 单输出宏
  // 多输出用 DEFINE_GET_WORKSPACE_FOR_OPS
};
```

`Init` 和 `Resize` 由基类处理，一般不需要重写。

### 6.3 注册宏

| 宏 | 用途 | 示例 |
| --- | --- | --- |
| `MS_ACLNN_KERNEL_FACTORY_REG(NAME, CLASS)` | Customize kernel 注册 | `MS_ACLNN_KERNEL_FACTORY_REG(Conv2DExt, Conv2DExtAscend)` |
| `MS_ACLNN_COMMON_KERNEL_FACTORY_REG(NAME, TYPE, N)` | 路径 1 通用注册（模板类） | `MS_ACLNN_COMMON_KERNEL_FACTORY_REG(RealDiv, aclnnDiv, 3)` |

### 6.4 输入取参方式
KBK 通过 `KernelTensor` 按索引取参，需要定义索引常量：
```cpp
constexpr size_t kInputQueryIndex = 0;
constexpr size_t kInputKeyIndex = 1;
// ...
auto query = device::ascend::ConvertKernelTensor<int64_t>(inputs[kInputQueryIndex]);
```

### 6.5 强约束
- 前向/反向**分文件、分注册**
- 头/实现命名空间保持一致（否则"未声明/未定义"）
- `aclnn_auto_gen/` 目录在 `.gitignore` 中，生成产物不提交

## 7. BPROP 接线要点（基于源码分析）

> 反向实现的进阶注意事项（OutZeros/ZerosLikeExt/inplace/Depend）另见 §14。

### 7.0 注册位置与宏
- **实现目录**：`ccsrc/frontend/expander/grad/grad_*.cc`（按算子类别分文件）
- **宏定义**：`ccsrc/frontend/expander/bprop/bprop_irbuilder.h`

```cpp
REG_BPROP_BUILDER("OpName").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(i0);
  // ... 构建反向子图
  return {grad_x, ib->OutZeros(y), ...};
});
```

关联方式：以前向 Primitive 名字符串为 key（`REG_BPROP_BUILDER("SparseFlashAttention")`）。

### 7.1 ib-> 常用 API

| API | 作用 |
| --- | --- |
| `ib->GetInput(iN)` | 获取第 N 个输入 |
| `ib->Emit("OpName", {args...}, attrs)` | 发射一个 CNode（调用算子） |
| `ib->OutZeros(node)` | 返回全零梯度（等价 `ZerosLike`） |
| `ib->TupleGetItem(tuple, i)` | 从 tuple 取第 i 个元素 |
| `ib->Value(val)` | 创建标量/常量 ValueNode |
| `ib->MakeTuple(inputs)` | 构造 Tuple |
| `ib->ZerosLike(node)` / `ib->ZerosLikeExt()` | 同形状零张量 |
| `ib->Add/Sub/Mul/Div` | 算术运算 |
| `ib->Reshape/Transpose/Cast` | 张量变换 |
| `ib->GetShape/GetDtype/GetRank` | 形状与类型信息 |

### 7.2 反向输入/输出个数规则
- **反向输入个数** = 正向输入个数 + 2（`out` 与 `dout`）
- **反向输出个数** = 正向输入个数（每个输入一个梯度）
- 多输出正向算子：`out` 在反向侧是 tuple，用 `TupleGetItem` 取对应输出

### 7.3 SetUnusedInputs（⚠️ 已标记 DEPRECATED）

源码中 `SetUnusedInputs` 已标记为 DEPRECATED（`bprop_irbuilder.h`）。
当前仍可使用，但**推荐使用替代 API**：

| 旧 API | 替代 API | 说明 |
| --- | --- | --- |
| `SetUnusedInputs({i5, i16})` | `FreeUselessValues_I({i5, i16})` | 释放未用输入 |
| - | `FreeUselessValues_O({o0})` | 释放未用输出 |
| - | `FreeUselessValues_IO(in, out)` | 同时指定输入和输出 |
| - | `FreeUselessValues(func)` | 自定义 `PynativeCallback` 释放逻辑 |

语义：标记 bprop 中未使用的输入索引，PyNative 下提前释放设备内存以降低峰值。

## 8. 测试策略（UT + ST）

### 8.1 Python UT（ops 层）
- 推导正确性：shape/type、动态/边界
- 错误路径：非法参数、None 语义（若不支持 None，要覆盖抛错）
- 固定随机种子：例如 `np.random.default_rng(seed)`

### 8.2 C++ UT（GeneralInfer）
典型构造（按项目现有 UT 工具）：
- 标量：`ShapeVector{}` + `CreateScalar<T>(value)`
- tuple：`ShapeArray{{}}` + `ValuePtrList{CreateScalar<...>(...)}`
- None：`kMetaTypeNone` + `kNone`
- unknown：使用 `kValueAny` 或项目等价占位

### 8.3 Ascend ST（对齐 torch_npu/参考实现）
- 优先"形状/类型"再比数值
- 需要严格对齐时可设 `atol/rtol=0`（以算子数值特性为准）
- 避免引入额外算子导致误差累积（例如反向里不要多余的 sum）
- bfloat16 比较前升精到 float32（避免 numpy bf16 限制）
- **数值验证标准**：**与 PTA 的对比**在 Step 4 精度零偏差验证中完成（固定种子 + md5sum 对比 MS/PTA 输出）。Step 2 的 ST 数值基线为 torch CPU（allclose）或由 Step 4/用户脚本生成的 reference；若用 atol/rtol 需与 PTA 接口人确认。
- **无 torch 大算子时的基线（小算子拼接）**：若不存在对应 torch CPU 大算子，需用小算子拼接时，**先**按业界通用或数学定义实现（如标准 KL 梯度、softmax 等，仅用 torch 小算子），保证形状与梯度结构正确；在代码中注释并提醒用户：**向 PTA/验收方索取「验证交付时使用的 torch CPU 小算子拼接实现」与「验收标准」**，拿到后可提供给 AI 用于替换当前参考并收紧对比。

## 9. 文档与导出

> 资料开发的详细规范（文件命名、接口列表字母序、中英文一致等）见 §11。

强一致性：
- 英文 function_doc（YAML）与中文 RST 参数名、默认值、必填/可选、示例必须一致。
- ops 包显式导出算子 API；非 Ascend 设备提供占位实现并给出清晰错误。

## 10. 交付/转测"共识要点"（来自 `算子流程/.../Aclnn算子对接开发整体流程.md`）

### 10.1 适配方案的基本原则
- **优先对标 PyTorch/PTA**：PTA 不支持的功能可以不开发。
- CANN 不支持且 PTA 也不支持的功能可以不开发。
- 正反向尽量采用与 PTA 相同的 ACLNN/aten 组合，便于达成"精度 0 偏差"的目标。

### 10.2 影响面评估
若可能影响 GEOP / GPU / CPU / Lite 现有流程，需要给出消除影响方案（例如通过 Pass/Expander）。

### 10.3 交付件与验证范围（摘要）
转测时通常要求覆盖：
- 接口形态：NN / functional / Tensor（若完全一致可只验一个入口）
- 后端：Ascend +（GPU/CPU 不回退，若本来不支持则说明）
- 模式：动态图 / 静态图 / 静态图 KernelByKernel
- shape：动态/静态
- 维度：泛化性（dtype/shape）+ 精度 + 性能（按项目门槛要求）

## 11. 资料开发要点（来自 `算子流程/.../5. 资料开发指导.md`）

> 文档导出的基本要求见 §9。

### 11.1 总原则
- **中英文严格一致**：参数、默认值、必选/可选、约束、示例等必须一致。
- **接口列表按字母序添加**：减少冲突与重复。
- **文件名 / 文件内标题 / 文件内接口定义三者一致**：不一致会导致页面生成失败。
- 示例需要**完整 import**、确保可运行；必要时打印输出或 shape 便于理解。

### 11.2 常见场景与落点（摘要）
- 新增 functional：英文注释在实现 `.py`；中文在 `docs/api/api_python/ops/` 下 `func_*.rst`；并更新接口列表。
- 新增 mint：需要同时处理 mint 的中英文列表与中文 rst（若是 import 原有接口可复用）。
- 新增 Tensor 方法：英文在 `tensor.py`，中文在 `docs/api/api_python/mindspore/Tensor/`，并更新列表。

### 11.3 开始前了解事项（来自参考资料 5.1）
1. **明确开发场景**：在 §11.4 表格中找到对应场景，完成该场景下的英文 + 中文 + 接口列表全部任务；**所有文档修改建议同一 PR**（接口列表易冲突可另起 PR，但需同日合入）。
2. **写作前必读**：参考资料中的中英文 API **内容要求**与**格式要求**链接，避免低级问题。
3. **中英文内容一致**：参数、默认值、必选/可选、示例等必须一致；格式按中英文各自要求（见 §11.6）。
4. **自检与预览**：提交 PR 后可在 PR 下评论 `/build_api_doc` 生成官网风格预览（mint 对接开发中）。
5. **检视**：资料完成后找 API 文档负责人检视；**低级问题过多不接收检视**，提交前先自检。
6. **友商参考**（避免思维定式）：如 tf.raw_ops.ScatterNd、torch.nn.Conv1d。
7. **合入后自检**：PR 合入约两日后，到官网 Python API br_base / C++ API br_base 页面确认自己写的 API 展示正确，再转测。

### 11.4 按场景落点（六种开发场景，来自参考资料 5.2）

| 场景 | 英文 API 位置 | 英文接口列表 | 中文 RST 位置 | 中文接口列表 |
| --- | --- | --- | --- | --- |
| **mint** | 原 ops/nn 实现 .py 或 yaml | `docs/api/api_python_en/mindspore.mint.rst` | `docs/api/api_python/mint/` 下以接口名命名的 rst，**仅 mint 模块需 func_ 前缀** | `docs/api/api_python/mindspore.mint.rst` |
| **functional** | 接口实现 .py | `docs/api/api_python_en/mindspore.ops.rst` | `docs/api/api_python/ops/` 下 **func_ 前缀** rst | `docs/api/api_python/mindspore.ops.rst` |
| **nn** | 接口实现 .py | `docs/api/api_python_en/mindspore.nn.rst` | `docs/api/api_python/nn/` 下以接口名命名 | `docs/api/api_python/mindspore.nn.rst` |
| **Tensor 方法** | `mindspore/python/mindspore/common/tensor.py` | `docs/api/api_python_en/mindspore/mindspore.Tensor.rst` | `docs/api/api_python/mindspore/Tensor/`，method/property 分开 | `docs/api/api_python/mindspore/mindspore.Tensor.rst` |
| **C++ PrimitiveC** | 算子 .h 头文件（`\brief`、类内方法描述） | — | — | — |
| **ops Primitive** | 接口实现 .py | `docs/api/api_python_en/mindspore.ops.primitive.rst` | `docs/api/api_python/ops/` 下**无 func_ 前缀**（如 `mindspore.ops.Add.rst`） | `docs/api/api_python/mindspore.ops.primitive.rst` |

### 11.5 mint 特例（参考资料 5.2.1 注）
- **ops.xxx_ext / nn.xxxExt**：若功能与原不带 ext 有差异需特别说明；若仅参数名或默认值不同，在参数说明中体现即可。
- **import 原接口到 mint**：若为直接 `from mindspore.ops.xxx` 或 `from mindspore.nn.xxx` 无 `as` 的复用，**中文 rst 可不写**，资料生成时会从 ops/nn 拷贝。
- **mint 样例代码**（除 mint.optim 外）：只写**原 ops/nn 接口**的样例，资料生成时会替换成 mint。写作格式必须为：`from mindspore import ops` + `ops.xxx()` / `ops.xxx_ext()` 等，或 `from mindspore import nn` + `nn.xxx()` / `nn.xxxExt()` 等；`import mindspore.ops as ops` 会替换为 `from mindspore import mint`。替换规则见参考资料 5.2.1 中的「原接口 → 替换成 mint 接口」表格。
- **mint 与 mint.nn.functional / mint.linalg 同时展示**：需在接口列表中分别添加 `mindspore.mint.xxx` 与 `mindspore.mint.nn.functional.xxx`（或 `mindspore.mint.linalg.xxx`）。
- **mint 分类**：所有分类参考 PyTorch（mint↔torch，mint.nn↔torch.nn，mint.nn.functional↔torch.nn.functional，mint.optim↔torch.optim）。
- **YAML 方式接口**：写作指导见参考资料中给出的 Gitee 链接（yaml 方式实现接口）。

### 11.6 常见问题摘要（参考资料 5.3，避免低级问题）

**内容**：接口描述建议含原理、公式、论文出处或配图；primitive 文档要求给出公式并解释公式参数。实验性接口需在描述中说明（中文：这是一个实验性API，后续可能修改或删除；英文：This is an experimental API that is subject to change or deletion）。英文 API 需写明支持平台和样例；中文不需写，生成时从英文提取。样例代码需**完整 import**、典型完整且可运行，必要时展示输出或 shape。

**格式**：文件名、文件内标题名、文件内接口定义三者**严格一致**（function 仅文件名多 func_ 前缀）；接口名下方 `=` 长度 ≥ 标题名。参数严格与代码对应，个数与名称完全对应；ops/nn 分为 Args、Inputs、Outputs；function/Tensor 分为 Args、Returns。Args 和 Raises 内容换行需缩进 4 个空格。一般变量和接口名用一个反引号 \` 包裹；参数取值用两个反引号 \`\` 包裹。中文 RST 中被 \` 或 \`\` 包裹的内容需与前后各留一个空格。描述中提及 MindSpore 接口时用内部跳转（:class:\`mindspore.ops.AvgPool\`、:func:\`mindspore.ops.avg_pool1d\`）。中文描述除代码相关外使用中文标点。描述具体 shape 时使用 :math:\`(x, x, x)\` 格式。Python 接口若 Class 下子方法不对外，可通过方法名前加 `_` 或删除方法注释实现。

## 12. 性能自验工具 apitimewrapper（来自 `算子流程/.../7. 接口性能自验工具.md`）

### 12.1 用途
对 MindSpore / PyTorch 脚本里的 API 做 wrap 打点，得到端到端耗时与（可选的）接口内部耗时分解。

### 12.2 使用要点（摘要）
- 安装：`pip install apitimewrapper-0.0.3-py3-none-any.whl`
- 整网打点：在网络入口启动 `start_hook_net(hook_inside=False)`，需要在网络执行前调用。
- 单 API：可同时启用 `start_hook_net` 与 `start_hook_torch_net`，用 `start_analysis()/end_analysis()` 包住循环。

## 13. 开源运作（RFC）流程要点（来自 `算子流程/.../11.算子开发开源运作规范.md`）

### 13.1 核心变化
需求分析/方案评审等传统会议流程简化为 RFC：在 Issue/RFC 里把信息链写全，通过评论完成讨论与共识。

### 13.2 RFC 内容建议（摘要）
- 背景与目标、交付范围、使用约束、遗留问题
- 方案设计（必要时外链归档）
- 验收与自测依据：UT/ST 覆盖、稳定性（多次运行无偶现）
- 代码 PR 与测试仓 PR 链接，@maintainers 检视

## 14. 反向实现注意事项（来自 `算子流程/.../算子反向注意事项.md`）

> 基础 BPROP 接线规则（I/O 个数、need_compute_grad_out、SetUnusedInputs）见 §7。

### 14.1 不可微分入参
Torch 里不可微分的入参（如 index/mode 等），MS 侧反向输出必须与输入个数一致：
- 对不可微分入参返回 `ib->OutZeros(x)`。
- 若全部入参都不可微分，可用 `ReturnZeros`（以框架现状为准）。

### 14.2 "梯度就是 0"时的实现选择
当某输入梯度理论上为 0，建议使用 `ib->ZerosLikeExt()`，确保走到 ACLNN/后端期望路径。

### 14.3 inplace 算子反向
- 若反向需要用到更新前的 self，需要注册 `CloneInplaceInput(...)` 让框架保留旧值。
- KBK 动态 shape 场景下在反向里使用 inplace 可能不保序时，可用 `ib->Depend(target, inplace_call)` 规避。

## 15. 安全编码与代码检视（来自 `算子流程/.../安全编码培训-算子代码检视.md`）

建议把"要检视的代码范围"当作改动面 checklist：
- Python 原语 + NN/functional/tensor + vmap
- C++ Infer
- bprop（Python/C++）
- 后端 kernel（CPU/GPU/AICPU/Ascend）及 Grad 单算子

## 16. Resize/Launch 优化要点（来自 `算子流程/.../ResizeKernelLaunch实现优化.md`）

> KBK 的基础结构与注册见 §6。

### 16.1 禁止在 InferShape 中修改属性
不要在 InferShape/InferType 内设置或修改算子属性（Pynative 下会引入问题）。

### 16.2 Resize/Launch 职责分离
- **能在 Init 确定的放 Init**；与 shape 强相关的放 Resize；Launch 尽量只做发射/调用。
- 运行期不要做 device 内存申请（例如 GPU 的 `cudaMalloc/cudaFree`），统一通过 workspace 由框架管理。

### 16.3 无意义输出忽略
对于预留/无意义输出，覆写 `GetUseLessOutputIdx()`（或等价接口）避免 dump/溢出误检/确定性副作用。

### 16.4 计算依赖（compute-depend）输出
按框架要求：分配最大可能输出 + 同步/更新输出 shape（如 NonZero 类模式）。

## 17. 精度零偏差与显存对齐自验

> 合并自原"精度零偏差自验指导"与"显存占用情况自验指导"。

### 17.1 精度零偏差（bitwise 一致）
当目标是与 PTA 输出二进制一致时：
- 固定随机种子，保存输出为 `.npy`（或 .npz）
- 用 `md5sum`（或等价方式）对比两个输出文件的哈希确保一致
- **Agent 须产出可执行脚本**：在 `tests/st/ops/ascend/` 下提供验证脚本（如 `verify_{op_name}_pta_md5sum.py`），脚本内实现固定种子、分别跑 MS 与 PTA、保存输出、md5 对比（或打印 md5 供用户比对），并在**脚本头或同目录 README** 中注明：运行环境（MS + torch_npu）、**运行命令**、**需要用户回传的内容**（如 md5 结果、通过/不通过）、**结果记录位置**（见下），供用户直接运行。
- **用户结果必须落到固定位置**：交付时明确告知用户将验证结果记录到 **Feature 文档 §14 验收报告 → 功能验证表** 中「是否与 PTA 计算结果 0 偏差」一行的「自测结果」「备注」列（可粘贴 md5 或「通过/不通过」及简要说明），以便 Agent 后续可直接读取该文件判断是否通过。
- 来源：`算子流程/.../精度零偏差自验指导.md`

### 17.2 显存占用对齐
关键点：MS 与 PTA 在**相同阶段**统计 max memory（避免把初始化/编译混入）。
- MS：`mindspore.runtime.max_memory_allocated()`
- PTA：`torch_npu.npu.max_memory_allocated()`
- 来源：`算子流程/.../显存占用情况自验指导.md`

## 18. 用 Cursor 辅助分析 PyTorch 算子（来自 `算子流程/.../基于 AI 工具Cursor进行pytorch算子分析.md`）

可用于"对标实现定位/反向路径查找"的提示词模板：
- `torch.<op> 算子的正反向是如何实现的？代码在哪里？`
配合查找：正向实现、autograd/derivatives 注册、NPU 插件映射等线索。

## 19. 接口开发要点（functional / nn / Tensor）（来自 `算子流程/.../2. 接口开发.md`）

### 19.1 functional 接口（强约束）
- functional 内部**务必使用** `_get_cache_prim` 获取 Primitive 实例，避免反复 __init__ 造成性能问题。
- 复杂接口允许"一对多映射/组合映射"：按参数分支选择不同 Primitive 或组合算子实现。

### 19.2 nn 接口
- nn 接口是 `Cell` 子类：在 `__init__` 初始化算子与属性，在 `construct` 做执行路径。
- `construct` 类似编译器入口：不要在其中直接 `raise`；需要编译期校验/抛错时，用 `@constexpr` 的辅助函数。

### 19.3 Tensor 方法（含 GE 映射要点）
- Tensor 方法需要覆盖不同模式：PyNative/KBK 与 GE（若项目要求）。
- GE 模式往往需要：
  - 在 `resource.cc` 注册映射；
  - 在 `standard_method.py` 实现（该处校验函数不能接收 Tensor 作为入参，需用对应封装）。

### 19.4 原语与接口接入策略（必须在 Pre-B 阶段确定）

在开始 YAML 定义之前，必须完成接口分析并确定原语/接口策略。

#### 19.4.1 接口分析五要素（来自 `aclnn开发示例.md` §1.2）

通过对比 MindSpore 与 PTA/torch 的文档和实现，搞清以下问题：
1. 功能是否一致
2. 参数定义是否一致
3. 支持的数据类型是否一致
4. **是否要新增原语**（Primitive）
5. **是新增接口还是复用原有接口**

> PTA/torch 可能存在**同名接口重载**（同函数名、不同参数签名），需逐一分析。

#### 19.4.2 YAML 三种场景（来自 `aclnn开发示例.md` §2.1）

| 场景 | YAML 操作 | 示例 |
| --- | --- | --- |
| **已有 YAML + 复用原有原语** | 在现有 YAML 上加 `dispatch` 字段 | `eye`：已有原语，加 `dispatch.Ascend: EyeAscend` |
| **已有 YAML + 新增原语** | 新建 YAML，加 `_ext` 后缀 | `zeros_like_ext`：已有 `zeros_like` 但参数不兼容 |
| **没有 YAML** | 新建 YAML，通常不加 `_ext` | 全新算子直接创建 |

**复用原有原语**示例：
```yaml
# 在已有 eye YAML 上加 dispatch 字段即可
dispatch:
  enable: True
  Ascend: EyeAscend
```

**新增原语 + `_ext`**示例：
```yaml
zeros_like_ext:
    args:
        input:
            dtype: tensor
        dtype:
            dtype: TypeId
            arg_handler: dtype_to_type_id
            default: None
    returns:
        y:
            dtype: tensor
    function:
        disable: True
    dispatch:
        enable: True
        Ascend: ZerosLikeExtAscend
```

#### 19.4.3 `ops.extend` 命名空间（来自 `Aclnn算子对接开发整体流程.md` §1.4.6）

> 若 aclnn 功能与存量 `ops.xx` 方法不一致，且不能对 ops 存量方法做兼容性修改 → 需要新增 extend 接口。

MindSpore 接口命名空间（来自 `5. 资料开发指导.md`）：
- `ops.xxx` / `ops.xxx_ext()` / `ops.auto_generate.xxx_ext()` / `ops.extend.xxx_ext()`
- `nn.xxx` / `nn.xxxExt()` / `nn.extend.xxx()`

#### 19.4.4 修改已有原语参数签名（文档缺口，需参考相似代码）

实际开发中，经常遇到已有 Primitive 需要**扩展参数**的情况（如 PTA 新版多了参数、需要支持 ACLNN 特有参数等）。原始文档提到了相关参考（"前端api接口重载开发指导"、"Tensor重载指导"），但详细内容未收录在当前知识库中。

**遇到此场景时的实践策略**：
1. **搜索 MS 仓库中相似算子**的处理方式作为参考（如同类算子是如何加参数的）
2. **具体分析兼容性**：新参数是否能设默认值、是否影响已有调用方、是否影响其他后端
3. **判断走哪条路**：
   - 可兼容修改（加可选参数不破坏已有行为）→ 直接修改现有 YAML + Infer + 接口
   - 不可兼容 → 新增原语加 `_ext` 后缀，或走 `ops.extend`
4. **确保不破坏已有功能**：修改后已有 UT/ST 全部回归通过
5. **遵循评审规则**（见 §19.4.5）

#### 19.4.5 评审规则（来自 `Aclnn算子对接开发整体流程.md` §2.2）

| 变更类型 | 评审要求 |
| --- | --- |
| 无新增接口，功能与之前完全一致 | 无需评审 |
| 无新增接口，功能有扩展 | 需要评审 |
| 新增接口 | **重点评审** |
| 存量接口功能非兼容修改 | **原则上不允许**，特殊情况评审 |
| 新增算子 | 需要评审 |
| 存量算子功能非兼容修改 | 需要评审 |

> 无论接口/算子对外与否，都需要按规则评审。评审后需发送**接口变更邮件**通知 MindSpore 各组件。

## 20. 问题处理与"书面结论"（来自 `Aclnn算子对接开发整体流程.md`）

当出现"无法修复的问题"，建议按来源分类并固化证据：

### 20.1 CANN 算子问题
- 需要拿到**正式书面记录**：规格说明书、邮件/会议纪要结论、DTS 单等。
- 聊天记录等非正式内容不作为正式结论。

### 20.2 MindSpore 框架问题 / 方案限制
- 与框架相关负责人确认后，如结论是"允许带问题转测"，需要形成会议纪要并抄送相关人员。
- 会议纪要建议包含：议题、时间、人员、背景、结论。

## 21. 质量门禁与格式要求（与本 skill 核心行为准则一致，见 SKILL 核心行为准则）

建议在 checklist 中显式跟踪这些点：
- 行长不超过 120；避免行尾空格；尽量统一空格缩进。
- UTF-8 编码、制表符检查等基础规范。
- 代码质量检查工具（项目列出的 Check_*）在本地/CI 中应通过。

## 22. 当 ACLNN / PTA 文档不完善：用"探测脚本"补齐事实范围

现实中 ACLNN/PTA 文档可能滞后或缺失细节（尤其是不同 CANN/PTA 版本支持范围变化时）。此时不要猜：

### 22.1 必须先记录版本矩阵
让用户确认并记录（写入 RFC/验收报告/测试输出）：
- torch 版本、torch_npu 版本
- CANN 版本（或可追溯的安装路径/镜像版本信息）
- 芯片型号/驱动信息（能打印则打印）

### 22.2 生成并运行"PTA 支持范围探测脚本"

**Agent 必须提供可执行脚本，不能只让用户"自己去跑"：**

- 本 skill 提供**模板**：`scripts/probe_pta_sparse_flash_attention.py`（以 `sparse_flash_attention` 为例）。
- **若目标算子不是 sparse_flash_attention**：Agent 须基于该模板生成**当前算子的探测脚本**，产出到 skill 仓库或用户仓库的可执行路径，例如：
  - `scripts/probe_pta_{op_name}.py`（在 skill 的 `scripts/` 下），或
  - 用户仓库内 `tests/st/ops/ascend/probe_pta_{op_name}.py`
- 脚本内需修改：`run_case()` 的输入构造与 PTA API 调用、`main()` 中的测试矩阵（dtype/layout/关键参数组合）。
- 用途：自动枚举一组 dtype/layout/关键参数组合，记录成功/失败与错误信息，并输出 JSON 汇总。

运行方式（示例，脚本路径以实际产出为准）：

```bash
python scripts/probe_pta_sparse_flash_attention.py --device npu:0 --out pta_probe.json
# 快速模式（只跑核心组合）：
python scripts/probe_pta_sparse_flash_attention.py --device npu:0 --quick --out pta_probe_quick.json
```

**用户结果必须落到固定位置，方便 Agent 后续读取：**

- 要求用户将运行结果**记录到**以下之一（交付时明确告知）：
  - **Feature 文档**：在「§8 与 PTA 的差异与对齐」或单独小节「PTA 探测结果」中粘贴：版本矩阵 + `pta_probe.json` 的 summary 与关键失败用例错误信息；若文件较大，可写「见附件 `pta_probe_{op_name}.json`」并将该 JSON 放在与 Feature 文档同目录或 `docs/` 下。
  - 或**固定文件**：`docs/pta_probe_results/{op_name}.json`（或项目约定的路径），并在 Feature 文档 §8 注明「PTA 探测结果见 `docs/pta_probe_results/{op_name}.json`」。
- 这样 Agent 在后续步骤中可直接读取该文件/段落，无需依赖用户再次粘贴到对话。

你需要用户回传的证据（且须落在上述记录位置）：
- `pta_probe.json`（或其中 summary + 关键失败用例的错误信息）
- 同一份输出里的版本信息（torch/torch_npu/env/npu-smi）

### 22.3 用探测结果驱动"接口对齐与约束落地"
基于探测结果再决定：
- sparse_size 是否被固定（例如某些 CANN 版本要求 2048）
- attention_mode / return_softmax_lse / layout 组合是否可用
- dtype 支持是否确实只有 fp16/bf16，是否存在隐藏限制
并把这些结论同步到 YAML/Infer/文档/测试中。

## 23. vmap 支持（按需）

> 来源：`算子流程/.../4. 算子关键特性.md`。当前 skill 主要覆盖 ACLNN 算子的前向/反向/
> 推导/测试/文档流程，vmap 作为**可选扩展**列在此处。如果目标算子不需要 vmap 支持，可跳过本节。

### 23.1 何时需要 vmap
- 算子需要被 `vmap`/`vectorize_cell` 调用时。
- 项目要求覆盖 vmap 路径时（部分算子交付件要求 vmap 验证）。

### 23.2 关键要点（摘要）
- 注册 vmap rule：在框架指定位置注册（以项目现有 vmap 注册模式为准）。
- 测试：需单独补 vmap UT，验证批量化后的 shape/数值正确性。
- 注意 vmap 路径可能不走 ACLNN（而是退回到组合算子/循环展开），需确认性能是否可接受。

## 24. 代码骨架模板（可直接复制改造）

> 以下骨架来自"先自动生成再自定义改造"的实际经验，仅作**起步参考**。
> 真正使用时必须对照同目录下相似算子的现有代码调整宏名、命名空间、参数列表。

### 24.1 YAML 最小模板（op_def + api_def + function_doc）

```yaml
# ---- op_def ----
op_name: "OpNameCustomize"
args:
  - {name: "input_x", type: "Tensor", desc: "..."}
  - {name: "scale", type: "float", default: 1.0, desc: "..."}
outputs:
  - {name: "output", type: "Tensor", desc: "..."}
dispatch:
  enable: True
  Ascend: "OpNameAscendCustomize"
# ---- api_def ----
api:
  py_method: "op_name"
  module: "mindspore.ops"
# ---- function_doc ----
function_doc:
  desc: "Brief English description of the operator."
  args:
    input_x: "Description of input_x."
    scale: "Description of scale. Default: ``1.0``."
  returns: "Description of output."
  examples: |
    >>> import mindspore as ms
    >>> from mindspore import ops
    >>> x = ms.Tensor([1.0, 2.0, 3.0])
    >>> out = ops.op_name(x, scale=1.0)
```

### 24.2 GeneralInfer 骨架（C++）

```cpp
// op_name_general_infer.cc
#include "ops/ops_func_impl/op_name.h"
// 具体 include 路径以仓库实际为准

namespace mindspore::ops {

BaseShapePtr OpNameFuncImpl::InferShape(
    const PrimitivePtr &prim,
    const std::vector<AbstractBasePtr> &input_args) const {
  // 1. 取输入 shape
  auto x_shape = input_args[0]->GetShape()->GetShapeVector();

  // 2. 动态 rank 回退
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  // 3. 取关键标量参数（可能 unknown）
  // auto scale_opt = GetScalarValueWithCheck<float>(input_args[1]->GetValue());
  // if (!scale_opt.has_value()) {
  //   // 关键参数 unknown -> 对应维度回退动态维
  //   out_shape[dim_idx] = abstract::TensorShape::kShapeDimAny;
  // }

  // 4. 精确推导
  ShapeVector out_shape = x_shape;  // 按算子语义计算
  return std::make_shared<abstract::TensorShape>(out_shape);
}

TypePtr OpNameFuncImpl::InferType(
    const PrimitivePtr &prim,
    const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[0]->GetType();
  // 通常输出 dtype 与输入一致或按算子语义确定
  return x_type->Clone();
}

}  // namespace mindspore::ops
```

### 24.3 PyBoost customize 骨架（C++）

```cpp
// op_name_ascend_customize.cc
#include "plugin/device/ascend/kernel/pyboost/customize/op_name_ascend_customize.h"
// 具体 include 以仓库实际为准

namespace mindspore::kernel::pyboost {

// 前向
tensor::TensorPtr OpNameAscendCustomize::Call(
    const tensor::TensorPtr &input_x,
    const std::optional<float> &scale) {
  // 1. 输出 tensor 分配
  auto output = std::make_shared<tensor::Tensor>(input_x->data_type(), out_shape);

  // 2. 参数转换（tuple->vector / None 处理等）
  // auto scale_val = scale.value_or(1.0f);

  // 3. ACLNN 两段式调用（以项目封装宏为准）
  // LAUNCH_ACLNN(aclnnOpName, stream, input_x, scale_val, output);

  return output;
}

}  // namespace mindspore::kernel::pyboost
```

### 24.4 KBK kernel 骨架（C++）

```cpp
// op_name_aclnn_kernel.cc
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel/op_name_aclnn_kernel.h"
// 具体 include 以仓库实际为准

namespace mindspore::kernel {

void OpNameAclnnKernel::GetWorkSpaceInfo(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &outputs) {
  // 取参（标量/tuple 等）
  // auto scale = inputs[1]->GetValueWithCheck<float>();

  // 获取 workspace
  // GetWorkspaceForResize(aclnnOpNameGetWorkspaceSize, ...);
}

bool OpNameAclnnKernel::Launch(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &workspace,
    const std::vector<KernelTensor *> &outputs,
    void *stream_ptr) {
  // RunOp(aclnnOpName, stream, ...);
  return true;
}

// 注册
MS_ACLNN_KERNEL_FACTORY_REG(OpName, OpNameAclnnKernel);

}  // namespace mindspore::kernel
```

### 24.5 BPROP builder 骨架（C++）

```cpp
// grad_xxx_ops.cc (在合适的 bprop 注册文件中添加)

REG_BPROP_BUILDER("OpName").SetBody([](const BpropBuilder *ib) -> NodePtrList {
  // 取正向输入
  auto input_x = ib->GetInput(kIndex0);
  auto scale = ib->GetInput(kIndex1);
  // 取正向输出与上游梯度
  auto out = ib->GetInput(kIndex2);   // 正向输入个数 + 0
  auto dout = ib->GetInput(kIndex3);  // 正向输入个数 + 1

  // 构建反向子图
  NodePtr dx;
  if (ib->need_compute_grad_out(kIndex0)) {
    dx = ib->Emit("OpNameGrad", {input_x, out, dout, scale});
  } else {
    dx = ib->OutZeros(input_x);
  }

  // 非张量参数（如 scale）返回零梯度占位
  auto d_scale = ib->OutZeros(scale);

  return {dx, d_scale};
});
```

## 25. PTA 源码审查方法（必做）

> PTA 文档可能滞后或遗漏细节。开发前**必须**同时审查 op-plugin 仓库中的实际代码，
> 结合文档与源码一起参考。若两者不一致，不要猜，让用户找接口人确认（见 §25.5）。

### 25.1 需要审查的三类关键文件

| 文件 | 路径模式 | 提取什么信息 |
| --- | --- | --- |
| **函数签名 YAML** | `op_plugin/config/op_plugin_functions.yaml` | 精确的参数名、类型、默认值、返回值结构、是否走 `op_api` / `acl_op` / `gen_opapi` |
| **反向注册 YAML** | `op_plugin/config/derivatives.yaml` | 哪些输入可微、grad 函数名、参数传递顺序、`output_differentiability` |
| **C++ 实现** | `op_plugin/ops/opapi/XxxKernelNpuOpApi.cpp`（含 Grad 变体） | 实际调用的 `aclnnXxx`、参数预处理逻辑、输出 tensor 构造方式、硬编码默认值 |

### 25.2 审查时重点关注的差异点

从 PTA 代码中经常能发现文档未提及的关键细节：

**1. 前向与反向的参数命名/顺序不一致**
- 例：前向用 `actual_seq_lengths_query`，反向用 `actual_seq_qlen`
- 例：前向 `layout_query`/`layout_kv` 分开，反向退化为单个 `layout`
- **影响**：MS 侧 bprop builder 的参数传递必须按反向的实际签名，不能照搬前向

**2. 反向 ACLNN 调用的额外/隐藏参数**
- 例：反向里硬编码 `deterministic_const = true`（前向无此参数）
- 例：反向缺少 `block_table`（前向有但反向不传）
- **影响**：MS 侧 KBK/PyBoost 的反向实现要对齐这些隐藏行为

**3. 可选参数的 None 处理方式**
- 例：`query_rope` 为 None 时，PTA 构造 `at::Tensor()`（空 tensor）传给 ACLNN
- 例：`query_rope` 的梯度在 None 时输出 `at::empty({0}, ...)`（形状为 [0]，非零张量）
- **影响**：MS 侧必须同步 None 语义，否则 ACLNN 可能报错或结果不一致

**4. 输出 tensor 个数与构造**
- 例：前向返回 `(output, softmax_max, softmax_sum)` 共 3 个，反向返回 5 个
- `softmax_max`/`softmax_sum` 的 shape 推导逻辑在 C++ 里有明确的 3D/4D 分支
- **影响**：MS Infer 必须对齐输出 shape 推导逻辑，bprop 必须正确传递中间结果

**5. `derivatives.yaml` 中的梯度传递**
- 例：`result0`/`result1`/`result2` 分别对应前向的第 0/1/2 个输出
- 哪些输入标记为 `non_differentiable`，哪些参与 grad
- **影响**：MS bprop 的 `GetInput` 索引和 `OutZeros` 占位必须对齐

### 25.3 审查操作步骤

1. **在 `op_plugin_functions.yaml` 中搜索算子名**：提取前向/反向的精确签名，记录参数差异。
2. **在 `derivatives.yaml` 中搜索算子名**：提取反向注册，确认可微输入和 grad 参数传递。
3. **找到对应的 C++ 实现文件**（`ops/opapi/` 下），阅读：
   - 输出 tensor 的 shape 构造逻辑
   - 可选参数的 None 处理（`value_or` 的默认值是什么）
   - 实际传给 `EXEC_NPU_NO_FORMAT_CHECK_CMD` / `aclnnXxx` 的参数列表与顺序
   - 是否有硬编码参数（如 `deterministic`）
4. **记录发现的差异**，作为 MS 适配的依据写入验证闭环的"关键证据"部分。
5. **若发现代码与文档不一致 → 必须暂停并确认**（见 25.5）。

### 25.4 典型差异记录模板

```text
算子：npu_sparse_flash_attention

前向 vs 反向参数差异：
- actual_seq_lengths_query (fwd) → actual_seq_qlen (bwd)
- layout_query + layout_kv (fwd) → layout 单个 (bwd)
- block_table (fwd 有) → bwd 不传
- return_softmax_lse (fwd 有) → bwd 不传

反向隐藏行为：
- deterministic_const = true（硬编码）
- query_rope 为 None 时 d_query_rope = at::empty({0}, ...)

输出结构：
- 前向：(output, softmax_max, softmax_sum) = 3 个
- 反向：(d_query, d_key, d_value, d_query_rope, d_key_rope) = 5 个

derivatives.yaml 可微输入：
- query, key, value, query_rope, key_rope（5 个）
- sparse_indices, block_table 等不可微
```

### 25.5 代码与文档不一致时的处理流程

> **核心原则**：文档与源码一致时，结合两者参考，高效推进。不一致时不要猜，直接让用户去确认。

**一致时**：结合文档（了解语义/约束）和源码（了解实现细节/隐藏行为）同步参考，直接推进开发。

**不一致时**：
1. **整理差异清单**：逐条列出"文档说的是 X，代码实际是 Y"，给出文件路径和行号。
2. **立即交给用户确认**：不自行判断以哪边为准，让用户找 ACLNN/PTA 算子开发接口人确认。
3. **拿到结论后继续**：将确认结论记录到方案文档/RFC 中，据此推进 MS 适配。

差异确认输出模板：

```text
⚠️ PTA 代码与文档不一致，需要确认

差异清单：
| # | 内容 | 文档描述 | 代码实际行为 | 文件/行号 |
| - | ---- | -------- | ------------ | --------- |
| 1 | ... | ... | ... | ... |

建议找以下接口人确认以哪边为准：
- ACLNN 算子开发接口人
- PTA 算子开发接口人

请确认后告知结论，我再继续开发。
```

## 26. InferValue 常量折叠（可选优化）

> 来源：`算子流程/.../3. 算子开发.md`。当算子的输入在编译期全部已知时，可通过 InferValue 直接
> 推导出结果值，跳过运行时计算，提升整图执行性能。

### 26.1 两种实现方式
- **Python 回调**（如 concat）：在 `mindspore/python/mindspore/ops/operations/manually_defined/ops_def.py`
  中注册 InferValue 回调函数。
- **C++ 实现**（如 add）：在 `mindspore/ops/infer/ops_frontend_func_impl/` 下实现。
- **C++ 性能优于 Python**，优先使用 C++ 实现。

### 26.2 验证方法
- 增加 InferValue 的 UT 用例（全常量输入场景）。
- 运行测试脚本查看 IR 图，确认常量折叠生效（输出节点变为 ValueNode）。

### 26.3 适用场景
- 算子输入在编译期可确定（如 shape 计算、类型转换等辅助算子）。
- 大多数 ACLNN 计算算子的输入在运行时才确定，**不需要实现 InferValue**。

## 27. 动态 shape 分类与处理策略

> 来源：`算子流程/.../4. 算子关键特性.md`。Infer 推导的快速回退策略另见 §4.4。

### 27.1 动态 shape 三种类型
| 类型 | 含义 | 典型算子 | Infer 策略 |
| --- | --- | --- | --- |
| **InputDynamic** | 输入 shape 编译期未知 | 大多数算子 | 输出对应维度设为 -1（`kShapeDimAny`） |
| **OutputDynamic (Input Value Depend)** | 输出 shape 依赖输入的值 | `DynamicBroadcastTo` | 用 `GetShapeValue()` 取输入值作为输出 shape |
| **OutputDynamic (Compute Depend)** | 输出 shape 需运行时计算 | `NonZero`、`UniqueConsecutive` | 输出分配最大可能 size + 运行后 `SyncOutputShape` |

### 27.2 InputDynamic 处理要点
- 输入 shape 中 -1 维度：输出对应维度也设为 -1。
- 输入动态秩（-2）：输出回退动态秩。
- 关键标量参数 unknown（`HasUnknownValue`）：依赖该参数的输出维度回退 -1。

### 27.3 Input Value Depend 处理要点
- 使用 `GetShapeValue()` 接口提取输入 tensor 中的值作为输出 shape。
- 若输入值 unknown（`kValueAny`），回退动态维。
- 典型场景：`Reshape`（新 shape 作为输入传入）。

### 27.4 Compute Depend 处理要点
- 输出分配最大可能 size（编译期估算上界）。
- 运行后通过 `Sync` + `SyncOutputShape` 更新实际输出 shape。
- 需覆写 `GetUseLessOutputIdx()` 避免 dump/溢出误检。

## 28. ACLNN 调用链分析与子算子盘点（组合场景）

> PTA 的一个 `torch_npu.npu_xxx()` 接口，底层不一定只调用一个 `aclnnXxx` 大算子，
> 常见模式是**多个 ACLNN 小算子串联**（前向/反向均可能）。在这种场景下，MS 必须先盘点
> 所有子算子的接入情况，补齐缺失的子算子，再按相同方式组合。

### 28.1 何时需要做调用链分析

- PTA C++ 实现中出现了**多个 `EXEC_NPU_CMD` / `EXEC_NPU_NO_FORMAT_CHECK_CMD`** 调用。
- PTA C++ 实现中调用了其他 `at_npu::native::` 函数（间接组合）。
- ACLNN 文档/头文件中没有与 PTA 接口一一对应的单个大算子。
- 反向实现不是单个 `aclnnXxxGrad`，而是用多个小算子拼出梯度计算。

### 28.2 调用链提取方法

1. **找到 PTA 前向 C++ 实现**（`ops/opapi/XxxKernelNpuOpApi.cpp`），逐行标注：
   - 每个 `EXEC_NPU_CMD(aclnnYyy, ...)` 或 `OpApiFunc(aclnnYyy, ...)`
   - 中间 tensor 的构造（`at::empty(...)` / `npu_preparation::apply_tensor(...)`）
   - 参数预处理（类型转换、默认值填充、None 处理）
2. **同样分析反向 C++ 实现**（`XxxGradKernelNpuOpApi.cpp` 或 `derivatives.yaml` 指向的函数）。
3. **产出调用链图**（文本即可）：

```text
torch_npu.npu_foo(q, k, v, scale) 前向调用链：
  ① aclnnBarPrepare(q, k) → intermediate_qk     # 预处理
  ② aclnnAttentionScore(intermediate_qk, v, scale) → output  # 主计算
  ③ aclnnSoftmaxLse(output) → softmax_lse        # 辅助输出

torch_npu.npu_foo 反向调用链：
  ① aclnnAttentionScoreGrad(dout, q, k, v, softmax_lse) → (dq, dk, dv)
  （反向为单个大算子，无需拆分）
```

### 28.3 MS 侧覆盖盘点方法

对调用链中的每个 `aclnnYyy`，在 MS 仓库中搜索确认：

| 搜索对象 | 搜索关键词 | 说明 |
| --- | --- | --- |
| YAML 定义 | `aclnnYyy` 或对应 op_name | 确认 op_def 是否存在 |
| PyBoost 实现 | `LAUNCH_ACLNN(aclnnYyy` 或 `aclnnYyyGetWorkspaceSize` | 确认 Pynative 路径 |
| KBK kernel | `MS_ACLNN_KERNEL_FACTORY_REG` + 对应类名 | 确认 Graph 路径 |
| Infer | 对应 `FuncImpl` 类 | 确认推导是否存在 |
| aclnn_config.yaml | 算子名映射 | 确认调度映射 |

### 28.4 盘点结果模板

```text
目标接口：torch_npu.npu_foo → mindspore.ops.foo

ACLNN 调用链盘点：
| # | aclnnXxx | 用途 | MS 状态 | 备注 |
| - | -------- | ---- | ------- | ---- |
| 1 | aclnnBarPrepare | 前向-预处理 | ✅ 已接入 | YAML/Infer/PyBoost/KBK 齐全 |
| 2 | aclnnAttentionScore | 前向-主计算 | ⚠️ 仅有 YAML+Infer | 缺 PyBoost customize 和 KBK |
| 3 | aclnnSoftmaxLse | 前向-辅助输出 | ❌ 未接入 | 需走完整开发流程 |
| 4 | aclnnAttentionScoreGrad | 反向 | ✅ 已接入 | 无需额外工作 |

实施计划：
1. 先补 #3（aclnnSoftmaxLse）：走 YAML→Infer→PyBoost→KBK→UT 全流程
2. 再补 #2 的 PyBoost/KBK
3. 最后在 foo 的 customize 中组合 #1+#2+#3
```

### 28.5 实施顺序原则

- **叶子先、组合后**：先实现所有独立的子算子，再实现组合算子。
- **前向先、反向后**：反向可能复用前向子算子，先确保前向链完整。
- **每个子算子走完整流程**：缺失的子算子按 SKILL.md 的步骤 1-8 逐步实现
  （但通常不需要独立导出/文档，只需 YAML+Infer+PyBoost+KBK+UT）。
- **组合算子在最后实现**：确认所有子算子可用后，再写组合层的 customize。

## 29. 组合实现模式（PyBoost/KBK 多 ACLNN 串联）

> 当目标算子需要串联多个 ACLNN 调用时，PyBoost 和 KBK 的写法与单算子直连有显著差异。

### 29.1 PyBoost 组合模式

```cpp
// foo_ascend_customize.cc — 组合多个 ACLNN 调用
tensor::TensorPtr FooAscendCustomize::Call(
    const tensor::TensorPtr &query,
    const tensor::TensorPtr &key,
    const tensor::TensorPtr &value,
    const std::optional<float> &scale) {

  // ---- 阶段 1：预处理子算子 ----
  auto intermediate_qk = std::make_shared<tensor::Tensor>(
      query->data_type(), infer_qk_shape(query, key));
  // LAUNCH_ACLNN(aclnnBarPrepare, stream, query, key, intermediate_qk);

  // ---- 阶段 2：主计算子算子 ----
  auto output = std::make_shared<tensor::Tensor>(
      query->data_type(), infer_output_shape(intermediate_qk, value));
  auto scale_val = scale.value_or(1.0f);
  // LAUNCH_ACLNN(aclnnAttentionScore, stream, intermediate_qk,
  //              value, scale_val, output);

  // ---- 阶段 3：辅助输出子算子（可选）----
  // auto softmax_lse = ...;
  // LAUNCH_ACLNN(aclnnSoftmaxLse, stream, output, softmax_lse);

  return output;  // 或 MakeTuple(output, softmax_lse)
}
```

**关键注意事项**：
- 中间 tensor 必须**手动分配**（shape 需自行推导或从 Infer 获取）。
- 每个 ACLNN 调用都是两段式（workspace + launch），stream 在同一个上下文中顺序执行。
- 中间 tensor 的生命周期仅限本函数，不会暴露给框架（除非作为输出返回）。

### 29.2 KBK 组合模式

```cpp
// foo_aclnn_kernel.cc — 组合多个 ACLNN 调用
void FooAclnnKernel::GetWorkSpaceInfo(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &outputs) {
  // 每个子算子分别计算 workspace，累加到总 workspace
  // GetWorkspaceForResize(aclnnBarPrepareGetWorkspaceSize, ...);
  // GetWorkspaceForResize(aclnnAttentionScoreGetWorkspaceSize, ...);
}

bool FooAclnnKernel::Launch(
    const std::vector<KernelTensor *> &inputs,
    const std::vector<KernelTensor *> &workspace,
    const std::vector<KernelTensor *> &outputs,
    void *stream_ptr) {
  // 按顺序调用多个 RunOp
  // RunOp(aclnnBarPrepare, stream, ...);
  // RunOp(aclnnAttentionScore, stream, ...);
  return true;
}
```

**关键注意事项**：
- **workspace 管理**：多个子算子的 workspace 需要分别获取并累加，或按最大值取。
  具体策略取决于框架的 workspace 管理接口（以仓库现有组合算子实现为准）。
- **中间 tensor**：KBK 中中间 tensor 可能需要通过 workspace 分配，
  或在 `GetWorkSpaceInfo` 中额外申请。参考仓库中已有的组合算子实现。
- **错误处理**：任一子算子调用失败应立即返回 false，不继续后续调用。

### 29.3 组合场景的 Infer 要点

- 组合算子的 Infer **只需推导最终输出的 shape/type**，不需要推导中间 tensor。
- 中间 tensor 的 shape 推导逻辑放在 PyBoost/KBK 的实现代码中。
- 如果最终输出依赖于中间结果的 shape（级联推导），Infer 中需要**重复这段推导逻辑**
  或者直接按已知输入推导最终输出（跳过中间步骤）。

### 29.4 组合场景的分层验证策略

| 阶段 | 验证内容 | 方法 |
| --- | --- | --- |
| **子算子级** | 每个子算子独立正确 | 子算子各自的 UT/ST |
| **组合级-中间值** | 中间 tensor 与 PTA 对齐 | 在 customize 中临时保存中间 tensor，与 PTA 逐阶段对比 |
| **组合级-最终输出** | 最终输出与 PTA 对齐 | 标准 ST 对齐流程（shape/type/数值） |
| **反向级** | 梯度正确性 | 反向 ST + 数值梯度检查（若适用） |

> **调试技巧**：组合算子出错时，先定位是哪个子算子调用出了问题。
> 在 PyBoost 中可以临时在每个子算子调用后 dump 中间 tensor 进行排查。

---

## §30 Feature 文档（评审与交付必须产物）

> **来源**：实际交付的 Feature 文档（`==符号重载Feature.md`、`CosineEmbeddingLoss Feature.md`、`NsaCompressAttention_Feature_文档.md`、`参考feature.md`）。

### 30.1 什么是 Feature 文档

Feature 文档是算子**评审和转测交付的必须文件**，它将方案设计、接口定义、实现细节、测试计划和验收结果整合到一份标准化文档中。
评审委员会根据此文档判断算子是否可以合入主干。

### 30.2 Feature 文档标准章节

| 序号 | 章节 | 填写时机 | 说明 |
| ---- | ---- | -------- | ---- |
| 1 | 背景描述 | Pre-B | 算子来源、动机、MindSpore 为何需要 |
| 2 | 标杆与接口 | Pre-B | 标杆接口（PTA/Torch）、MindSpore 接口（functional/nn/tensor） |
| 3 | 任务清单 | Pre-B 初始化 → 开发中更新状态 | **标准 13 大类表格**（见下方 §30.3） |
| 4 | 功能与接口说明 | Pre-B | 计算公式、接口签名、参数说明 |
| 5 | YAML 定义 | Step 1 后 | `op_def` YAML 内容 |
| 6 | 约束与类型 | Pre-B | 设备、dtype、shape 约束、空 Tensor 策略 |
| 7 | 执行模式与适配 | Step 4/5 后 | PyBoost / KBK 实现说明 |
| 8 | 与 PTA 的差异与对齐 | Pre-B 初始化 → 开发中补齐 | 功能/精度/API 语义差异 |
| 9 | 动态 Shape/Rank 支持 | Step 3 后 | 动态维/动态秩推导策略 |
| 10 | 异常与校验 | Step 3/4 后 | 推导期/运行期校验 |
| 11 | 反向（BPROP） | Step 6 后 | BPROP 注册、反向接口、梯度处理 |
| 12 | 测试方案 | Step 8 后 | UT/ST/TEST_OP 覆盖说明 |
| 13 | 代码与文件改动说明 | 开发完成后 | 所有新增/修改文件的完整路径 |
| 14 | 验收报告 | 转测前 | 四张自测表：资料验证 + 功能验证 + 性能验证 + 安全编码（见 §30.4） |

### 30.3 任务清单标准 13 大类

Feature 文档中的"任务清单"是标准化表格，每个算子**必须逐项填写**：

| 序号 | 任务项 | 子项 |
| ---- | ------ | ---- |
| 1 | 接口基本功能 | Primitive / functional / nn / tensor |
| 2 | 后端及数据类型支持 | Ascend / GPU / CPU |
| 3 | 支持 vmap | — |
| 4 | 支持动态 Shape | 动态 Shape / 动态 Rank |
| 5 | 支持反向 | bprop 函数 / 复数支持 |
| 6 | 补齐资料 | API 映射 / 接口中英文资料 |
| 7 | 性能优化 | CPU / GPU / Ascend |
| 8 | 功能 | 空 Tensor / inf-nan / 0~8 维 / 其他功能点 |
| 9 | 门禁用例补齐 | UT / ST / TEST_OP |
| 10 | 支持 MS Adapter | — |
| 11 | 自动并行切分 | — |
| 12 | 混合精度（AMP） | — |
| 13 | 安全与异常 | 异常用例与报错规范 |

每项需标注：`新增` / `修改` / `无变更` / `不涉及`，并在备注中简要说明。

### 30.4 验收报告四张表

#### 资料验证表（17 项）

涵盖：接口列表、UT/ST 用例、中英文文档、接口描述、公式、参数描述、输入描述、输出描述、
输出尺寸与输入关系、Raises、平台填写、格式检查、样例提供、样例打印结果、样例可执行、API 沙盘。

#### 功能验证表（27 项）

涵盖：默认参数、空 Tensor、inf/nan、dtype 对齐、取值范围、维度覆盖 0D-8D、dtype 全覆盖、
隐式类型转换、广播、输入约束、正向精度、反向支持、反向单算子实现、异常报错信息、报错白名单、
functional 用例、动态 shape/rank、退避关闭验证、测试仓回归、bf16、bprop 按需求导、
输出 shape 计算依赖、非连续输入、PTA 0 偏差、存量接口影响、AMP、多 Tensor dtype 不一致。

#### 性能验证表（4 项）

涵盖：广播场景性能、反向显存优化（SetUnusedInputs）、多规格性能（≥3 种）、显存持平 PTA。

#### 安全编码检视表（12 项）

涵盖：指针判空、先用后校、越界、除零、内存泄露、异常路径释放、nothrow、安全函数库、
类型转换溢出、冗余代码、敏感信息、弱随机数。

### 30.5 Feature 文档生成流程

```
Pre-B 阶段：
  1. 从模板 templates/feature-document.md 复制一份
  2. 填写 §1-§4（背景/标杆/任务清单/接口说明）和 §6（约束）和 §8（PTA 差异初始化）
  3. 提交给评审委员会做方案评审

开发过程中：
  4. 每完成一个 Workflow Step，回填对应章节
     - Step 1 → §5（YAML）
     - Step 3 → §9（动态Shape）, §10（异常）
     - Step 4/5 → §7（执行模式）
     - Step 6 → §11（反向）
     - Step 8 → §12（测试方案）

转测交付前：
  5. 补齐 §13（代码改动）
  6. 填写 §14 验收报告的四张自测表（资料/功能/性能/安全编码）
  7. 更新 §3 任务清单中每项的最终状态
  8. 完整 Feature 文档随代码 PR 一起提交
```

### 30.6 不同类型算子的 Feature 文档差异

| 场景 | 差异 |
| ---- | ---- |
| **单 ACLNN 算子** | 标准流程，§7 中 PyBoost/KBK 各调用一个 ACLNN 接口 |
| **组合算子（小算子拼接）** | §4 需描述 ACLNN 调用链，§7 描述多 ACLNN 组合，§12 需分层验证 |
| **符号重载（如 ==）** | §4 需描述 MultitypeFuncGraph 适配，§3 中 functional/tensor 列为"修改" |
| **纯 Python 组合（无 Primitive）** | §3 中 Primitive 列为"不涉及"，§7 只描述 functional 层实现 |

### 30.7 模板位置

- 模板文件：`templates/feature-document.md`
- 参考实例：用户提供的已有 Feature 文档（建议在开发前找到相似算子的 Feature 作参考）

## 31. Skill 维护策略

本节约定 skill 各文件的体量分工、格式约束、反馈与更新、维护者溯源，避免说明散落多处。

### 31.1 文件体量与分工

- **SKILL.md**：严控在 **500 行**内，只放流程总览、核心行为准则、执行清单、排障路径。细节、案例、背景知识解耦到本文件（reference.md）或 `examples.md`，需要时再按章节读取。
- **reference.md**：可按图索骥的细节与模板，按需查阅。内容来源于原始文档，已由 `traceability.md` 做溯源映射。**本文件多为摘要**，细节以源文档为准；凡 workflow 对应单一明确源文档时，维护或全面检查时应用源文档逐节核对 workflow/checklists，避免仅依赖本文件摘要导致遗漏（见 `traceability.md`「Workflow 与源文档对齐」）。
- **checklists.md**：可复制的自检清单，只放"是/否"可判定的检查项，不放解释性文字。

### 31.2 格式与编码约束

- **行长** ≤ 120 字符（代码与文档均适用）。
- **路径风格**：文档中统一使用 Unix 风格路径（`a/b/c`），禁止 Windows 反斜杠。
- **编码**：UTF-8，无 BOM；英文 YAML 文档禁止夹杂中文字符（Windows GBK 环境下可能导致生成/编译问题）。
- **Skill 安装位置**：项目内 `.cursor/skills/`（或 `.claude/skills/`）；禁止放到 `~/.cursor/skills-cursor/`（该目录为内置技能，不受版本控制）。

### 31.3 反馈与更新

使用中遇到 skill 指引与实际不符时，直接告诉 AI 具体情况（哪个算子、卡在哪步、实际做法）；AI 会按 SKILL 中的反馈收集机制评估是否需要更新 skill、更新哪些文件。常见排障可补充到本文件对应章节。

### 31.4 维护者溯源

修改或核对某条要求的来源时，见 `traceability.md`（源文档 → skill 落点对应表；**Workflow 与源文档对齐状态表**）。源文档不随 skill 分发，仅维护时参考。
