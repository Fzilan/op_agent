# Workflow 与源文档核对报告

本报告记录各 workflow 与对应源文档的逐节核对结果，用于落实 `traceability.md` 中「Workflow 与源文档对齐」纪律。  
核对方法：打开源文档，按章节/表格逐项问「该条在 workflow 或 checklists 中是否体现？」；未体现则记入下表「遗漏/建议补充」。

---

## 08-testing ↔ 3.算子开发、7.接口性能自验工具、4.算子关键特性

| 源文档 | 章节/内容 | workflow/checklists 现状 | 遗漏/建议补充 |
| --- | --- | --- | --- |
| 3. 算子开发 | 3.4.3 调试反向：save_graphs / save_graphs_path 存 IR 图，13_execute_xxx.ir 为后端图 | 08 未提 bprop 调试存图 | 08 补充：反向调试可设 `save_graphs=True` 查看 IR，13_execute_*.ir 为后端图（见 reference §7 / 源文档 3.4.3） |
| 3. 算子开发 | 3.2.3 InferValue 验证：UT 增加 infervalue 用例 + IR 图看常量折叠 | 08 未提 InferValue 验证 | 按需在 08 或 checklists 注明：InferValue 可通过 UT 与 IR 图验证（见 3.2.3） |
| 7. 接口性能自验工具 | 整网打点：start_hook_net(hook_inside)，需在网络执行前；单 API：start_analysis/end_analysis 包住循环 | 08 仅「性能/显存需用户执行」，未写 apitimewrapper 步骤 | 08 补充「性能自验（按需）」：整网用 start_hook_net；单 API 用 start_analysis/end_analysis；参考 reference §12、源文档 7 |
| 4. 算子关键特性 | 4.2 set_inputs：Cell.set_inputs(动态 shape Tensor) 用于编译，再传实际输入；-1 表示该维动态 | 08 未提 set_inputs 自验动态 shape | 08 或 checklists 补充：动态 shape 测试可用 net.set_inputs(shape 含 None 的 Tensor) 自验（见 4.2） |
| 4. 算子关键特性 | 4.3 vmap：注册 VmapRule、测试结果/IR/效率、性能自测表、owner 检视 | 08 未提 vmap；reference §23 为可选 | 已在 reference §23 标为可选；08 成功标准已含「覆盖场景」，vmap 若需要则在 08 加一步「vmap 测试（按需）」并引用 §23 |

**08 已覆盖**：C++ UT、Python ST、Python UT、精度零偏差、显存对齐、组合分层验证、用户配合环节、稳定性 100 次。

---

## 07-export ↔ 2.接口开发

| 源文档 | 章节/内容 | workflow/checklists 现状 | 遗漏/建议补充 |
| --- | --- | --- | --- |
| 2.1 | 接口对标 PyTorch：接口名、参数名、顺序与默认值、算法与精度、输入范围(dtype)、性能≥1/2 PyTorch、Parameter 名/name；无法一致需 CCB/评审 | 07 未提对标要求 | 07 补充「开始前/约束」：接口需对标 PyTorch（名/参/精度/范围/性能），见 reference §19、源文档 2.1 |
| 2.2.1 | functional：_get_cache_prim 获取原语；导入需在 __init__.py 对应类别 + __all__ 两处添加 | 07 有 _get_cache_prim；未强调 __all__ 与类别两处 | 07 Step 1 明确：导出时在对应类别和 __all__ **两处**添加算子名（见 2.1.3/2.2） |
| 2.3 | nn：Cell 子类；init 初始化算子与属性；construct 为执行入口，**不能 raise**，用 @constexpr 做编译期校验 | 07 表格有「construct 不直接 raise，用 @constexpr」 | 已覆盖 |
| 2.4 | Tensor：PyNative/KBK 见 Tensor 重载指导；**GE 需改 resource.cc + standard_method.py**，校验函数不能接收 Tensor 入参 | 07 仅写「覆盖 PyNative/KBK/GE」 | 07 补充：GE 模式需在 resource.cc 注册映射、standard_method.py 实现，校验不能接收 Tensor（见 2.4） |

**07 已覆盖**：ops 导出、functional/nn/Tensor 要点表、非 Ascend 占位、_get_cache_prim。

---

## 06-bprop ↔ 算子反向注意事项、aclnn 开发示例（反向分析）

| 源文档 | 章节/内容 | workflow/checklists 现状 | 遗漏/建议补充 |
| --- | --- | --- | --- |
| 算子反向注意事项 §1 | 不可微入参用 ib->OutZeros；**全部不可微用 ReturnZeros** | 06 Step 3 表有 ReturnZeros | 已覆盖 |
| §2 | 梯度确实为 0 时用 **ZerosLikeExt**，不用 ZerosLike/OutZeros | 06 Step 3 表有 ZerosLikeExt | 已覆盖 |
| §3 | inplace 反向：x 与输出为同一对象时，**只要有一个被用于反向就不能加入 unused_inputs**；需旧 self 时用 **CloneInplaceInput** | 06 未写 inplace 不能误标 unused；未写 CloneInplaceInput | 06 补充：inplace 算子若输入/输出同一对象，用于反向的不能进 SetUnusedInputs；需「更新前 self」时注册 CloneInplaceInput（见 reference §14、源文档 §3） |
| §6 | 反向中使用 inplace 时 KBK 动态 shape 可能不保序，用 **ib->Depend(target, inplace_call)** 规避 | 06 Step 3 表有「KBK 动态 shape inplace → Depend」 | 已覆盖 |
| §7 | str 类型参数位置若返回 OutZeros，**kbk 反向动态 shape 可能报错** | 06 未提 | 06 或 reference §14 补充：str 参数梯度返回 OutZeros 时，KBK 动态 shape 可能报错，需以实际为准（见源文档 §7） |

**06 已覆盖**：基本接线、I/O 个数、need_compute_grad_out、OutZeros、SetUnusedInputs、inplace Depend。

---

## 05-kbk、04-pyboost ↔ aclnn 开发示例、ResizeKernelLaunch

- **aclnn 开发示例**：yaml 三类（直通/仅映射名/自定义）、**default: None 无法自动生成需按第三类**、aclnn_config 映射、手写 Customize 流程。04/05 已有「路径 1/路径 2」与 reference §5/§6/§16/§29，未单独写「yaml 中 default: None 须按第三类处理」——建议在 01-yaml 或 04 的输入里注明。
- **ResizeKernelLaunch**：已在 reference §16 与 05 Step 3 体现；05 未再展开 compute-depend 的 Sync 版本与 GetOutputs——reference 已有，可维持现状。

**结论**：04/05 与 reference 一致即可，主要源文档为 aclnn 开发示例 + reference；少量可补充点已列上，标为「已核对，小补」。

---

## 03-general-infer、02-code-generation、01-yaml-definition、00-pre-checks、10-delivery

- **03**：源文档 3.2.2 GeneralInfer、3.2.3 InferValue、4.1 动态 shape 分类；reference §4/§27 已摘要；03 引用 §4/§27，**已核对**，无大漏。
- **02**：多源（YAML/gen_ops）；reference §2/§3；02 已引用，**已核对**。
- **01**：多源（YAML/接口策略）；reference §2/§19；01 已引用，**已核对**。
- **00**：2.接口开发、ACLNN 整体流程、PTA 审查；reference §19/§25/§28；00 已引用，**已核对**。
- **10**：11.算子开发开源运作规范、Aclnn 整体流程；reference §10/§13/§21/§30；10 已引用，**已核对**。

---

## 核对状态汇总

| Workflow | 主要源文档 | 核对结果 | 补充动作 |
| --- | --- | --- | --- |
| 09-docs | 5. 资料开发指导 | 已对齐（2025-02） | 已补全 |
| 08-testing | 3、7、4 | 已核对，有遗漏 | 见上表，补 4 条 |
| 07-export | 2. 接口开发 | 已核对，有遗漏 | 见上表，补 3 条 |
| 06-bprop | 算子反向注意事项 | 已核对，有遗漏 | 见上表，补 2 条 |
| 05-kbk | aclnn 示例、ResizeKernelLaunch | 已核对，小补 | 可选：01 注明 default:None |
| 04-pyboost | aclnn 示例 | 已核对，小补 | 同上 |
| 03/02/01/00/10 | 多源 / reference 摘要 | 已核对 | 维持引用 reference |

下次全面检查时，按本报告逐项确认「补充动作」是否已合入，并更新 traceability 对齐状态表日期。
