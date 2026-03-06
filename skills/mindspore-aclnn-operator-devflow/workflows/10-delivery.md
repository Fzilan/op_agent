# Workflow 10: 转测交付

## 目标

完成 RFC issue、PR 提交、验收 checklist，确保交付件齐全。

## 输入

- **全部实现代码**：YAML / Infer / PyBoost / KBK / BPROP / 接口 / 测试 / 文档

## 输出

- **Feature 文档**：完整填写所有章节（评审与转测必须产物）
- **RFC issue**：完整的需求/方案/验收信息
- **PR**：代码提交（通过质量门禁）
- **验收 checklist**：填写完整

---

## 执行步骤

### Step 1：质量门禁检查（`reference.md` §21）

提交前确保通过：
- 行长 ≤ 120
- UTF-8 编码
- 无行尾空格
- Check_Pylint / Check_Notebooklint / Check_Shellcheck 等

### Step 2：补齐 Feature 文档（`reference.md` §30）

Feature 文档是评审与转测交付的**必须产物**，此时需补齐开发过程中各章节：

1. **检查前半部分**：确认 Pre-B 阶段填写的 §1-§4、§6、§8 是否需要根据实际开发更新
2. **补齐开发章节**：
   - §5（YAML 定义）← Step 1 产出
   - §7（执行模式与适配）← Step 4/5 产出
   - §9（动态 Shape/Rank）← Step 3 产出
   - §10（异常与校验）← Step 3/4 产出
   - §11（反向 BPROP）← Step 6 产出
   - §12（测试方案）← Step 8 产出
3. **填写 §13（代码与文件改动说明）**：列出所有新增/修改的文件路径
4. **填写 §14（验收报告）**：
   - 资料验证表（17 项）：逐项自测并标注结果
   - 功能验证表（27 项）：逐项自测并标注结果
   - 性能验证表（4 项）：逐项自测并标注结果
   - 安全编码检视表（12 项）：逐项自测并标注结果
5. **更新 §3 任务清单**：每项的最终状态改为"新增/修改/无变更/不涉及"

### Step 3：RFC issue（`reference.md` §13）

RFC 内容建议：
- 背景与目标、交付范围、使用约束
- 方案设计（必要时外链归档）
- 验收与自测依据：UT/ST 覆盖、稳定性
- 代码 PR 与测试仓 PR 链接

### Step 4：交付件检查（`reference.md` §10.3）

转测覆盖范围：
- 接口形态：NN / functional / Tensor
- 后端：Ascend（+ GPU/CPU 不回退说明）
- 模式：动态图 / 静态图 / KBK
- shape：动态/静态
- 维度：泛化性 + 精度 + 性能

### Step 5：PR 提交

- 提交前过 `checklists.md` 末尾的"提交前必检 Top-25"
- PR 描述包含 RFC 链接和关键变更说明

---

## 成功标准

- [ ] Feature 文档所有 14 个章节已填写完整
- [ ] Feature 文档验收报告四张表（资料验证 + 功能验证 + 性能验证 + 安全编码）已逐项自测
- [ ] 质量门禁全部通过
- [ ] RFC issue 完整，包含方案/验收/PR 链接
- [ ] 提交前 Top-25 checklist 全部 ✅
- [ ] PR 描述清晰，关联 RFC 和 Feature 文档
- [ ] 交付件覆盖范围确认

---

## 流程完成

```
✅ Pre-A/B/C: 前置检查与方案设计
✅ Step 1: YAML 定义
✅ Step 2: 代码生成
✅ Step 3: GeneralInfer
✅ Step 4: PyBoost
✅ Step 5: KBK
✅ Step 6: BPROP 注册
✅ Step 7: 导出与占位
✅ Step 8: 测试
✅ Step 9: 文档
✅ Step 10: 转测交付
```

**ACLNN 算子开发全流程完成！**
