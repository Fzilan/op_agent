---
name: op_info_test
description: Generate and validate MindSpore Python ST op_info tests end-to-end (case generation, temporary isolation patch, remote deploy/test, rerun, and final cleanup). Use when users ask to add/repair/verify op_info-based ST for one or more operators, run op_info smoketest, or produce remote test evidence and coverage summary for op_info cases.
---

执行 op_info 测试端到端闭环，优先直接落地，不在中途停下提问；仅在权限受限、关键信息缺失且无法合理推断时提问。

## 快速路由

- 指令包含 `smoketest` / `冒烟`：直接执行 [workflow/smoke.md](workflow/smoke.md)。
- 其他 op_info 用例新增/修复请求：执行下述完整闭环。

## 完整闭环流程

1. 从 `master` 新建分支：`add_opinfo_test_[op_name]`。
2. 按 [workflow/op_info_generation.md](workflow/op_info_generation.md) 生成/更新目标接口用例。
3. 提交测试用例：commit message 使用 `op_info_test: add xxx`（`xxx` 为接口名或接口组）。
4. 按 [workflow/patch_out_old_tests.md](workflow/patch_out_old_tests.md) 添加“仅保留新增接口测试”的临时 patch。
5. `git push` 当前分支。
6. 按 [workflow/remote_deploy_and_test.md](workflow/remote_deploy_and_test.md) 执行远端部署与测试。
7. 根据远端结果循环处理：
   - `error_type=testcase`：修复用例并重复步骤 3-6，直到通过。
   - `error_type=infra`：停止自动改用例，转环境排障并记录阻塞点。
8. 远端通过后，移除临时 patch 提交并整理历史（保留干净的用例提交），再 `git push`。
9. 在当前工作目录生成 `分支名.md` 总结文档（不加入 git）：
   - 新增/修改的接口与用例清单。
   - 覆盖场景与边界。
   - 未覆盖场景及原因。
   - 远端任务与结果摘要。

## 完成判定

仅在以下条件全部满足时判定闭环完成：

1. 远端任务 `status=success`。
2. `summary.json` 的 `failed_cases` 为空。
3. 不存在未处理的 `error_type=testcase`。

## 执行约束

- 优先复用本 skill 内现有文档、脚本与模板，不重复造轮子。
- 不改动与目标接口无关的测试内容。
- 每次重跑都先确认临时 patch 状态与提交历史符合预期。
- 输出总结文档时明确写出“已覆盖/未覆盖/阻塞原因”，避免模糊表述。
