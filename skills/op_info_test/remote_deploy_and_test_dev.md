# remote_deploy_and_test 实现说明（仅用于 skill 开发演进）

## 1. 文档定位

1. 本文档描述远端 runner 的实现约束与输出契约。
2. 本文档不作为 skill 执行输入，执行阶段只使用 `workflow/remote_deploy_and_test.md`。
3. 本文档中的输出规范用于后续脚本演进
4. 设计不用过度考虑兼容性问题

## 2. 任务产物目录

1. 每个任务的产物目录固定为：`<artifact_root>/<job_id>/`。
2. 默认 `artifact_root=/tmp/op_info_artifacts`。
3. 目录内输出件包括：`pytest.log`、`junit.xml`、`env.txt`、`deploy_meta.json`、`summary.json`。

## 3. 输出件总览

| 输出件 | 格式 | 复杂度 | 生成条件 |
| --- | --- | --- | --- |
| `pytest.log` | UTF-8 文本 | 中 | 始终生成 |
| `junit.xml` | XML（pytest junit） | 中 | `test_cmd` 为 pytest 且输出到默认路径时 |
| `env.txt` | UTF-8 文本（key=value） | 低 | 始终生成 |
| `deploy_meta.json` | JSON | 中 | 始终生成 |
| `summary.json` | JSON | 高 | 始终生成 |

说明：

1. “复杂度=中/高”的输出件需要模板，模板位于 `template/` 目录。
2. JSON 输出统一由服务端 `write_json` 写出，编码 UTF-8、`ensure_ascii=true`、2 空格缩进、key 排序。

## 4. 各输出件详细规范

### 4.1 `pytest.log`

内容要求：

1. 每行前缀时间戳，格式：`[YYYY-MM-DD HH:MM:SS]`。
2. 至少包含结构化记录：`[runner] job_id=...`。
3. 至少包含结构化记录：`[runner] payload=...`（JSON 字符串）。
4. 每条命令至少包含：`[exec] cwd=...`、`[exec] cmd=...`、`[exec] return_code=...`。
5. 命令 stdout/stderr 合并写入同一日志文件，按读取顺序追加。

格式要求：

1. UTF-8 文本。
2. 换行符 `\n`。
3. 允许包含非结构化业务输出，不允许破坏上述结构化记录。

模板：

1. `template/pytest_log.template.txt`

### 4.2 `junit.xml`

内容要求：

1. 必须为 pytest 可解析的 junit XML。
2. 失败用例需体现在 `testcase/failure` 或 `testcase/error`。
3. `classname` 与 `name` 字段应可拼接为用例标识（`classname::name`）。

生成要求：

1. 当 `test_cmd` 含 `pytest` 且未显式设置 `--junitxml` 时，服务端自动追加：
   `--junitxml=<artifact_dir>/junit.xml`。
2. 若用户已在 `test_cmd` 中设置 `--junitxml`，服务端不覆盖该参数。

模板：

1. `template/junit_xml.template.xml`

### 4.3 `env.txt`

内容要求：

1. 每行 `key=value`。
2. 当前最小字段集必须包含：`time`。
3. 当前最小字段集必须包含：`host`。
4. 当前最小字段集必须包含：`python`。
5. 当前最小字段集必须包含：`git`。

格式要求：

1. UTF-8 文本。
2. 末尾保留换行。

### 4.4 `deploy_meta.json`

字段与要求：

1. `branch`（string，必填）：任务使用分支。
2. `commit`（string，必填，可为空）：任务指定提交；未指定时为空字符串。
3. `deploy_time`（string，必填）：UTC ISO8601 时间，示例 `2026-03-04T03:15:20+00:00`。
4. `runner_id`（string，必填）：服务端主机名（`os.uname().nodename`）。

模板：

1. `template/deploy_meta.template.json`

### 4.5 `summary.json`

字段与要求：

1. `job_id`（string，必填）。
2. `status`（string，必填）：`success`、`failed`、`timeout`、`canceled`。
3. `failed_cases`（array[string]，必填）：失败用例列表，去重且保持原顺序。
4. `error_type`（string，必填）：`""`、`infra`、`testcase`。
5. `top_traceback`（string，必填）：关键报错片段；`success` 时必须为空字符串。
6. `generated_at`（string，必填）：UTC ISO8601 时间。

状态一致性要求：

1. `status=success` 时：`failed_cases=[]`、`error_type=""`、`top_traceback=""`。
2. `status=failed` 且属于测试失败时：`error_type=testcase`。
3. `status=failed` 且属于环境/部署/依赖等失败时：`error_type=infra`。
4. `status=timeout` 或 `status=canceled` 时：`error_type=infra`。

模板：

1. `template/summary.template.json`

## 5. API 输出件规范（状态查询）

`GET /jobs/{job_id}` 返回 job 状态输出，其结构用于客户端轮询和闭环判断。

字段要求：

1. `job_id`（string，必填）。
2. `status`（string，必填）。
3. `payload`（object，必填）：原始提交参数标准化副本。
4. `artifact_uri`（string，必填）：`/artifacts/<job_id>/`。
5. `artifact_bundle_uri`（string，必填）：`/jobs/<job_id>/artifacts.zip`。
6. `error_type`（string，必填）。
7. `created_at`、`started_at`、`finished_at`（string/null，必填）。
8. `api_version`（string，必填）。
9. `summary`（object，终态必填）：与 `summary.json` 同结构。

模板：

1. `template/job_status_response.template.json`

## 6. API 输出件规范（产物下载）

`GET /jobs/{job_id}/artifacts.zip` 返回当前 job 的全部产物压缩包，用于客户端在 `status=failed` 后拉取分析。

行为要求：

1. 若 `job_id` 不存在，返回 404。
2. 若产物目录不存在，返回 404。
3. 返回 `Content-Type: application/zip`。
4. 返回 `Content-Disposition: attachment; filename="<job_id>_artifacts.zip"`。

## 7. 模板列表

1. `template/summary.template.json`
2. `template/deploy_meta.template.json`
3. `template/job_status_response.template.json`
4. `template/pytest_log.template.txt`
5. `template/junit_xml.template.xml`
