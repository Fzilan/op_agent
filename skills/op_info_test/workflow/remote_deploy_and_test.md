# remote_deploy_and_test 操作手册

## 概述

本章节用于指导 `op_info_test` 的远端执行闭环：提交测试任务、等待任务完成、读取结果、失败时下载产物用于分析。

目标是通过统一 API 完成“客户端发起 + 服务端执行”的流程，避免人工登录服务端逐步操作。

## 背景与角色

该流程包含两个角色：

1. `server`（`remote_runner_server.py`）：常驻服务，负责接收任务、在工作目录拉代码并执行测试、产出 `summary.json`/日志等文件。
2. `client`（`remote_runner_client.py`）：命令行入口，负责提交任务、轮询状态、查询摘要、下载失败产物到本地。

部署关系有两种：

1. `server` 和 `client` 在同一台机器（local 模式），可直接访问 `/tmp/op_info_artifacts/<job_id>/`。
2. `server` 在远端机器、`client` 在本地机器（remote 模式），通过 `--server http://<server_ip>:18080` 调用 API，推荐使用 `status/download` 获取结果和产物。

## 1. 前置条件

本地已完成用例生成、修改并推送分支。

如果接收任务指令中未明确server_ip, 则使用localhost作为server_ip。

## 2. 标准操作流程

### 步骤 0：启动服务端（远端机器）

```bash
cd $MINDSPORE_ROOT
python .agents/skills/op_info_test/scripts/remote_runner_server.py \
  --host 0.0.0.0 \
  --port 18080 \
  --state-file /tmp/op_info_state.json \
  --lock-file /tmp/op_info_runner.lock \
  --artifact-root /tmp/op_info_artifacts \
  --workspace-root /tmp/op_info_workspace
```

服务端已在测试任务开始前独立启动, 在测试任务中无需关注。

### 步骤 1：提交任务（客户端机器）

```bash
cd $MINDSPORE_ROOT
python .agents/skills/op_info_test/scripts/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  submit \
  --repo /home/panzh/mindspore \
  --branch <your_branch> \
  --test-cmd "pytest tests/st/ops/op_info_tests/*.py -q --maxfail=1 --tb=short" \
  --timeout-sec 3600
```

记录返回的 `job_id`。

### 步骤 2：等待任务结束

```bash
python .agents/skills/op_info_test/scripts/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  wait --job-id <job_id> --poll-interval-sec 10 --wait-timeout-sec 7200
```

如需手动查询状态：

```bash
python .agents/skills/op_info_test/scripts/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  status --job-id <job_id>
```

### 步骤 3：读取测试摘要

推荐在客户端通过 API 读取摘要（适用于服务端在远端机器）：

```bash
python .agents/skills/op_info_test/scripts/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  status --job-id <job_id>
```

若当前命令是在服务端机器本地执行，也可直接读取本地产物文件：

```bash
cat /tmp/op_info_artifacts/<job_id>/summary.json
```

重点字段：

1. `status`
2. `error_type`
3. `failed_cases`
4. `top_traceback`

若 `status=failed`，可下载远端产物到本地：

```bash
python .agents/skills/op_info_test/scripts/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  download --job-id <job_id> --output ./<job_id>_artifacts.zip
```

### 步骤 4：按结果进入下一轮

1. `status=success`：闭环结束。
2. `error_type=testcase`：按 `failed_cases/top_traceback` 修正用例，重新执行步骤 1 到步骤 4。
3. `error_type=infra`：停止自动改用例，先处理环境问题。

## 3. 可选操作

取消任务：

```bash
python .agents/skills/op_info_test/scripts/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  cancel --job-id <job_id>
```
