# ACLNN Builder Test Framework

This directory contains the runtime test framework for `skills/aclnn-builder`.

## What It Tests

Each test case defines:

- a prompt that triggers the skill
- one or more source repositories that must be isolated for the run
- the exact source files expected to be `added`, `modified`, or `deleted` when the task finishes
- optional artifact globs for markdown or trace files that are allowed to exist without being treated as source changes

The framework is aimed at end-to-end skill validation:

- a fresh executor process is used for every case
- every repo fixture is isolated per case using `git worktree` by default
- file-change validation is based on the isolated repo diff after the run
- non-committed analysis artifacts are captured separately from source changes

## Execution Model

The runner lives in `framework.py` and the CLI entrypoint is `run_skill_cases.py`.

Per case, the runner:

1. creates a new case directory under `aclnn_skill_test_runs/<run_id>/<case_id>/`
2. prepares isolated checkouts for each declared repo
3. renders the case prompt with placeholders such as `{{repo:mindspore}}`, `{{artifact_dir}}`, and `{{skill:aclnn_builder}}`
4. invokes `codex exec` in an ephemeral session
5. collects the git diff for each isolated repo
6. validates the actual `added/modified/deleted` source files against the case contract
7. writes JSON summaries and raw executor logs into the case directory

This isolation model keeps both workspace state and model session context clean across consecutive runs.
Use `--workers N` to run multiple cases concurrently; result aggregation happens after all workers finish.

## Case Contract

The manifest lives in `cases.yaml`.

Important fields:

- `prompt`: the exact task prompt
- `{{skill:<name>}}`: skill invocation placeholder; renders as `$<name>` for Codex and `/<name>` for OpenCode/Claude Code
- `repos[*].source`: source repo path; for the primary `mindspore` repo this may be omitted and will default to the invocation cwd or `--ms-root`
- `repos[*].expected_changes`: required source file contract; every listed path must appear in the matching add/modify/delete bucket, while extra diff entries are allowed
- `artifact_globs`: files that may appear but should not count as source changes
- `enabled: false`: useful for templates or expensive live cases that should not run by default

## Usage

By default, invoke the runner from the MindSpore repository root. If the current working directory is not the MindSpore repo, pass `--ms-root /path/to/mindspore`.
If you do not pass `--runs-root`, outputs go under `<ms_root>/aclnn_skill_test_runs`.
If you do not pass `--skill-path`, the runner expects skills under `<ms_root>/.codex/skills` and will fail if that directory does not exist.
Before invoking Codex, the runner copies the resolved skills tree into the isolated MindSpore checkout under `.codex/skills`.
Use `--op-plugin` to pass the `op-plugin` repository path. In prompts, `{{op_plugin_dir}}` is replaced with that resolved path. `--op-plugin` is required for non-dry-run execution.
Use `--executor` to choose `codex` (default), `opencode`, or `claudecode`.
Use `--workers` to control case-level parallelism; the default is `1`.
A typical run of all test cases:

```bash
python <op_agent_root>/skills/aclnn-builder/tests/run_skill_cases.py --op-plugin=<op_plugin_dir> --workers 4
```

Dry-run prompt rendering and sandbox setup:

```bash
python3 <op_agent_root>/skills/aclnn-builder/tests/run_skill_cases.py --dry-run --include-disabled
```

Run one live case:

```bash
python3 <op_agent_root>/skills/aclnn-builder/tests/run_skill_cases.py --case example_aclnn_abs --include-disabled
```

Run from another working directory:

```bash
python3 <op_agent_root>/skills/aclnn-builder/tests/run_skill_cases.py \
  --executor codex \
  --ms-root /path/to/mindspore \
  --skill-path /path/to/mindspore/.codex/skills \
  --op-plugin /path/to/op-plugin \
  --case example_aclnn_abs \
  --include-disabled
```

By default, isolated worktrees are kept for manual inspection. Clean them up automatically with:

```bash
python3 <op_agent_root>/skills/aclnn-builder/tests/run_skill_cases.py --cleanup-sandboxes
```

## Plan

The framework is intentionally split into layers:

- manifest contract: stable, reviewable case definitions in YAML
- isolation layer: fresh worktrees and a fresh executor session per case
- execution layer: pluggable executor, with `CodexExecutor` as the default
- validation layer: exact source diff matching plus separate artifact capture
- unit-test layer: local pytest coverage that exercises the framework without a live model call

This is enough to start adding real `aclnn-builder` cases while keeping the harness deterministic and cheap to validate locally.
