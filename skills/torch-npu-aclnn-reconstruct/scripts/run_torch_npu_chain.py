#!/usr/bin/env python3
"""Run torch_npu reconstruct + render with gap scan enabled."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
import re


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run torch_npu chain reconstruction and render reports")
    p.add_argument("--workspace", type=Path, default=None)
    p.add_argument("--op-plugin-root", type=Path, default=None)
    p.add_argument("--repos-config", type=Path, default=None, help="path to repos config yaml")
    p.add_argument("--top-ops", type=str, required=True, help="comma separated, e.g. add,div")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--max-depth", type=int, default=6)
    return p.parse_args()


def parse_repos_yaml(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Minimal parser for:
    # repos:
    #   op_plugin: /abs/path
    in_repos = False
    for ln in text.splitlines():
        raw = ln.rstrip()
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        if re.match(r"^\s*repos\s*:\s*$", raw):
            in_repos = True
            continue
        if not in_repos:
            continue
        m = re.match(r"^\s{2,}([A-Za-z0-9_\-]+)\s*:\s*(.+?)\s*$", raw)
        if not m:
            # leave repos block when dedented to top-level key
            if re.match(r"^\S", raw):
                break
            continue
        k = m.group(1).strip()
        v = m.group(2).strip().strip("'").strip('"')
        if v:
            out[k] = v
    return out


def is_op_plugin_root(p: Path) -> bool:
    return (p / "op_plugin" / "config" / "op_plugin_functions.yaml").exists()


def discover_op_plugin_root(repo_root: Path) -> Path | None:
    candidates: list[Path] = []
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "workspace" / "op-plugin",
            cwd / "code" / "op-plugin",
            repo_root / "workspace" / "op-plugin",
            repo_root.parent / "code" / "op-plugin",
            repo_root.parent / "workspace-mindspore-framework" / "code" / "op-plugin",
        ]
    )
    for c in candidates:
        if is_op_plugin_root(c.resolve()):
            return c.resolve()
    return None


def resolve_paths(args: argparse.Namespace, repo_root: Path) -> tuple[Path, Path, Path]:
    cfg_path = (args.repos_config.resolve() if args.repos_config else (repo_root / ".op_factory" / "repos.yaml").resolve())
    cfg = parse_repos_yaml(cfg_path)

    # op-plugin root: arg > env > config > discover
    op_root = None
    if args.op_plugin_root:
        op_root = args.op_plugin_root.resolve()
    elif os.getenv("OP_PLUGIN_ROOT"):
        op_root = Path(os.environ["OP_PLUGIN_ROOT"]).expanduser().resolve()
    elif cfg.get("op_plugin"):
        op_root = Path(cfg["op_plugin"]).expanduser().resolve()
    else:
        op_root = discover_op_plugin_root(repo_root)

    if not op_root or not is_op_plugin_root(op_root):
        raise SystemExit(
            "op-plugin root not resolved. Set one of: "
            "--op-plugin-root /abs/path, OP_PLUGIN_ROOT env, .op_factory/repos.yaml(repos.op_plugin), "
            "or provide a discoverable path containing op_plugin/config/op_plugin_functions.yaml"
        )

    # workspace: arg > env > config.workspace > parent(op-plugin)
    workspace = None
    if args.workspace:
        workspace = args.workspace.resolve()
    elif os.getenv("ANALYZER_WORKSPACE"):
        workspace = Path(os.environ["ANALYZER_WORKSPACE"]).expanduser().resolve()
    elif cfg.get("workspace"):
        workspace = Path(cfg["workspace"]).expanduser().resolve()
    else:
        workspace = op_root.parent

    # out-dir: arg > workspace/runs/<ts>/reconstruct-chains
    if args.out_dir:
        out_dir = args.out_dir.resolve()
    else:
        run_id = time.strftime("run-%Y%m%d-%H%M%S")
        out_dir = (repo_root / "workspace" / "runs" / run_id / "reconstruct-chains").resolve()

    return workspace, op_root, out_dir


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    tool_root = repo_root / "tools" / "op-analysis-lsp"
    aclnn_set = tool_root / "exports" / "torch_npu_aclnn_op_full_set.txt"
    run_py = tool_root / "reconstruct-chains" / "torch_npu" / "run.py"
    render_py = tool_root / "reconstruct-chains" / "common" / "render_report.py"
    workspace, op_plugin_root, out_dir = resolve_paths(args, repo_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[preflight] workspace={workspace}", flush=True)
    print(f"[preflight] op_plugin_root={op_plugin_root}", flush=True)
    print(f"[preflight] out_dir={out_dir}", flush=True)

    cmd1 = [
        "python3",
        str(run_py),
        "--workspace",
        str(workspace),
        "--op-plugin-root",
        str(op_plugin_root),
        "--top-ops",
        args.top_ops,
        "--aclnn-set",
        str(aclnn_set.resolve()),
        "--enable-aclnn-gap-scan",
        "--max-depth",
        str(args.max_depth),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd1, check=True)

    cmd2 = [
        "python3",
        str(render_py),
        "--chains-json",
        str((out_dir / "chains.json").resolve()),
        "--summary-json",
        str((out_dir / "summary.json").resolve()),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd2, check=True)
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
