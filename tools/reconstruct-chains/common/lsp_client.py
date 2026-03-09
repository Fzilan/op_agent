#!/usr/bin/env python3
"""Minimal JSON-RPC LSP client over stdio."""

from __future__ import annotations

import json
import os
import select
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LspError(RuntimeError):
    """Raised when an LSP request fails."""


@dataclass
class Position:
    line: int
    character: int


class LspClient:
    def __init__(self, cmd: list[str], cwd: Path, timeout: float = 8.0) -> None:
        self.cmd = cmd
        self.cwd = cwd
        self.timeout = timeout
        self._next_id = 1
        self._buffer = b""
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        if self.proc.stdin is None or self.proc.stdout is None:
            raise LspError(f"failed to start LSP process: {' '.join(cmd)}")

    def _send(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii")
        assert self.proc.stdin is not None
        self.proc.stdin.write(header + raw)
        self.proc.stdin.flush()

    def _try_parse(self) -> tuple[dict[str, Any] | None, bytes]:
        marker = b"\r\n\r\n"
        header_end = self._buffer.find(marker)
        sep_len = 4
        if header_end < 0:
            marker = b"\n\n"
            header_end = self._buffer.find(marker)
            sep_len = 2
        if header_end < 0:
            return None, self._buffer

        header_blob = self._buffer[:header_end].decode("ascii", errors="replace")
        headers: dict[str, str] = {}
        for line in header_blob.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()

        content_length = headers.get("content-length")
        if content_length is None:
            raise LspError(f"missing content-length from {' '.join(self.cmd)}")
        length = int(content_length)
        body_start = header_end + sep_len
        body_end = body_start + length
        if len(self._buffer) < body_end:
            return None, self._buffer

        body = self._buffer[body_start:body_end]
        remain = self._buffer[body_end:]
        return json.loads(body.decode("utf-8")), remain

    def _read_message(self, timeout: float | None = None) -> dict[str, Any]:
        assert self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        budget = self.timeout if timeout is None else max(timeout, 0.0)
        deadline = time.time() + budget

        while True:
            parsed, remain = self._try_parse()
            if parsed is not None:
                self._buffer = remain
                return parsed

            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"timed out reading from {' '.join(self.cmd)}")

            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise TimeoutError(f"timed out reading from {' '.join(self.cmd)}")

            chunk = os.read(fd, 65536)
            if not chunk:
                raise LspError(f"LSP process closed stdout: {' '.join(self.cmd)}")
            self._buffer += chunk

    def request(self, method: str, params: dict[str, Any]) -> Any:
        req_id = self._next_id
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.time() + self.timeout

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting response for {method} from {' '.join(self.cmd)}")
            msg = self._read_message(timeout=remaining)
            if msg.get("id") != req_id:
                continue
            if "error" in msg:
                raise LspError(f"{method} failed: {msg['error']}")
            return msg.get("result")

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def initialize(self, root_uri: str, process_id: int) -> None:
        result = self.request(
            "initialize",
            {
                "processId": process_id,
                "rootUri": root_uri,
                "clientInfo": {"name": "reconstruct-chains", "version": "0.1"},
                "capabilities": {
                    "textDocument": {
                        "callHierarchy": {"dynamicRegistration": False},
                    },
                    "workspace": {
                        "symbol": {"dynamicRegistration": False},
                    },
                },
            },
        )
        if not isinstance(result, dict) or "capabilities" not in result:
            raise LspError(f"invalid initialize response from {' '.join(self.cmd)}")
        self.notify("initialized", {})

    def did_open(self, uri: str, language_id: str, text: str, version: int = 1) -> None:
        self.notify(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": version,
                    "text": text,
                }
            },
        )

    def prepare_call_hierarchy(self, uri: str, pos: Position) -> Any:
        return self.request(
            "textDocument/prepareCallHierarchy",
            {
                "textDocument": {"uri": uri},
                "position": {"line": pos.line, "character": pos.character},
            },
        )

    def definition(self, uri: str, pos: Position) -> Any:
        return self.request(
            "textDocument/definition",
            {
                "textDocument": {"uri": uri},
                "position": {"line": pos.line, "character": pos.character},
            },
        )

    def outgoing_calls(self, item: dict[str, Any]) -> Any:
        return self.request("callHierarchy/outgoingCalls", {"item": item})

    def close(self) -> None:
        try:
            self.request("shutdown", {})
        except Exception:
            pass
        try:
            self.notify("exit", {})
        except Exception:
            pass

        try:
            self.proc.terminate()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()
