"""Minimal newline-delimited JSON-RPC client for the public MCP stdio surface.

The child is spawned with a read-only profile, a throwaway vault path, and an
environment allowlist that contains no provider or cloud keys. Replies are
matched by request id; out-of-order replies are kept rather than discarded.
No environment value is printed anywhere in this module.

Binary resolution (honest boundary): this module does NOT read an environment
variable or search the filesystem itself — ``Client`` receives an explicit
``binary`` path from its caller. The bench runner resolves that path as follows:

* ``MEMNOTH_MCP_BIN`` (exact env var name) wins when set;
* otherwise it auto-discovers ``memnoth-mcp`` / ``memnoth-mcp.exe`` in the CLI
  checkout's ``target/debug`` and ``target/release`` dirs (the sibling ``CLI/``
  checkout is the default; override the root with ``MN_CLI_ROOT``);
* with neither available, the runner records a structured ``blocked`` result
  for the mcp-stdio lane rather than fabricating a binary or a score.

No absolute machine path is baked into this module or the runner's discovery.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

SCHEMA = "mcp-stdio-adapter/1"
PROTOCOL_VERSION = "2026-07-28"
CLIENT_NAME = "memnoth-bench"
CLIENT_VERSION = "0.1.0"
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_PAGES = 32

_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "WINDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)
_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|"
    r"AUTHORIZATION|AUTH|COOKIE)[A-Z0-9_]*)\s*[=:]\s*(\"[^\"]*\"|'[^']*'|\S+)"
)
_MAX_DIAGNOSTIC_CHARS = 400
_STDERR_TAIL = 32


def redact(text: Any) -> str:
    """Bound and redact child-produced diagnostic text."""
    raw = "" if text is None else str(text)
    return _ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", raw)[
        :_MAX_DIAGNOSTIC_CHARS
    ]


def _meta() -> dict[str, Any]:
    return {
        "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientCapabilities": {"tools": {}},
    }


def _request_message(request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Pure JSON-RPC request builder; useful for wire-level self-checks."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }


class MCPError(RuntimeError):
    """A transport, protocol, or JSON-RPC error with a machine-readable kind."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        code: int | None = None,
        stderr_tail: list[str] | None = None,
    ) -> None:
        super().__init__(redact(message))
        self.kind = kind
        self.code = code
        self.stderr_tail = stderr_tail or []

    def envelope(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": str(self),
            "code": self.code,
            "stderr_tail": [redact(line) for line in self.stderr_tail[-8:]],
        }


class Client:
    """One stdio MCP server process, driven with newline-delimited JSON-RPC.

    Usage::

        with Client(binary, db_path) as client:
            client.handshake()
            tools = client.tools_list()
            result = client.call("mem_search", {"query": "serial"})
    """

    def __init__(
        self,
        binary: Path | str,
        db_path: Path | str,
        profile: str = "memory",
        read_only: bool = True,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_pages: int = MAX_PAGES,
    ) -> None:
        self.binary = Path(binary).expanduser()
        self.db_path = Path(db_path).expanduser()
        self.profile = profile
        self.read_only = read_only
        self.timeout = timeout
        self.max_pages = max_pages

        self.proc: subprocess.Popen[bytes] | None = None
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL)
        self.stashed: dict[int, dict[str, Any]] = {}
        self.next_id = 0
        self.handshaken = False
        self.handshake_result: dict[str, Any] | None = None

    # -- lifecycle ---------------------------------------------------------

    def _child_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for name in _ENV_ALLOWLIST:
            value = os.environ.get(name)
            if value is not None:
                env[name] = value
        env["MEMNOTH_HOME"] = str(Path(self.db_path).parent)
        env["MEMNOTH_MCP_PROFILE"] = self.profile
        env["MEMNOTH_MCP_READ_ONLY"] = "1" if self.read_only else "0"
        return env

    def _reader(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            for raw in self.proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else str(raw).strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self.queue.put({"kind": "bad_line", "text": redact(line)})
                    continue
                if isinstance(message, dict):
                    self.queue.put({"kind": "message", "message": message})
        except (OSError, ValueError):
            pass
        finally:
            self.queue.put({"kind": "eof"})

    def _stderr_reader(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        try:
            for raw in self.proc.stderr:
                line = raw.decode("utf-8", errors="replace").rstrip() if isinstance(raw, bytes) else str(raw).rstrip()
                self.stderr_tail.append(redact(line))
        except OSError:
            pass

    def start(self) -> None:
        """Spawn the server. Idempotent; a missing binary raises ``MCPError``."""
        if self.proc is not None:
            return
        if not self.binary.is_file():
            raise MCPError(
                "binary_not_found",
                "MCP server binary not found; set MEMNOTH_MCP_BIN or build the public server",
            )
        argv = [str(self.binary), "--db", str(self.db_path)]
        try:
            self.proc = subprocess.Popen(
                argv,
                env=self._child_env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise MCPError("spawn_failed", f"cannot start MCP server: {type(exc).__name__}") from None

        # Daemon threads: a hung server must never keep this process alive.
        subprocess_thread = threading.Thread(target=self._reader, daemon=True)
        stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
        subprocess_thread.start()
        stderr_thread.start()

    def close(self) -> None:
        """Stop the child. Safe to call twice and safe on a partially started client."""
        proc = self.proc
        if proc is None:
            return
        self.proc = None
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
                proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass

    def __enter__(self) -> "Client":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- wire --------------------------------------------------------------

    def _round_trip(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.start()
        assert self.proc is not None and self.proc.stdin is not None
        self.next_id += 1
        request_id = self.next_id
        payload = json.dumps(_request_message(request_id, method, params), separators=(",", ":"))
        try:
            self.proc.stdin.write(payload.encode("utf-8") + b"\n")
            self.proc.stdin.flush()
        except (OSError, BrokenPipeError):
            raise MCPError(
                "broken_pipe",
                "MCP server closed its stdin before the reply",
                stderr_tail=list(self.stderr_tail),
            ) from None

        deadline = time.monotonic() + self.timeout
        stashed = self.stashed
        if request_id in stashed:
            return stashed.pop(request_id)

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MCPError(
                    "timeout",
                    f"timed out after {self.timeout} seconds waiting for {method}",
                    stderr_tail=list(self.stderr_tail),
                )
            try:
                event = self.queue.get(timeout=remaining)
            except queue.Empty:
                raise MCPError(
                    "timeout",
                    f"timed out after {self.timeout} seconds waiting for {method}",
                    stderr_tail=list(self.stderr_tail),
                ) from None

            if event["kind"] == "bad_line":
                raise MCPError(
                    "protocol",
                    "server wrote a line that is not JSON-RPC",
                    stderr_tail=list(self.stderr_tail),
                )
            if event["kind"] == "eof":
                raise MCPError(
                    "eof",
                    "MCP server closed stdout before the reply",
                    stderr_tail=list(self.stderr_tail),
                )
            reply = event["message"]
            reply_id = reply.get("id")
            if isinstance(reply_id, int) and reply_id != request_id:
                stashed[reply_id] = reply
                continue
            if "error" in reply:
                error = reply["error"]
                code = error.get("code") if isinstance(error, dict) else None
                message = error.get("message", "JSON-RPC error") if isinstance(error, dict) else "JSON-RPC error"
                raise MCPError(
                    "rpc_error",
                    str(message),
                    code=code if isinstance(code, int) else None,
                    stderr_tail=list(self.stderr_tail),
                )
            return reply

    # -- public API --------------------------------------------------------

    def _client_info(self) -> dict[str, str]:
        return {"name": CLIENT_NAME, "version": CLIENT_VERSION}

    def handshake(self) -> dict[str, Any]:
        """Discover the server revision. Falls back to ``initialize`` once."""
        params = {"_meta": _meta(), "clientInfo": self._client_info()}
        try:
            reply = self._round_trip("server/discover", params)
        except MCPError as exc:
            if exc.kind != "rpc_error" or exc.code != -32601:
                raise
            reply = self._round_trip(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "clientInfo": self._client_info(),
                },
            )
        self.handshake_result = reply.get("result") if isinstance(reply.get("result"), dict) else {}
        self.handshaken = True
        return self.handshake_result

    def tools_list(self) -> list[dict[str, Any]]:
        """Return every advertised tool, following cursors with a page cap."""
        if not self.handshaken:
            self.handshake()
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _page in range(self.max_pages):
            params: dict[str, Any] = {"_meta": _meta()}
            if cursor:
                params["cursor"] = cursor
                if cursor in seen_cursors:
                    raise MCPError("protocol", "tools/list cursor did not advance")
                seen_cursors.add(cursor)
            reply = self._round_trip("tools/list", params)
            result = reply.get("result")
            if not isinstance(result, dict):
                raise MCPError("protocol", "tools/list reply has no result object")
            page_tools = result.get("tools", [])
            if not isinstance(page_tools, list):
                raise MCPError("protocol", "tools/list result.tools is not a list")
            tools.extend(item for item in page_tools if isinstance(item, dict))
            next_cursor = result.get("nextCursor")
            cursor = next_cursor if isinstance(next_cursor, str) and next_cursor else None
            if not cursor:
                return tools
        raise MCPError("protocol", f"tools/list exceeded {self.max_pages} pages")

    def call(self, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call one tool. A tool-level ``isError`` is returned, not fabricated away."""
        if not self.handshaken:
            self.handshake()
        reply = self._round_trip(
            "tools/call",
            {
                "name": tool,
                "arguments": args if isinstance(args, dict) else {},
                "_meta": _meta(),
            },
        )
        result = reply.get("result")
        return result if isinstance(result, dict) else {"result": result}


if __name__ == "__main__":
    print(f"{SCHEMA}: self-check")
    print(json.dumps(_request_message(1, "server/discover", {"_meta": _meta()}), sort_keys=True))
    print(redact("MEMNOTH_MCP_TOKEN=example-not-a-real-token"))
