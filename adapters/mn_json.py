"""Adapter for the public ``mn <verb> --json`` surface.

The child process gets a minimal environment: no inherited provider keys, no
cloud credentials, and a throwaway home directory. Nothing in this module reads
a key value, and nothing here prints the environment. Errors are returned as
structured envelopes so a caller can aggregate a failure without swallowing it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "mn-json-adapter/1"
DEFAULT_TIMEOUT_SECONDS = 60.0

# Only these variables reach the child. Keys and tokens are deliberately absent;
# use an allowlist instead of copying os.environ and trying to strip everything.
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
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)

# Redacts ``NAME=value`` / ``NAME: value`` shaped assignments in diagnostic
# text. Only text that came from a child process is passed through this.
_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|"
    r"AUTHORIZATION|AUTH|COOKIE)[A-Z0-9_]*)\s*[=:]\s*(\"[^\"]*\"|'[^']*'|\S+)"
)
_MAX_DIAGNOSTIC_CHARS = 800


def _utc_now() -> str:
    """Real wall-clock UTC timestamp, never a cached constant."""
    return datetime.now(timezone.utc).isoformat()


def redact(text: Any) -> str:
    """Return a bounded, key-redacted string for diagnostics only."""
    raw = "" if text is None else str(text)
    return _ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", raw)[
        :_MAX_DIAGNOSTIC_CHARS
    ]


def _isolated_env(home: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for name in _ENV_ALLOWLIST:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    env["HOME"] = home
    env["USERPROFILE"] = home
    env["MEMNOTH_HOME"] = home
    env.pop("MN_BIN", None)
    return env


def _candidate_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("mn.exe", "mn", "memnoth.exe", "memnoth")
    return ("mn", "memnoth")


def _candidate_dirs(repo_root: Path) -> list[Path]:
    cli_root = repo_root.parent / "CLI"
    dirs = [
        cli_root / "target" / "debug",
        cli_root / "target" / "release",
        cli_root / "target",
    ]
    configured = os.environ.get("CARGO_TARGET_DIR")
    if configured:
        target = Path(configured).expanduser()
        dirs.extend([target / "debug", target / "release", target])
    deduped: list[Path] = []
    for path in dirs:
        if path not in deduped:
            deduped.append(path)
    return deduped


def discover_bin() -> Path:
    """Find the public CLI binary.

    Order: ``MN_BIN``, ``PATH``, then the sibling CLI checkout's target dirs.
    Raises ``FileNotFoundError`` with a remediation hint when nothing is found.
    """
    override = os.environ.get("MN_BIN")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    found = shutil.which("mn")
    if found:
        return Path(found).resolve()

    repo_root = Path(__file__).resolve().parents[1]
    for directory in _candidate_dirs(repo_root):
        for name in _candidate_names():
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()

    raise FileNotFoundError("mn binary not found; set MN_BIN to a public CLI build")


def _decode(chunk: Any) -> str:
    if chunk is None:
        return ""
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", errors="replace")
    return str(chunk)


def _parse_envelope(stdout: str) -> dict[str, Any]:
    """Parse the single JSON object the CLI writes for ``--json``.

    If a wrapper emitted a banner first, scan backwards for the last parseable
    object rather than guessing at a prefix to discard.
    """
    stripped = stdout.strip()
    if not stripped:
        raise ValueError("empty stdout")
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    for line in reversed(stripped.splitlines()):
        candidate = line.strip()
        if not candidate.startswith("{"):
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no JSON object found on stdout")


def _error(
    kind: str,
    message: str,
    *,
    verb: str,
    args: list[str],
    returncode: int | None = None,
    stderr: str = "",
) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "schema": SCHEMA,
        "ok": False,
        "command": " ".join(["mn", verb, *args, "--json"]),
        "verb": verb,
        "version": None,
        "data": None,
        "error": {
            "kind": kind,
            "message": redact(message),
            "returncode": returncode,
            "stderr_tail": redact(stderr),
        },
        "observed_at": _utc_now(),
        "adapter": "mn-json",
    }
    return envelope


def run_verb(
    verb: str,
    args: list[str],
    bin: Path | None,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Run ``mn <verb> <args...> --json`` and return a normalized envelope.

    ``bin=None`` triggers discovery. ``cwd`` is only a working directory for
    committed public fixtures; the environment stays isolated either way.
    Every failure path returns the same envelope shape with ``ok=False`` and a
    structured ``error``; no exception escapes for a failed subprocess, and no
    environment value is ever emitted.
    """
    normalized_args = [str(item) for item in args]
    if bin is None:
        try:
            executable = discover_bin()
        except FileNotFoundError as exc:
            return _error("binary_not_found", str(exc), verb=verb, args=normalized_args)
    else:
        candidate = Path(bin).expanduser()
        if not candidate.is_file():
            return _error(
                "binary_not_found",
                "configured mn binary is not a file",
                verb=verb,
                args=normalized_args,
            )
        executable = candidate.resolve()

    argv = [str(executable), verb, *normalized_args, "--json"]
    with tempfile.TemporaryDirectory(prefix="mn-bench-home-") as home:
        env = _isolated_env(home)
        if cwd is not None and not Path(cwd).is_dir():
            return _error(
                "working_directory_not_found",
                "requested working directory for public fixtures does not exist",
                verb=verb,
                args=normalized_args,
            )
        try:
            completed = subprocess.run(
                argv,
                env=env,
                cwd=str(cwd) if cwd is not None else None,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return _error(
                "timeout",
                f"mn {verb} exceeded {timeout} seconds",
                verb=verb,
                args=normalized_args,
                stderr=_decode(exc.stderr),
            )
        except OSError as exc:
            return _error(
                "spawn_failed",
                f"cannot start mn: {type(exc).__name__}",
                verb=verb,
                args=normalized_args,
            )

    if completed.returncode != 0:
        return _error(
            "nonzero_exit",
            f"mn {verb} exited with status {completed.returncode}",
            verb=verb,
            args=normalized_args,
            returncode=completed.returncode,
            stderr=completed.stderr,
        )

    try:
        envelope = _parse_envelope(completed.stdout)
    except ValueError as exc:
        return _error(
            "invalid_json",
            str(exc),
            verb=verb,
            args=normalized_args,
            returncode=completed.returncode,
            stderr=completed.stderr,
        )

    missing = [key for key in ("ok", "command", "version", "data") if key not in envelope]
    if missing:
        return _error(
            "invalid_envelope",
            "missing envelope fields: " + ", ".join(missing),
            verb=verb,
            args=normalized_args,
            returncode=completed.returncode,
        )

    result = dict(envelope)
    result["adapter"] = "mn-json"
    result["observed_at"] = _utc_now()
    result["command_argv"] = ["mn", verb, *normalized_args, "--json"]
    result["data_sha256"] = hashlib.sha256(
        json.dumps(result["data"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    return result


if __name__ == "__main__":
    print(f"{SCHEMA}: self-check")
    print(redact("OPENAI_API_KEY=example-not-a-real-key"))
    try:
        print(f"binary: {discover_bin()}")
    except FileNotFoundError as exc:
        print(f"binary: unavailable ({exc})")
