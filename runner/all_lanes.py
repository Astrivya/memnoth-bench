"""One CLI to dispatch the existing derived benchmark lanes.

Derived diagnostics only: this tool never produces an official score, asserts
no rank, and makes no win claim. It reuses the EXISTING runners as subprocesses
and does not reimplement any recall, reader, judge, or scoring logic.

Keys are supplied by environment path only. ``MEMNOTH_KEYS_FILE`` names the
local key file; key values are never accepted as arguments, never printed,
never logged, and never written to a summary. Only the env var *name* is shown
in dry-run output (its path value is redacted). Every emitted envelope is
stamped official=False / diagnostic="derived" / score_asserted=False; counts
are passed through with their own denominators and are never aggregated into
a score or a comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Single source of truth for the key-file name (env override or local default).
from runner.kilo_pool import DEFAULT_KEYS_PATH  # noqa: E402

VERSION = "0.1.0"
SCHEMA = "memnoth-bench-all-lanes/1"
NOTE = (
    "INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE. Derived lanes dispatched "
    "through their own runners; no rank, no win claim, no aggregate score. "
    "Keys by env path only; no key values in args, env logs, or artifacts."
)

LANES = ("locomo", "beam-1m", "beam-10m")
SCRIPTS: dict[str, str] = {
    "locomo": "locomo_run.py",
    "beam-1m": "beam_run.py",
    "beam-10m": "beam_run.py",
}
BEAM_TIERS: dict[str, str] = {"beam-1m": "1m", "beam-10m": "10m"}

DATA_ROOT = Path(r"D:\memnoth-bench-data")
DEFAULT_CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
DEFAULT_RESULTS_DIR = DATA_ROOT / "results"
# beam_run.py has no --results flag; its RESULTS_DIR is fixed to DATA_ROOT/results.
BEAM_RESULTS_DIR = DATA_ROOT / "results"
DRIVE = "D:\\"
MIN_FREE_GB = 100.0          # D: must have MORE than this much free space
DATA_HARD_CAP_GB = 80.0      # D:\memnoth-bench-data must stay below this
GATE_CEILING = 16            # matches locomo_reader.MEMNOTH_MODEL_GATE ceiling
VISIBLE_ENV = (
    "MEMNOTH_KEYS_FILE",
    "MEMNOTH_MODEL_GATE",
    "MEMNOTH_BENCH_LANE",
    "MEMNOTH_BEAM_TIER",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def directory_bytes(path: Path) -> int:
    """Sum file sizes under ``path``; unreadable entries are skipped, not guessed."""
    total = 0
    if not path.is_dir():
        return 0
    for folder, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(folder) / name).stat().st_size
            except OSError:
                continue
    return total


def check_space() -> tuple[bool, dict[str, Any]]:
    """Preflight the D: hard limits before any derived lane is launched."""
    gib = float(1024 ** 3)
    report: dict[str, Any] = {
        "drive": DRIVE,
        "free_gb_required_gt": MIN_FREE_GB,
        "data_root": str(DATA_ROOT),
        "data_cap_gb": DATA_HARD_CAP_GB,
    }
    try:
        free_gb = shutil.disk_usage(DRIVE).free / gib
    except OSError as exc:
        report.update(ok=False, reason=f"disk_usage_unavailable_{type(exc).__name__}")
        return False, report
    report["free_gb"] = round(free_gb, 2)
    if free_gb <= MIN_FREE_GB:
        report.update(ok=False, reason="D: free space must be >100 GB")
        return False, report
    data_gb = directory_bytes(DATA_ROOT) / gib
    report["data_gb"] = round(data_gb, 3)
    if data_gb >= DATA_HARD_CAP_GB:
        report.update(ok=False, reason="D:\\memnoth-bench-data hard cap 80 GB reached")
        return False, report
    report.update(ok=True, reason="within limits")
    return True, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lane", choices=LANES, required=True, help="derived lane to dispatch")
    parser.add_argument(
        "--max-cases", type=int, default=0,
        help="case cap passed to the existing runner (0 = all cases)",
    )
    parser.add_argument(
        "--keys", type=int, default=0,
        help="maximum pooled keys passed to the existing runner (0 = all loaded keys)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="locomo case workers; for beam only the model gate is derived from it",
    )
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the exact commands with key references redacted; launch nothing",
    )
    args = parser.parse_args(argv)
    if args.max_cases < 0 or args.keys < 0 or args.workers < 1:
        parser.error("--max-cases and --keys must be >= 0 and --workers >= 1")
    return args


def build_env(args: argparse.Namespace, lane: str) -> dict[str, str]:
    """Child env: key FILE PATH by env name only, plus a bounded model gate.

    The parent's own environment (including any KILO key values) is inherited
    because the existing pool reads it; this tool never inspects or prints it.
    """
    env = os.environ.copy()
    # Value is a path, never a key value. DEFAULT_KEYS_PATH already honours an
    # externally supplied MEMNOTH_KEYS_FILE; no path is invented here.
    env["MEMNOTH_KEYS_FILE"] = str(DEFAULT_KEYS_PATH)
    gate_raw = (os.environ.get("MEMNOTH_MODEL_GATE") or "").strip()
    try:
        gate = max(1, min(int(gate_raw), GATE_CEILING)) if gate_raw else max(1, min(args.workers, GATE_CEILING))
    except ValueError:
        gate = max(1, min(args.workers, GATE_CEILING))
    env["MEMNOTH_MODEL_GATE"] = str(gate)
    env["MEMNOTH_BENCH_LANE"] = lane
    if lane in BEAM_TIERS:
        env["MEMNOTH_BEAM_TIER"] = BEAM_TIERS[lane]
    return env


def build_command(args: argparse.Namespace, lane: str) -> list[str]:
    """Argument list for the EXISTING runner; no runner logic is duplicated."""
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    cmd = [sys.executable, str(ROOT / "runner" / SCRIPTS[lane])]
    if lane == "locomo":
        cmd += [
            "--max-cases", str(args.max_cases),
            "--keys", str(args.keys),
            "--workers", str(args.workers),
            "--checkpoint-dir", str(checkpoint_dir),
            "--results", str(results_dir),
        ]
    else:
        # beam_run.py takes --tier but no --workers and no --results override.
        cmd += [
            "--tier", BEAM_TIERS[lane],
            "--max-cases", str(args.max_cases),
            "--keys", str(args.keys),
            "--checkpoint-dir", str(checkpoint_dir),
        ]
    return cmd


def render_command(cmd: list[str], env: dict[str, str]) -> str:
    """Exact Windows command line; only the key-file PATH value is withheld."""
    visible: list[str] = []
    for name in VISIBLE_ENV:
        if name not in env:
            continue
        value = "<redacted>" if name == "MEMNOTH_KEYS_FILE" else env[name]
        visible.append(f"{name}={value}")
    visible.append(subprocess.list2cmdline(cmd))
    return " ".join(visible)


def expected_result_path(args: argparse.Namespace, lane: str) -> Path:
    """Where the existing runner writes its JSON, without overriding it."""
    results_dir = Path(args.results_dir).expanduser().resolve()
    if lane == "locomo":
        suffix = str(args.max_cases) if args.max_cases else "all"
        return results_dir / f"locomo_conv0_max{suffix}.json"
    return BEAM_RESULTS_DIR / f"beam_{BEAM_TIERS[lane]}_{args.max_cases}.json"


def read_denominators(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    den = data.get("denominators") if isinstance(data, dict) else None
    return den if isinstance(den, dict) else {}


def render_denominators(den: dict[str, Any], limit: int = 160) -> str:
    text = ", ".join(f"{key}={value}" for key, value in sorted(den.items()))
    return text if text and len(text) <= limit else (text[:limit] + "...") if text else "none"


def run_lane(args: argparse.Namespace, lane: str) -> dict[str, Any]:
    env = build_env(args, lane)
    cmd = build_command(args, lane)
    result_path = expected_result_path(args, lane)
    row: dict[str, Any] = {
        "lane": lane,
        "official": False,
        "diagnostic": "derived",
        "score_asserted": False,
        "status": "dry-run" if args.dry_run else "launched",
        "command_redacted": render_command(cmd, env),
        "env_names": [name for name in VISIBLE_ENV if name in env],
        "result_path": str(result_path),
        "exit_code": None,
        "result_sha256": None,
        "denominators": {},
    }
    print(f"[lane] {lane}  (official=false diagnostic=derived)")
    print("  env names: " + ", ".join(row["env_names"]) + " (path/values withheld)")
    print(f"  cmd: {row['command_redacted']}")
    print(f"  expected_result: {result_path}")
    if args.dry_run:
        print("  dry-run: subprocess NOT launched")
        return row
    completed = subprocess.run(cmd, env=env, cwd=str(ROOT), check=False)
    row["exit_code"] = completed.returncode
    row["status"] = "completed" if completed.returncode == 0 else "nonzero_exit"
    row["result_sha256"] = sha256_file(result_path)
    row["denominators"] = read_denominators(result_path)
    return row


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("=== derived lanes consolidated summary ===")
    print("official=false diagnostic=derived score_asserted=false")
    print("no rank, no win claim, no aggregate score; denominators stay with their lane")
    print(f"{'lane':<10} {'exit':<5} {'sha256[:12]':<14} denominators")
    for row in rows:
        exit_code = "-" if row["exit_code"] is None else str(row["exit_code"])
        digest = (row["result_sha256"] or "missing")[:12]
        print(
            f"{row['lane']:<10} {exit_code:<5} {digest:<14} "
            f"{render_denominators(row['denominators'])}"
        )
    for row in rows:
        print(f"  result[{row['lane']}]: {row['result_path']}")
    print("key source: MEMNOTH_KEYS_FILE (env var name only; path and values withheld)")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    space_ok, space_report = check_space()
    print(
        f"space: D free={space_report.get('free_gb', 'unavailable')} GB "
        f"(need >{MIN_FREE_GB}) data={space_report.get('data_gb', 'unavailable')} GB "
        f"(cap {DATA_HARD_CAP_GB} GB) -> {space_report.get('reason', 'unknown')}"
    )
    if not space_ok:
        if args.dry_run:
            print("space gate blocked; dry-run continues without launching")
        else:
            print("refusing to launch any lane: " + str(space_report.get("reason")))
            return 2

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    if args.dry_run:
        print(f"dry-run: would create {checkpoint_dir} and {results_dir} if absent")
    else:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    rows = [run_lane(args, lane) for lane in (args.lane,)]
    print_summary(rows)

    if not args.dry_run:
        summary = {
            "schema": SCHEMA,
            "version": VERSION,
            "observed_at": utc_now(),
            "official": False,
            "diagnostic": "derived",
            "score_asserted": False,
            "note": NOTE,
            "space": {key: value for key, value in space_report.items()},
            "lanes": rows,
            "limitations": [
                "Each lane is dispatched to its own existing runner; no logic is reimplemented.",
                "beam_run.py has no --results flag; its JSON stays under D:\\memnoth-bench-data\\results.",
                "Key values never appear in args, env logs, or this summary; only the env var name is shown.",
                "Denominators are the runner's own counts, passed through with no score aggregation.",
            ],
        }
        summary_path = results_dir / f"all_lanes_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        write_json(summary_path, summary)
        print(f"summary: {summary_path}")

    return 0 if all(row["exit_code"] in (0, None) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
