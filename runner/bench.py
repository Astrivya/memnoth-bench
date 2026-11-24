"""One CLI for public, reproducible Memnoth benchmark diagnostics.

All output is stamped ``diagnostic='derived'``, ``official=False`` and
"INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE". Counts always travel with their
denominators; no bare percentages and no ranking language are emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import judge as judge_adapter  # noqa: E402
from adapters import mcp_stdio, mn_json  # noqa: E402

VERSION = "0.1.0"
NOTE = "INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE"
SCHEMA_SMOKE = "memnoth-bench-smoke/1"
SCHEMA_MANIFEST = "memnoth-bench-manifest/1"
SCHEMA_RESULT = "memnoth-bench-result/1"
SCHEMA_VERIFY = "memnoth-bench-verify/1"
SCHEMA_BLOCKED_RUN = "memnoth-bench-run-blocked/1"

# Public CLI subcommands behind the short lane names used by `smoke`.
SMOKE_LANES: dict[str, tuple[str, list[str]]] = {
    "repo-qa": ("bench", ["repo-qa-verify"]),
    "trs": ("bench", ["trs-verify"]),
    "lme": ("bench", ["lme-verify"]),
    "kg": ("bench", ["kg-verify"]),
    "mcp": ("bench", ["mcp-verify"]),
}

MODEL_LANES: tuple[str, ...] = (
    "memoryagentbench-mh",
    "longmemeval-temporal",
    "locomo-small",
)
ALL_RUN_LANES: tuple[str, ...] = (*MODEL_LANES, "mcp-stdio")

_PERCENT = re.compile(r"(?<![\w])(\d+(?:\.\d+)?)\s*%")


def resolve_cli_root(override: Path | str | None = None) -> Path:
    """Where the public CLI's committed ``ci/`` fixtures live.

    ``MN_CLI_ROOT`` wins, then an explicit argument, then the sibling ``CLI/``
    checkout. No absolute path is baked into this runner.
    """
    from_env = os.environ.get("MN_CLI_ROOT")
    value = from_env if from_env else override
    if value:
        return Path(value).expanduser()
    return ROOT.parent / "CLI"


def utc_now() -> str:
    """Real wall-clock UTC ISO-8601 timestamp. Never hard-coded."""
    return datetime.now(timezone.utc).isoformat()


def base_envelope(schema: str, **extra: Any) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "schema": schema,
        "version": VERSION,
        "observed_at": utc_now(),
        "diagnostic": "derived",
        "official": False,
        "note": NOTE,
    }
    envelope.update(extra)
    return envelope


def no_bare_percentages(value: Any) -> Any:
    """Turn ``42%`` strings into ``42 per 100``; recurse through JSON trees."""
    if isinstance(value, str):
        return _PERCENT.sub(lambda match: f"{match.group(1)} per 100", value)
    if isinstance(value, dict):
        return {key: no_bare_percentages(item) for key, item in value.items()}
    if isinstance(value, list):
        return [no_bare_percentages(item) for item in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_inputs(root: Path = ROOT) -> dict[str, Any]:
    """Pinned CLI lock and fixture slice, each with a real file hash."""
    lock_path = root / "targets" / "cli.lock"
    pins_path = root / "fixtures" / "pinned-ids.json"
    lock = load_json(lock_path)
    pins = load_json(pins_path)
    inputs = [
        {
            "path": "targets/cli.lock",
            "sha256": sha256_file(lock_path),
            "bytes": lock_path.stat().st_size,
        },
        {
            "path": "fixtures/pinned-ids.json",
            "sha256": sha256_file(pins_path),
            "bytes": pins_path.stat().st_size,
        },
    ]
    return {"cli_lock": lock, "pins": pins, "inputs": inputs}


def denominators_for(lane: str, pins: dict[str, Any] | None) -> dict[str, Any]:
    """Counts with an explicit denominator; never a bare rate."""
    pins = pins or {}
    if lane == "memoryagentbench-mh":
        total = pins.get("memoryagentbench", {}).get("fact_consolidation_mh")
    elif lane == "longmemeval-temporal":
        total = pins.get("longmemeval", {}).get("count")
    elif lane == "locomo-small":
        total = pins.get("locomo", {}).get("questions")
    else:
        total = None
    if total is None:
        return {"steps_total": 2, "steps_completed": 0}
    return {
        "cases_total": total,
        "cases_attempted": 0,
        "cases_scored": 0,
        "official_split": bool(
            pins.get(lane.split("-")[0], {}).get("official_split", False)
        ),
    }


def estimated_spend(
    lane: str, max_cases: int | None, pins: dict[str, Any] | None
) -> dict[str, Any]:
    """Call-count estimate only. No dollar amount is invented for BYOK lanes."""
    if lane == "mcp-stdio":
        return {
            "currency": "USD",
            "amount_usd": 0.0,
            "amount_asserted": True,
            "basis": "model-free stdio wire probe; this runner makes no API call",
            "calls": {"generation_max": 0, "judge_max": 0},
            "key_names_required": [],
        }
    pins = pins or {}
    if lane == "memoryagentbench-mh":
        total = pins.get("memoryagentbench", {}).get("fact_consolidation_mh")
    elif lane == "longmemeval-temporal":
        total = pins.get("longmemeval", {}).get("count")
    else:
        total = pins.get("locomo", {}).get("questions")
    count = max_cases if max_cases is not None else total
    return {
        "currency": "USD",
        "amount_usd": None,
        "amount_asserted": False,
        "basis": (
            "BYOK: your provider bills generation and OPENAI_API_KEY bills judging. "
            "No price table is embedded here; no amount is asserted."
        ),
        "calls": {
            "generation_max": count if isinstance(count, int) else None,
            "judge_max": count if isinstance(count, int) else None,
        },
        "key_names_required": ["KILO_API_KEY", "OPENAI_API_KEY"],
    }


def print_estimate(lane: str, estimate: dict[str, Any]) -> None:
    amount = estimate["amount_usd"]
    rendered = "not asserted" if amount is None else str(amount)
    print(f"lane: {lane}")
    print(f"estimated spend: {rendered} {estimate['currency']}")
    print(f"  basis: {estimate['basis']}")
    print(f"  max calls: {json.dumps(estimate['calls'], sort_keys=True)}")


def extract_counts(data: Any) -> dict[str, Any]:
    """Only observed integer counters from a CLI envelope, with their keys.

    The public verifiers name their denominators differently: ``queries`` for
    repo-qa, ``cases`` for trs/lme/kg, and ``summary.gates_total`` for mcp.
    They are normalized here to explicit ``cases_total`` / ``gates_total``
    counts rather than being replaced by a rate.
    """
    wanted = (
        "cases",
        "queries",
        "symbols",
        "denominator",
        "cases_total",
        "cases_passed",
        "cases_failed",
        "gates_total",
        "gates_passed",
        "gates_failed",
        "total",
        "passed",
        "failed",
    )
    if not isinstance(data, dict):
        return {"envelopes_total": 1, "envelopes_observed": 1}
    sources = [data]
    for key in ("summary", "metrics"):
        if isinstance(data.get(key), dict):
            sources.append(data[key])
    counts: dict[str, Any] = {}
    for source in sources:
        for key in wanted:
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                counts[key] = value
    if "cases" in counts:
        counts.setdefault("cases_total", counts["cases"])
    if "queries" in counts:
        counts.setdefault("cases_total", counts["queries"])
    if "denominator" in counts:
        counts.setdefault("gates_total", counts["denominator"])
    if "passed" in counts and "failed" in counts and not any(
        key in counts for key in ("cases_total", "gates_total", "total")
    ):
        counts["cases_total"] = counts["passed"] + counts["failed"]
    if not counts:
        counts = {"envelopes_total": 1, "envelopes_observed": 1}
    return counts


def cmd_smoke(args: argparse.Namespace) -> int:
    """Run the five model-free public CLI lanes and aggregate them honestly."""
    try:
        inputs = load_inputs(ROOT)
    except (OSError, json.JSONDecodeError):
        inputs = None
    bin_path: Path | None = None
    if args.bin:
        bin_path = Path(args.bin).expanduser()
    else:
        try:
            bin_path = mn_json.discover_bin()
        except FileNotFoundError:
            bin_path = None

    cli_root = resolve_cli_root(getattr(args, "cli_root", None))
    cwd = cli_root if cli_root.is_dir() else None
    lanes: dict[str, Any] = {}
    for lane, (verb, verb_args) in SMOKE_LANES.items():
        result = mn_json.run_verb(
            verb, [*verb_args, "--max-cases", str(args.max_cases)] if args.max_cases else verb_args,
            bin_path,
            timeout=args.timeout,
            cwd=cwd,
        )
        ok = bool(result.get("ok")) and "error" not in result
        lane_result: dict[str, Any] = {
            "status": "ok" if ok else "fail",
            "command": result.get("command", " ".join(["mn", verb, *verb_args, "--json"])),
            "cli_version": result.get("version"),
            "denominators": (
                extract_counts(result.get("data"))
                if ok
                else {"cases_reported": 0, "error_envelopes": 1}
            ),
            "envelope": no_bare_percentages(result),
        }
        if not ok:
            lane_result["error"] = result.get("error", {"kind": "cli_ok_false", "message": "mn returned ok=false"})
        lanes[lane] = lane_result

    passed = sum(1 for item in lanes.values() if item["status"] == "ok")
    envelope = base_envelope(
        SCHEMA_SMOKE,
        gate="ok" if passed == len(SMOKE_LANES) else "fail",
        scope="model-free public CLI verifiers; no keys; no network; no spend",
        cli_lock=inputs["cli_lock"] if inputs else None,
        lanes=lanes,
        denominators={
            "lanes_total": len(SMOKE_LANES),
            "lanes_run": len(lanes),
            "lanes_passed": passed,
            "lanes_failed": len(lanes) - passed,
        },
        limitations=[
            "Derived diagnostics from a local CLI build, not official suite scores.",
            "A failed lane is recorded as a failure, not omitted or scored.",
        ],
    )
    print(json.dumps(no_bare_percentages(envelope), indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if passed == len(SMOKE_LANES) else 1


def unique_run_dir(root: Path, run_date: str, lane: str) -> Path:
    parent = root / "results" / "raw"
    base = parent / f"{run_date}-{lane}"
    if not base.exists():
        return base
    for suffix in range(2, 100):
        candidate = parent / f"{run_date}-{lane}-{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("too many runs with the same date and lane")


def blocked_result(
    *,
    lane: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    denominators: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    result = base_envelope(
        SCHEMA_RESULT,
        lane=lane,
        run_id=manifest["run_id"],
        status="blocked",
        gate="fail",
        reason=reason,
        cases=[],
        denominators=denominators,
        manifest_path=str(manifest_path.relative_to(ROOT)),
        manifest_sha256=sha256_file(manifest_path),
    )
    result["limitation"] = "No score is asserted for an un-runnable lane."
    return result


def mcp_stdio_result(
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    denominators: dict[str, Any],
    cli_root: Path | None = None,
) -> dict[str, Any]:
    """Drive the public stdio wire with our own client, read-only, no model.

    This lane does not shell out to ``mn``: it uses the stdio JSON-RPC client,
    so there is no fixture working directory to redirect here.
    """
    binary_value = os.environ.get("MEMNOTH_MCP_BIN")
    binary: Path | None = Path(binary_value).expanduser() if binary_value else None
    if binary is None:
        root = resolve_cli_root(cli_root)
        for directory in (
            root / "target" / "debug",
            root / "target" / "release",
            root.parent / "CLI" / "target" / "debug",
            root.parent / "CLI" / "target" / "release",
        ):
            for name in ("memnoth-mcp.exe", "memnoth-mcp"):
                candidate = directory / name
                if candidate.is_file():
                    binary = candidate
                    break
            if binary is not None:
                break

    common = {
        "lane": "mcp-stdio",
        "run_id": manifest["run_id"],
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "denominators": denominators,
        "cases": [],
    }
    if binary is None:
        return blocked_result(
            lane="mcp-stdio",
            manifest=manifest,
            manifest_path=manifest_path,
            denominators=denominators,
            reason="MCP server binary unavailable; set MEMNOTH_MCP_BIN by name in env",
        )

    steps_completed = 0
    try:
        with tempfile.TemporaryDirectory(prefix="mn-bench-mcp-") as temporary:
            with mcp_stdio.Client(binary, Path(temporary) / "vault.db") as client:
                handshake = client.handshake()
                steps_completed += 1
                tools = client.tools_list()
                steps_completed += 1
    except mcp_stdio.MCPError as exc:
        result = base_envelope(SCHEMA_RESULT, **common)
        result.update(
            {
                "status": "fail",
                "gate": "fail",
                "error": exc.envelope(),
                "steps_completed": steps_completed,
                "denominators": {"steps_total": 2, "steps_completed": steps_completed},
            }
        )
        return result

    result = base_envelope(SCHEMA_RESULT, **common)
    result.update(
        {
            "status": "pass",
            "gate": "pass",
            "steps_completed": steps_completed,
            "denominators": {"steps_total": 2, "steps_completed": steps_completed},
            "tools_count": len(tools),
            "tools_sample": [item.get("name") for item in tools[:10] if isinstance(item.get("name"), str)],
            "handshake_result": no_bare_percentages(handshake),
            "profile": "memory",
            "read_only": True,
        }
    )
    return result


def cmd_run(args: argparse.Namespace) -> int:
    """Write a manifest first, then per-lane raw results. Never hide failures."""
    try:
        inputs = load_inputs(ROOT)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot load pinned inputs: {exc}", file=sys.stderr)
        return 2

    run_date = datetime.now(timezone.utc).date().isoformat()
    any_failed = False
    for lane in args.lane:
        estimate = estimated_spend(lane, args.max_cases, inputs["pins"])
        print_estimate(lane, estimate)
        if lane in MODEL_LANES and not args.confirm_spend:
            any_failed = True
            envelope = base_envelope(
                SCHEMA_BLOCKED_RUN,
                lane=lane,
                status="blocked",
                reason="confirm_spend_required",
                estimated_spend=estimate,
                denominators=denominators_for(lane, inputs["pins"]),
            )
            print(json.dumps(no_bare_percentages(envelope), indent=2, sort_keys=True, ensure_ascii=True))
            continue

        run_dir = unique_run_dir(ROOT, run_date, lane)
        manifest_path = run_dir / "manifest.json"
        expected_result_path = run_dir / f"{lane}.json"
        manifest = base_envelope(
            SCHEMA_MANIFEST,
            lane=lane,
            run_id=run_dir.name,
            run_date=run_date,
            cli_lock=inputs["cli_lock"],
            pins=inputs["pins"],
            inputs=inputs["inputs"],
            parameters={
                "max_cases": args.max_cases,
                "confirm_spend": bool(args.confirm_spend),
            },
            estimated_spend=estimate,
            results=[
                {
                    "path": str(expected_result_path.relative_to(ROOT)),
                    "sha256": None,
                    "status": "pending",
                }
            ],
            write_order="manifest written once, before any result file",
        )
        write_json(manifest_path, manifest)

        denominators = denominators_for(lane, inputs["pins"])
        if lane in MODEL_LANES:
            result = blocked_result(
                lane=lane,
                manifest=manifest,
                manifest_path=manifest_path,
                denominators=denominators,
                reason=(
                    "pinned external harness is not available in this checkout; "
                    "no model calls were made and no score is asserted"
                ),
            )
        else:
            result = mcp_stdio_result(
                manifest=manifest,
                manifest_path=manifest_path,
                denominators=denominators,
                cli_root=resolve_cli_root(args.cli_root),
            )

        result_path = expected_result_path
        write_json(result_path, result)
        # Sidecar keeps the manifest immutable after its first write, so the
        # result's manifest_sha256 stays verifiable byte-for-byte.
        write_json(
            run_dir / "artifacts.json",
            base_envelope(
                SCHEMA_VERIFY,
                lane=lane,
                run_id=manifest["run_id"],
                manifest_path=str(manifest_path.relative_to(ROOT)),
                manifest_sha256=sha256_file(manifest_path),
                artifacts=[
                    {
                        "path": str(result_path.relative_to(ROOT)),
                        "sha256": sha256_file(result_path),
                        "status": result["status"],
                    }
                ],
            ),
        )

        if result["status"] != "pass":
            any_failed = True
        print(
            json.dumps(
                no_bare_percentages(
                    {
                        "lane": lane,
                        "status": result["status"],
                        "manifest": str(manifest_path.relative_to(ROOT)),
                        "result": str(result_path.relative_to(ROOT)),
                        "denominators": result["denominators"],
                    }
                ),
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
            )
        )
    return 1 if any_failed else 0


def _case_hash(case: dict[str, Any]) -> str:
    ignored = {"sha256", "observed_at", "judged_at", "cache"}
    return hashlib.sha256(canonical_json({k: v for k, v in case.items() if k not in ignored}).encode("utf-8")).hexdigest()


def _variant_to_kebab(name: str) -> str:
    """RepoQaVerify -> repo-qa-verify, TrsVerify -> trs-verify."""
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


def _lane_names_from_source(cli_root: Path) -> set[str] | None:
    """Static check: read the public CLI's BenchCommand enum, no execution."""
    path = cli_root / "src" / "bench.rs"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"pub enum BenchCommand\s*\{(.*?)\n\}", text, re.DOTALL)
    if not match:
        return None
    variants = re.findall(r"^\s{4}([A-Z][A-Za-z0-9]*)\s*(?:\(|,|\{)", match.group(1), re.MULTILINE)
    return {_variant_to_kebab(name) for name in variants}


def _lane_names_from_help(bin_path: Path | None, cli_root: Path) -> set[str] | None:
    """Read-only check: run ``mn bench --help``; never run a benchmark lane."""
    if bin_path is None:
        try:
            bin_path = mn_json.discover_bin()
        except FileNotFoundError:
            return None
    if not bin_path.is_file():
        return None
    keys = ("PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE")
    env = {key: os.environ[key] for key in keys if key in os.environ}
    try:
        completed = subprocess.run(
            [str(bin_path), "bench", "--help"],
            env=env,
            cwd=str(cli_root) if cli_root.is_dir() else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return set(re.findall(r"[a-z]+(?:-[a-z]+)+", completed.stdout))


def lane_name_checks(cli_root: Path | None = None) -> list[dict[str, Any]]:
    """Guard the five smoke verbs against the CLI renaming them again.

    Prefers ``mn bench --help`` (read-only, no lane execution); falls back to
    the public CLI's source enum when no binary is available.
    """
    expected = [verb_args[0] for _verb, verb_args in SMOKE_LANES.values()]
    root = cli_root or resolve_cli_root()
    help_names = _lane_names_from_help(None, root)
    source_names = help_names if help_names is not None else _lane_names_from_source(root)
    basis = "mn bench --help" if help_names is not None else "public CLI BenchCommand enum"
    if source_names is None:
        return [
            {
                "name": "smoke_lane_names",
                "ok": False,
                "detail": "cannot check five lane names: no mn binary and no CLI source available",
            }
        ]
    checks: list[dict[str, Any]] = []
    for name in expected:
        checks.append(
            {
                "name": f"lane_name:{name}",
                "ok": name in source_names,
                "detail": f"{name} is an accepted bench subcommand per {basis}",
            }
        )
    return checks


def verify_pair(
    manifest_path: Path,
    result_path: Path,
    root: Path = ROOT,
    *,
    check_lane_names: bool = True,
) -> dict[str, Any]:
    """Structural re-check of hashes and honesty stamps. No re-judging."""
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    manifest: Any = None
    result: Any = None
    try:
        manifest = load_json(manifest_path)
        check("manifest_json", isinstance(manifest, dict), "manifest parses as an object")
    except (OSError, json.JSONDecodeError) as exc:
        check("manifest_json", False, f"cannot read manifest: {type(exc).__name__}")
    try:
        result = load_json(result_path)
        check("result_json", isinstance(result, dict), "result parses as an object")
    except (OSError, json.JSONDecodeError) as exc:
        check("result_json", False, f"cannot read result: {type(exc).__name__}")

    if isinstance(manifest, dict):
        stamp_ok = (
            manifest.get("diagnostic") == "derived"
            and manifest.get("official") is False
            and manifest.get("note") == NOTE
            and isinstance(manifest.get("observed_at"), str)
            and manifest.get("schema") == SCHEMA_MANIFEST
        )
        check("manifest_stamps", stamp_ok, "derived / official:false / note / observed_at / schema")
        for entry in manifest.get("inputs", []):
            if not isinstance(entry, dict) or "path" not in entry:
                check("input_entry", False, "manifest input entry is malformed")
                continue
            path = root / str(entry["path"])
            ok = path.is_file() and sha256_file(path) == entry.get("sha256")
            check(f"input_hash:{entry['path']}", ok, "recomputed SHA-256 matches pinned input")
        recorded = {item.get("path"): item for item in manifest.get("results", []) if isinstance(item, dict)}
        relative = str(result_path.relative_to(root)) if result_path.is_absolute() else str(result_path)
        entry = recorded.get(relative)
        expected_hash = sha256_file(result_path) if result_path.is_file() else None
        check(
            "manifest_records_result",
            entry is not None
            and (entry.get("sha256") is None or entry.get("sha256") == expected_hash),
            "manifest declares this result path before the result was written",
        )
        sidecar = result_path.parent / "artifacts.json"
        if sidecar.is_file():
            artifacts = load_json(sidecar)
            recorded_hash = next(
                (
                    item.get("sha256")
                    for item in artifacts.get("artifacts", [])
                    if isinstance(item, dict) and item.get("path") == relative
                ),
                None,
            ) if isinstance(artifacts, dict) else None
            check(
                "artifact_sidecar_hash",
                recorded_hash == expected_hash and expected_hash is not None,
                "sidecar SHA-256 matches the result bytes",
            )

    if isinstance(result, dict):
        stamp_ok = (
            result.get("diagnostic") == "derived"
            and result.get("official") is False
            and result.get("note") == NOTE
            and isinstance(result.get("observed_at"), str)
            and result.get("schema") == SCHEMA_RESULT
        )
        check("result_stamps", stamp_ok, "derived / official:false / note / observed_at / schema")
        manifest_hash = sha256_file(manifest_path) if manifest_path.is_file() else None
        check(
            "result_manifest_link",
            result.get("manifest_sha256") == manifest_hash and manifest_hash is not None,
            "result points at the exact manifest bytes",
        )
        denominators = result.get("denominators")
        denominators_ok = isinstance(denominators, dict) and all(
            isinstance(value, int) or isinstance(value, bool) for value in denominators.values()
        )
        check("denominators", denominators_ok, "every denominator value is an integer count")
        serialized = canonical_json(no_bare_percentages(result))
        check("no_bare_percentages", "%" not in serialized, "no bare percent sign in the result")
        cases = result.get("cases", [])
        if isinstance(cases, list):
            check("cases_are_objects", all(isinstance(case, dict) for case in cases), "case list is structural")
            for index, case in enumerate(cases):
                if not isinstance(case, dict):
                    continue
                expected = case.get("sha256")
                if expected is not None:
                    check(
                        f"case_hash:{index}",
                        expected == _case_hash(case),
                        "case hash matches its canonical fields",
                    )

    if check_lane_names:
        checks.extend(lane_name_checks())

    passed = sum(1 for item in checks if item["ok"])
    return base_envelope(
        SCHEMA_VERIFY,
        ok=passed == len(checks) and bool(checks),
        checks=checks,
        denominators={"checks_total": len(checks), "checks_passed": passed},
        manifest=str(manifest_path),
        result=str(result_path),
        scope="structural hashes and honesty stamps only; no scoring and no re-judging",
    )


def cmd_verify(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser()
    result_path = (
        Path(args.result).expanduser()
        if args.result
        else manifest_path.parent / f"{load_json(manifest_path).get('lane')}.json"
    )
    report = verify_pair(manifest_path, result_path)
    print(json.dumps(no_bare_percentages(report), indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench.py",
        description="Derived, information-only benchmark runner for public Memnoth surfaces.",
        epilog=(
            "Fixture cwd: the CLI resolves ci/*.json from its own checkout root. "
            "Set MN_CLI_ROOT to that root (default: sibling CLI/), or pass --cli-root. "
            "No absolute path is hardcoded."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke",
        help="run model-free public CLI lanes (no keys, no spend)",
        epilog="Fixture cwd is MN_CLI_ROOT (default: sibling CLI/) or --cli-root.",
    )
    smoke.add_argument("--bin", help="path to the mn binary (default: MN_BIN then discovery)")
    smoke.add_argument(
        "--cli-root",
        default=None,
        help="CLI checkout root holding ci/ fixtures (default: MN_CLI_ROOT or sibling CLI/)",
    )
    smoke.add_argument("--max-cases", type=int, default=None, help="optional per-lane case cap")
    smoke.add_argument("--timeout", type=float, default=60.0, help="subprocess timeout in seconds")
    smoke.set_defaults(handler=cmd_smoke)

    run = subparsers.add_parser(
        "run",
        help="run a pinned lane; model lanes require --confirm-spend",
        epilog=(
            "Fixture cwd is MN_CLI_ROOT (default: sibling CLI/) or --cli-root. "
            "The mcp-stdio lane uses the JSON-RPC client directly and does not shell out to mn."
        ),
    )
    run.add_argument("--lane", action="append", choices=ALL_RUN_LANES, required=True, help="lane name (repeatable)")
    run.add_argument("--confirm-spend", action="store_true", help="confirm BYOK model/judge spend")
    run.add_argument("--max-cases", type=int, default=None, help="cap cases attempted")
    run.add_argument(
        "--cli-root",
        default=None,
        help="CLI checkout root holding ci/ fixtures (default: MN_CLI_ROOT or sibling CLI/)",
    )
    run.set_defaults(handler=cmd_run)

    verify = subparsers.add_parser("verify", help="structurally verify a manifest/result pair")
    verify.add_argument("--manifest", required=True, help="path to manifest.json")
    verify.add_argument("--result", default=None, help="path to result JSON (default: <lane>.json beside manifest)")
    verify.set_defaults(handler=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
