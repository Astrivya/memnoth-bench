"""One CLI for public, reproducible Memnoth benchmark diagnostics.

All output is stamped ``diagnostic='derived'``, ``official=False`` and
"INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE". Counts always travel with their
denominators; no bare percentages and no ranking language are emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
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


# --------------------------------------------------------------------------- #
# Pinned external lanes -- real adapter wiring
# --------------------------------------------------------------------------- #

LANE_MODULES: dict[str, str] = {
    "memoryagentbench-mh": "harnesses.memoryagentbench.adapter",
    "longmemeval-temporal": "harnesses.longmemeval.adapter",
    "locomo-small": "harnesses.locomo.adapter",
}

# Repo-relative paths only; prepare() itself enforces the SHA-256 pin.
DATASET_REL: dict[str, str] = {
    "memoryagentbench-mh": "data/memoryagentbench/Conflict_Resolution-00000-of-00001.parquet",
    "longmemeval-temporal": "data/longmemeval/longmemeval_oracle.json",
    "locomo-small": "data/locomo/locomo_conversation.json",
}

_SERIAL = re.compile(r"serial\s+(\d+)", re.IGNORECASE)


class LaneBlocked(RuntimeError):
    """A pinned lane cannot run; report it, never substitute a score."""


def adapter_for(lane: str):
    return importlib.import_module(LANE_MODULES[lane])


def _strip_percent_literals(value: Any) -> Any:
    """Keep the verify contract: no percent sign survives into a result."""
    if isinstance(value, str):
        return no_bare_percentages(value).replace("%", " percent")
    if isinstance(value, dict):
        return {key: _strip_percent_literals(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_percent_literals(item) for item in value]
    return value


def prepare_for_lane(lane: str, *, dry_run: bool) -> dict[str, Any]:
    """Exercise prepare() without network in dry-run (cached artifact only)."""
    module = adapter_for(lane)
    artifact = DATASET_REL[lane]
    cached = ROOT / artifact
    if dry_run:
        if not cached.is_file():
            return {
                "status": "skipped_dry_run",
                "artifact": artifact,
                "reason": "no cached artifact; --dry-run never uses the network",
            }
        try:
            path = module.prepare()  # cache hit: fetch_pinned re-hashes, no download
        except Exception as exc:
            return {
                "status": "blocked",
                "artifact": artifact,
                "reason": f"cached prepare failed: {type(exc).__name__}",
            }
        return {
            "status": "cached_prepare_ok",
            "artifact": artifact,
            "sha256": sha256_file(path),
        }
    try:
        path = module.prepare()
    except Exception as exc:
        raise LaneBlocked(f"dataset preparation failed: {type(exc).__name__}: {exc}") from None
    return {"status": "prepared", "artifact": artifact, "sha256": sha256_file(path)}


def load_dataset(lane: str) -> Any:
    """Load the SHA-verified artifact with stdlib JSON, or optional pyarrow."""
    path = ROOT / DATASET_REL[lane]
    if not path.is_file():
        raise LaneBlocked(f"verified dataset artifact is missing: {DATASET_REL[lane]}")
    if path.suffix == ".json":
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LaneBlocked(f"cannot read dataset JSON: {type(exc).__name__}") from None
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError:
        raise LaneBlocked(
            "local parquet reader unavailable; install pyarrow locally for the real "
            "MemoryAgentBench lane or use --dry-run (stdlib-only)"
        ) from None
    try:
        return pq.read_table(path).to_pylist()
    except Exception as exc:
        raise LaneBlocked(f"cannot read dataset parquet: {type(exc).__name__}") from None


def _entity_name(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")[:120] or "entity"


_STOPWORDS = {
    "the", "and", "for", "was", "were", "what", "when", "where", "who", "with",
    "that", "this", "from", "have", "had", "has", "did", "does", "how", "why",
    "about", "into", "its", "our", "your", "their", "they", "then", "than",
    "after", "before", "first", "last", "been", "are", "not", "you", "his", "her",
}


def _keyword_query(text: str, limit: int = 8) -> str:
    """Short FTS-friendly query: the public recall path ranks entity names."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", str(text or "").lower())
    return " ".join(token for token in tokens if token not in _STOPWORDS)[:200] or str(text)[:200]


def _slug(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[A-Za-z0-9]+", str(text or ""))[:max_words]
    return "-".join(word.lower() for word in words)[:60]


def _match_known(name: str, known: list[str]) -> str | None:
    """Map a recalled entity name back to one of the seeded pinned IDs."""
    candidate = _entity_name(name)
    for value in known:
        needle = _entity_name(value)
        if needle and (candidate == needle or candidate.startswith(needle + "-") or needle in candidate):
            return value
    return None


def _vault_remember(bin_path: Path | None, cli_root: Path | None, vault: Path, entity: str, text: str) -> bool:
    result = mn_json.run_verb(
        "vault",
        ["remember", _entity_name(entity), text[:2000], "--vault", str(vault)],
        bin_path,
        timeout=30.0,
        cwd=cli_root,
    )
    return bool(result.get("ok")) and "error" not in result


def _vault_remember_stamped(
    bin_path: Path | None, cli_root: Path | None, vault: Path, entity: str, text: str
) -> tuple[bool, Any]:
    """Remember a fact and return the CLI's own recorded_at transaction time."""
    result = mn_json.run_verb(
        "vault",
        ["remember", _entity_name(entity), text[:2000], "--vault", str(vault)],
        bin_path,
        timeout=30.0,
        cwd=cli_root,
    )
    if not result.get("ok") or "error" in result:
        return False, None
    data = result.get("data") or {}
    return True, data.get("recorded_at") or data.get("at")


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split()).casefold()


def _cap(items: list[Any], max_cases: int | None) -> list[Any]:
    """Same slicing rule as the harness limiter: none or 0 means no truncation."""
    if max_cases is None or isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases <= 0:
        return list(items)
    return list(items)[:max_cases]


def _interleave_contexts(
    configurations: list[Any], contexts: list[Any] | None = None
) -> list[Any]:
    """Round-robin configurations across context buckets before any cap.

    select() emits every ordinal of one context before the next, so slicing the
    first N configurations leaves every later context with a zero denominator.
    Preserve each bucket's fixture order, take one per context in fixture order,
    and stop when a bucket is exhausted. Never pads.
    """
    ordered: list[Any] = []
    for value in contexts or []:
        if value not in ordered:
            ordered.append(value)
    buckets: dict[Any, list[Any]] = {}
    for item in configurations:
        key = item.get("context_tokens") if isinstance(item, dict) else None
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(item)
        if key not in ordered:
            ordered.append(key)
    interleaved: list[Any] = []
    for index in range(max((len(bucket) for bucket in buckets.values()), default=0)):
        for key in ordered:
            bucket = buckets.get(key) or []
            if index < len(bucket):
                interleaved.append(bucket[index])
    return interleaved


def selection_for_lane(lane: str, module: Any, max_cases: int | None) -> dict[str, Any]:
    """Pin the lane's subset before capping; never let SH stand in for MH."""
    selection = dict(module.select(max_cases))
    if lane != "memoryagentbench-mh":
        return selection
    all_configurations = (module.select(None) or {}).get("configurations") or []
    mh = [
        item
        for item in all_configurations
        if isinstance(item, dict) and str(item.get("subset") or "").upper() == "MH"
    ]
    interleaved = _interleave_contexts(mh, list(selection.get("context_tokens") or []))
    capped = _cap(interleaved, max_cases)
    selection["configurations"] = capped
    selection["effective_denominator"] = len(capped)
    selection["selected_subset"] = "MH"
    selection["subset_filter_applied_before_cap"] = True
    context_denominators: dict[str, int] = {}
    for item in capped:
        if isinstance(item, dict):
            key = str(item.get("context_tokens"))
            context_denominators[key] = context_denominators.get(key, 0) + 1
    selection["context_denominators"] = context_denominators
    return selection


def _vault_recall(
    bin_path: Path | None, cli_root: Path | None, vault: Path, query: str, limit: int
) -> list[str]:
    result = mn_json.run_verb(
        "vault",
        ["recall", query[:2000], "--limit", str(limit), "--vault", str(vault)],
        bin_path,
        timeout=30.0,
        cwd=cli_root,
    )
    if not result.get("ok") or "error" in result:
        return []
    hits = (result.get("data") or {}).get("hits") or []
    names: list[str] = []
    for hit in hits:
        if isinstance(hit, dict):
            name = hit.get("name") or hit.get("canonical_name")
            if name:
                names.append(str(name))
    return names


def _recall_question(
    bin_path: Path | None, cli_root: Path | None, vault: Path, question: str, limit: int
) -> list[str]:
    """Recall with the question, then progressively shorter FTS keyword forms.

    The public recall verb is a token-AND search; a long natural-language
    question often matches nothing. Shorter real queries are still observations
    from the public surface, not synthesized answers.
    """
    ranked = _vault_recall(bin_path, cli_root, vault, question, limit)
    if ranked:
        return ranked
    terms = [token for token in _keyword_query(question).split() if token]
    queries: list[str] = []
    if terms:
        queries.append(" ".join(terms))
    for count in (3, 2, 1):
        if len(terms) >= count:
            queries.append(" ".join(terms[:count]))
    for term in terms[:6]:
        queries.append(term)
    seen: set[str] = set()
    found: list[str] = []
    for query in queries:
        for name in _vault_recall(bin_path, cli_root, vault, query, limit):
            if name not in seen:
                seen.add(name)
                found.append(name)
        if found:
            break
    return found


def _vault_asof(
    bin_path: Path | None, cli_root: Path | None, vault: Path, entity: str
) -> list[str]:
    result = mn_json.run_verb(
        "vault",
        ["asof", "--name", _entity_name(entity), "--vault", str(vault)],
        bin_path,
        timeout=30.0,
        cwd=cli_root,
    )
    if not result.get("ok") or "error" in result:
        return []
    facts = (result.get("data") or {}).get("facts") or []
    return [str(fact.get("content")) for fact in facts if isinstance(fact, dict)]


def _mab_facts(context: Any) -> list[tuple[int, str]]:
    facts: list[tuple[int, str]] = []
    for line in str(context or "").splitlines():
        match = re.match(r"\s*(\d+)\.\s+(.*\S)\s*$", line)
        if match:
            facts.append((int(match.group(1)), match.group(2)))
    return facts


def _answer_in_fact(answer: Any, fact_text: str) -> bool:
    needle = " ".join(str(answer or "").split()).casefold()
    return bool(needle) and needle in " ".join(fact_text.split()).casefold()


def _norm_fact_text(value: Any) -> str:
    """Whitespace-normalized fact text used for template comparison."""
    return " ".join(str(value or "").split())


def _template_pairs(
    facts: list[tuple[int, str]], gold_answers: list[Any]
) -> list[tuple[int, int]]:
    """Find (true serial, rewrite serial) pairs by shared subject template.

    A fact containing a gold answer is a rewrite candidate: split its normalized
    text around that answer into (prefix, suffix). Another fact with the same
    prefix/suffix but a different middle value is the superseded true fact; the
    rewrite middle must equal the gold answer. The serials are pinned fact
    indices, not a temporal ordering -- a rewrite index can be lower than its
    true counterpart -- so the index is never used to decide which is newer.
    """
    pairs: list[tuple[int, int]] = []
    for rewrite_serial, rewrite_text in facts:
        hay = _norm_fact_text(rewrite_text)
        low = hay.casefold()
        for answer in gold_answers:
            needle = _norm_fact_text(answer).casefold()
            if not needle:
                continue
            start = 0
            while True:
                position = low.find(needle, start)
                if position < 0:
                    break
                prefix, suffix = hay[:position], hay[position + len(needle):]
                prefix_fold, suffix_fold = prefix.casefold(), suffix.casefold()
                for true_serial, true_text in facts:
                    if true_serial == rewrite_serial:
                        continue
                    other = _norm_fact_text(true_text)
                    other_fold = other.casefold()
                    if not (
                        other_fold.startswith(prefix_fold)
                        and other_fold.endswith(suffix_fold)
                    ):
                        continue
                    if len(other) <= len(prefix) + len(suffix):
                        continue
                    middle = (
                        other[len(prefix): len(other) - len(suffix)]
                        if suffix
                        else other[len(prefix):]
                    )
                    if middle.strip() and middle.casefold() != needle:
                        pairs.append((true_serial, rewrite_serial))
                start = position + 1
    return pairs


def _build_mab_rows(
    selection: dict[str, Any],
    dataset: Any,
    bin_path: Path | None,
    cli_root: Path | None,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Drive the vault with the pinned true/rewrite pair for each configuration."""
    by_subset: dict[str, list[dict[str, Any]]] = {"SH": [], "MH": []}
    for record in dataset if isinstance(dataset, list) else []:
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        source = str(record.get("source") or metadata.get("source") or "").lower()
        for subset in by_subset:
            if f"factconsolidation_{subset.lower()}_" in source:
                by_subset[subset].append(record)
    for records in by_subset.values():
        records.sort(key=lambda item: str((item.get("metadata") or {}).get("source")))

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    attempted = 0
    counters = {"SH": 0, "MH": 0}
    for configuration in selection.get("configurations", []):
        if not isinstance(configuration, dict):
            continue
        attempted += 1
        subset = str(configuration.get("subset") or "").upper()
        expected_subset = str(selection.get("selected_subset") or "").upper()
        if expected_subset and subset != expected_subset:
            failures.append(f"wrong subset {subset}; lane requires {expected_subset}")
            continue
        candidates = by_subset.get(subset, [])
        if not candidates:
            failures.append(f"{subset}: no pinned dataset row")
            continue
        position = counters[subset] % len(candidates)
        counters[subset] = position + 1
        record = candidates[position]
        facts = _mab_facts(record.get("context"))
        questions = record.get("questions") or []
        answers = record.get("answers") or []
        if not facts or not questions:
            failures.append(f"{subset}: no facts or questions in pinned row")
            continue
        ordinal = int(configuration.get("ordinal") or 1)
        question_index = (ordinal - 1) % len(questions)
        gold_answers = answers[question_index] if question_index < len(answers) else []
        if not isinstance(gold_answers, list):
            gold_answers = [gold_answers]
        pairs = _template_pairs(facts, gold_answers)
        if not pairs:
            failures.append(f"{subset}: no true/rewrite fact pair for pinned question")
            continue
        # The index is only a deterministic tiebreak between candidate pairs;
        # older/newer are assigned by role (true fact vs rewrite fact), never by
        # comparing the two numeric fact indices.
        older_serial, newer_serial = sorted(pairs, key=lambda pair: (pair[1], pair[0]))[-1]
        fact_by_serial = dict(facts)
        older_text = fact_by_serial[older_serial]
        newer_text = fact_by_serial[newer_serial]
        if _SERIAL.search(older_text) or _SERIAL.search(newer_text):
            failures.append(f"{subset}: pinned fact text already carries a serial marker")
            continue
        # The serial is metadata only. It is never written into the recallable
        # answer text: the system must pick the current fact from its own
        # version/ordering signal (asof), and we map that text back to the
        # pinned fact index afterwards.
        text_to_serial = {_norm(older_text): older_serial, _norm(newer_text): newer_serial}
        case_id = f"fc-{subset.lower()}-{ordinal}-{question_index + 1}"
        with tempfile.TemporaryDirectory(prefix="mn-bench-mab-") as temporary:
            vault = Path(temporary) / "vault.db"
            older_ok, older_time = _vault_remember_stamped(
                bin_path, cli_root, vault, case_id, older_text
            )
            time.sleep(0.01)  # distinct valid_from for the two observations
            newer_ok, newer_time = _vault_remember_stamped(
                bin_path, cli_root, vault, case_id, newer_text
            )
            observed = _vault_asof(bin_path, cli_root, vault, case_id) if older_ok and newer_ok else []
        if not observed:
            failures.append(f"{case_id}: no recall observation")
            continue
        current = observed[-1]  # asof's own current fact, ordered by valid_from
        predicted = text_to_serial.get(_norm(current))
        if predicted is None:
            failures.append(f"{case_id}: recalled text did not map to a pinned fact")
            continue
        rows.append(
            {
                "case_id": case_id,
                "subset": subset,
                "context_tokens": int(configuration.get("context_tokens") or 4096),
                "update_type": "rewrite" if older_serial != newer_serial else "true",
                "gold_serial": str(newer_serial),
                "pred_serial": str(predicted),
                "gold_answer": " ".join(str(item) for item in gold_answers)[:300],
                "pred_answer": current[:300],
                "gold_derivation": "largest pinned fact index containing a gold answer",
                "pred_derivation": "system asof ordering mapped back to pinned fact text; no serial in fact text",
                "serial_signal": {
                    "kind": "transaction_time_valid_from",
                    "older_recorded_at": older_time,
                    "newer_recorded_at": newer_time,
                    "serial_embedded_in_fact_text": False,
                },
            }
        )
    return rows, attempted, failures


def _build_lme_rows(
    selection: dict[str, Any],
    dataset: Any,
    bin_path: Path | None,
    cli_root: Path | None,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    module = adapter_for("longmemeval-temporal")
    question_type = getattr(module, "QUESTION_TYPE_FILTER", "temporal-reasoning")
    records = [
        item for item in dataset if isinstance(item, dict) and item.get("question_type") == question_type
    ]
    records.sort(key=lambda item: str(item.get("question_id")))
    wanted_ids = [str(value) for value in (selection.get("ids") or [])]
    if wanted_ids:
        by_id = {str(item.get("question_id")): item for item in records}
        selected = [by_id[value] for value in wanted_ids if value in by_id]
    else:
        selected = records[: int(selection.get("effective_denominator") or 0)]

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    attempted = 0
    for record in selected:
        attempted += 1
        question_id = str(record.get("question_id") or "")
        session_ids = [
            str(value) for value in (record.get("haystack_session_ids") or []) if value is not None
        ]
        sessions = record.get("haystack_sessions") or []
        if not question_id or not session_ids:
            failures.append(f"{question_id}: missing question or session IDs")
            continue
        with tempfile.TemporaryDirectory(prefix="mn-bench-lme-") as temporary:
            vault = Path(temporary) / "vault.db"
            writes_ok = True
            for index, session_id in enumerate(session_ids):
                turns = sessions[index] if index < len(sessions) else []
                text = " ".join(
                    str(turn.get("content") or "")
                    for turn in turns
                    if isinstance(turn, dict) and turn.get("role") == "user"
                )
                if not text.strip():
                    text = " ".join(str(turn.get("content") or "") for turn in turns if isinstance(turn, dict))
                entity = f"{session_id}-{_slug(text)}"
                writes_ok = _vault_remember(bin_path, cli_root, vault, entity, text or session_id) and writes_ok
            ranked = _recall_question(bin_path, cli_root, vault, str(record.get("question") or ""), 10) if writes_ok else []
        matched: list[str] = []
        for name in ranked:
            session_id = _match_known(name, session_ids)
            if session_id is not None and session_id not in matched:
                matched.append(session_id)
        ranked = matched
        if not ranked:
            failures.append(f"{question_id}: no recall observation")
            continue
        rows.append(
            {
                "question_id": question_id,
                "question_type": question_type,
                "answer_session_ids": [str(value) for value in (record.get("answer_session_ids") or [])],
                "retrieved_session_ids": ranked,
                "answer": str(record.get("answer") or "")[:300],
            }
        )
    return rows, attempted, failures


def _build_locomo_rows(
    selection: dict[str, Any],
    dataset: Any,
    bin_path: Path | None,
    cli_root: Path | None,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    conversations = [item for item in dataset if isinstance(item, dict)] if isinstance(dataset, list) else []
    if not conversations:
        return [], 0, ["no pinned conversation"]
    record = conversations[0]
    conversation = record.get("conversation") if isinstance(record.get("conversation"), dict) else record
    sessions: dict[str, list[Any]] = {
        key: value
        for key, value in conversation.items()
        if key.startswith("session_") and isinstance(value, list)
    }
    questions = [item for item in (record.get("qa") or []) if isinstance(item, dict)]
    wanted = int(selection.get("effective_denominator") or 0)
    selected = questions[:wanted]

    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    attempted = 0
    with tempfile.TemporaryDirectory(prefix="mn-bench-locomo-") as temporary:
        vault = Path(temporary) / "vault.db"
        writes_ok = True
        entity_by_session: dict[str, str] = {}
        for session_id, turns in sessions.items():
            text = " ".join(
                f"{turn.get('speaker') or ''}: {turn.get('text') or ''}"
                for turn in turns
                if isinstance(turn, dict)
            )
            entity = f"{session_id}-{_slug(text)}"
            entity_by_session[session_id] = entity
            writes_ok = _vault_remember(bin_path, cli_root, vault, entity, text or session_id) and writes_ok
        for index, question in enumerate(selected):
            attempted += 1
            question_id = f"locomo-1-{index + 1}"
            if not writes_ok:
                failures.append(f"{question_id}: vault seed failed")
                continue
            hits = _recall_question(bin_path, cli_root, vault, str(question.get("question") or ""), 5)
            ranked: list[str] = []
            for name in hits:
                session_id = _match_known(name, list(sessions))
                if session_id is not None and session_id not in ranked:
                    ranked.append(session_id)
            if not ranked:
                failures.append(f"{question_id}: no recall observation")
                continue
            facts = _vault_asof(bin_path, cli_root, vault, entity_by_session[ranked[0]])
            if not facts or not str(facts[-1]).strip():
                failures.append(f"{question_id}: no recalled content")
                continue
            rows.append(
                {
                    "question_id": question_id,
                    "gold_answer": str(question.get("answer") or "")[:300],
                    "pred_answer": facts[-1][:500],
                    "answer_source": "vault-recall (no generation model)",
                    "retrieved_session_ids": ranked,
                }
            )
    return rows, attempted, failures


ROW_BUILDERS = {
    "memoryagentbench-mh": _build_mab_rows,
    "longmemeval-temporal": _build_lme_rows,
    "locomo-small": _build_locomo_rows,
}


def _synthetic_rows(lane: str, selection: dict[str, Any]) -> list[dict[str, Any]]:
    """Clearly labelled synthetic rows for offline wiring checks only."""
    if lane == "memoryagentbench-mh":
        configurations = [item for item in selection.get("configurations", []) if isinstance(item, dict)]
        return [
            {
                "case_id": f"dry-fc-{index + 1}",
                "subset": str(item.get("subset") or "MH").upper(),
                "context_tokens": int(item.get("context_tokens") or 4096),
                "update_type": "rewrite",
                "gold_serial": f"S-{index + 1}",
                "pred_serial": f"S-{index + 1}" if index % 2 == 0 else "S-stale",
                "gold_answer": f"synthetic answer {index + 1}",
                "pred_answer": f"synthetic answer {index + 1}",
                "fact_text": "synthetic fact text without any serial marker",
                "serial_embedded_in_fact_text": False,
            }
            for index, item in enumerate(configurations)
        ]
    count = int(selection.get("effective_denominator") or 0)
    if lane == "longmemeval-temporal":
        ids = [str(value) for value in (selection.get("ids") or [])]
        return [
            {
                "question_id": ids[index] if index < len(ids) else f"dry-q-{index + 1}",
                "question_type": "temporal-reasoning",
                "answer_session_ids": [f"s{index + 1}"],
                "retrieved_session_ids": [f"s{index + 1}"] if index % 2 == 0 else ["s-other"],
                "answer": "synthetic answer",
            }
            for index in range(count)
        ]
    return [
        {
            "question_id": f"dry-locomo-{index + 1}",
            "gold_answer": "synthetic answer",
            "pred_answer": "synthetic answer" if index % 2 == 0 else "synthetic other",
        }
        for index in range(count)
    ]


def model_lane_result(
    *,
    lane: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    args: argparse.Namespace,
    bin_path: Path | None,
    cli_root: Path | None,
) -> dict[str, Any]:
    """Run one pinned lane through its adapter and the public CLI surface."""
    module = adapter_for(lane)
    dry_run = bool(args.dry_run)
    common: dict[str, Any] = {
        "lane": lane,
        "run_id": manifest["run_id"],
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
        "mode": "dry_run" if dry_run else "real",
        "cases": [],
        "cases_synthetic": dry_run,
        "judge_fn": None,
    }
    try:
        preparation = prepare_for_lane(lane, dry_run=dry_run)
        selection = selection_for_lane(lane, module, args.max_cases)
    except (LaneBlocked, ValueError, OSError) as exc:
        result = base_envelope(SCHEMA_RESULT, **common)
        result.update(
            {
                "status": "blocked",
                "gate": "fail",
                "reason": mn_json.redact(str(exc)),
                "denominators": denominators_for(lane, manifest.get("pins")),
                "limitation": "No score is asserted for a lane that could not be prepared.",
            }
        )
        return result

    total = int(selection.get("effective_denominator") or 0)
    if total == 0:
        result = base_envelope(SCHEMA_RESULT, **common)
        result.update(
            {
                "status": "blocked",
                "gate": "fail",
                "reason": "selection produced no cases for the requested lane subset",
                "preparation": preparation,
                "selection": {
                    "effective_denominator": 0,
                    "selected_subset": selection.get("selected_subset"),
                },
                "denominators": {
                    "cases_total": 0,
                    "cases_attempted": 0,
                    "cases_scored": 0,
                    "cases_failed": 0,
                    "official_split": False,
                },
                "limitation": "No cases, no score; an empty slice is never substituted with another subset.",
            }
        )
        return result
    if dry_run:
        rows = _synthetic_rows(lane, selection)
        attempted, failures = len(rows), []
    else:
        try:
            dataset = load_dataset(lane)
        except LaneBlocked as exc:
            result = base_envelope(SCHEMA_RESULT, **common)
            result.update(
                {
                    "status": "blocked",
                    "gate": "fail",
                    "reason": str(exc),
                    "preparation": preparation,
                    "selection": {
                        "effective_denominator": total,
                        "planned_denominator": selection.get("planned_denominator"),
                    },
                    "denominators": {
                        "cases_total": total,
                        "cases_attempted": 0,
                        "cases_scored": 0,
                        "cases_failed": 0,
                        "official_split": False,
                    },
                    "limitation": "No dataset read, no score asserted.",
                }
            )
            return result
        rows, attempted, failures = ROW_BUILDERS[lane](selection, dataset, bin_path, cli_root)

    evaluation = module.evaluate(rows, judge_fn=None)
    # Synthetic rows are observations of the wiring, not of the product.
    scored = 0 if dry_run else (len(rows) if not failures else 0)
    case_records = [
        {
            "case_id": str(
                row.get("case_id") or row.get("question_id") or f"case-{index + 1}"
            ),
            "observed": True,
            "synthetic": dry_run,
        }
        for index, row in enumerate(rows)
    ]
    case_records.extend(
        {"case_id": f"failed-{index + 1}", "observed": False, "reason": reason}
        for index, reason in enumerate(failures)
    )

    result = base_envelope(SCHEMA_RESULT, **common)
    result.update(
        {
            "status": "ok" if (dry_run or (attempted > 0 and scored > 0)) else "fail",
            "gate": "ok" if (dry_run or (attempted > 0 and scored > 0)) else "fail",
            "preparation": preparation,
            "selection": {
                "effective_denominator": total,
                "context_denominators": selection.get("context_denominators"),
                "selected_subset": selection.get("selected_subset"),
                "subset_filter_applied_before_cap": bool(
                    selection.get("subset_filter_applied_before_cap", False)
                ),
                "planned_denominator": selection.get("planned_denominator"),
                "official_split": bool(selection.get("official_split", False)),
                "ids_pinned": bool(selection.get("ids_pinned", False)),
                "max_cases": args.max_cases,
            },
            "harness": {
                "module": LANE_MODULES[lane],
                "dataset_url": getattr(module, "DATASET_URL", None),
                "dataset_sha256": getattr(module, "DATASET_SHA256", None),
            },
            "adaptation_note": (
                "Serial-to-transaction-time mapping is a derived diagnostic, not the "
                "paper's native SubEM @6K/32K; figures are context only."
                if lane == "memoryagentbench-mh"
                else None
            ),
            "cases": case_records,
            "synthetic_rows": _strip_percent_literals(rows) if dry_run else [],
            "observation_rows": _strip_percent_literals(rows) if not dry_run else [],
            "denominators": {
                "cases_total": total,
                "cases_attempted": attempted,
                "cases_scored": scored,
                "cases_failed": len(failures),
                "evaluated_rows": int(evaluation.get("denominator") or 0),
                "official_split": False,
            },
            "evaluation": _strip_percent_literals(evaluation),
            "score_asserted": bool(scored > 0 and not dry_run),
        }
    )
    if not dry_run and scored > 0 and attempted == scored:
        result["score"] = evaluation.get("correct")
        result["score_denominator"] = evaluation.get("denominator")
    if dry_run:
        result["limitation"] = (
            "Synthetic rows exercise prepare/select/evaluate wiring only. No product "
            "observation, no dataset bytes, no keys, no spend; not a score."
        )
    elif failures:
        result["limitation"] = (
            f"{len(failures)} of {attempted} attempted cases produced no real "
            "observation; no score is asserted."
        )
        result["failure_reasons"] = failures[:20]
    return result


def cmd_run(args: argparse.Namespace) -> int:
    """Write a manifest first, then per-lane raw results. Never hide failures."""
    try:
        inputs = load_inputs(ROOT)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot load pinned inputs: {exc}", file=sys.stderr)
        return 2

    bin_path: Path | None = None
    if getattr(args, "bin", None):
        bin_path = Path(args.bin).expanduser()
    else:
        try:
            bin_path = mn_json.discover_bin()
        except FileNotFoundError:
            bin_path = None
    configured_root = resolve_cli_root(getattr(args, "cli_root", None))
    cli_root = configured_root if configured_root.is_dir() else None

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
            result = model_lane_result(
                lane=lane,
                manifest=manifest,
                manifest_path=manifest_path,
                args=args,
                bin_path=bin_path,
                cli_root=cli_root,
            )
        else:
            result = mcp_stdio_result(
                manifest=manifest,
                manifest_path=manifest_path,
                denominators=denominators,
                cli_root=cli_root,
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

        if result["status"] not in ("ok", "pass"):
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
    run.add_argument("--bin", default=None, help="path to mn (default: MN_BIN then discovery)")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise prepare/select/evaluate with synthetic rows; no network, keys, or spend",
    )
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
