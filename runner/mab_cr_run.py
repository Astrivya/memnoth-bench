"""Standalone, multi-key, checkpointed runner for the MAB CR SubEM lane.

This runner is NEW and does not go through ``runner/bench.py``. It reuses the
existing frozen reader transport (``runner.locomo_reader``), key pool
(``runner.kilo_pool``), and CLI-vault wrappers (``runner.locomo_run``) by import;
none of those files is edited.

HONESTY CONTRACT
----------------
* Official-style SubEM only; no judge is called. ``official: false`` because the
  reader, prompts, seeding and envelope are ours, not the benchmark's.
* ``diagnostic: derived``, ``comparability_tier: C``, ``score_asserted: false``.
  No rank, no win, no comparison to a published row.
* Gold answers are used ONLY for scoring/hashing. They are never seeded into the
  vault, never sent to the reader, and never written to a checkpoint or result.
* Key values are never printed, logged or written; checkpoints carry a gold hash
  prefix only. ``--dry-run`` returns before any key, CLI or model boundary and
  records zeroed call counters as proof.
* Dataset bytes are SHA-256 verified before use and stay outside the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import mn_json  # noqa: E402  (single-source CLI JSON adapter)
from harnesses.mab_cr import adapter  # noqa: E402
from runner import locomo_reader, locomo_run  # noqa: E402
from runner.kilo_pool import KeyPool, load_keys  # noqa: E402

VERSION = "0.1.0"
SCHEMA = "memnoth-bench-mab-cr-run/1"
NOTE = (
    "Official-style MemoryAgentBench CR substring_exact_match on the pinned "
    "Conflict_Resolution parquet. Judge-free, but our reader/prompts/seeding "
    "are not the benchmark's: official=false, comparability_tier=C, "
    "score_asserted=false. No rank or win is implied."
)

DATA_ROOT = adapter.DATA_ROOT
DEFAULT_CHECKPOINT_DIR = DATA_ROOT / "checkpoints" / "mab_cr"
DEFAULT_RESULTS_DIR = DATA_ROOT / "results"
# Optional already-verified copy for offline dry runs; never a pin source.
LOCAL_CACHE = ROOT / "data" / "memoryagentbench" / adapter.DATASET_FILE

MAX_WORKERS_CEILING = 16
MAX_CLI_WRITE_WORKERS = 8
MAX_KEY_RETRIES = 3
RECALL_TOP_K = 10
MAX_EVIDENCE_FACTS = 20
_ENTITY_RE = re.compile(r"mabcr-(\d{6})$")
_SERIAL_LEAK_RE = re.compile(r"^\d{1,6}\.\s+\S")

BASE_URL = locomo_reader.DEFAULT_BASE_URL

# Pinned CR reader prompt. It deliberately contains "oldest-first" and
# "UNSUPPORTED" so locomo_reader._effective_system_prompt uses it verbatim
# instead of appending the LoCoMo date-grounding clause.
READER_PROMPT_CR = (
    "You answer a question using ONLY the provided numbered facts.\n"
    "Facts are listed oldest-first; newer facts have a larger serial number.\n"
    "When two facts conflict, treat the fact with the larger serial number as "
    "the current one.\n"
    "Reason internally, then output ONLY the short final answer value on one "
    "line, with no explanation and no labels. If the value is genuinely absent "
    "from the facts, output UNSUPPORTED."
)


# --------------------------------------------------------------------------- #
# Boundary guard: the dry-run proof
# --------------------------------------------------------------------------- #


class CallGuard:
    """Count and, in dry-run, block every key/vault/model boundary call."""

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = bool(dry_run)
        self.key_loads = 0
        self.pool_builds = 0
        self.model_calls = 0
        self.cli_calls = 0

    def _block(self, label: str) -> None:
        if self.dry_run:
            raise RuntimeError(f"dry-run: {label} call was blocked")

    def keys(self, *args: Any, **kwargs: Any) -> list[str]:
        self.key_loads += 1
        self._block("key load")
        return load_keys(*args, **kwargs)

    def pool(self, keys: list[str]) -> KeyPool:
        self.pool_builds += 1
        self._block("key pool construction")
        return KeyPool(keys)

    def reader(self, *args: Any, **kwargs: Any) -> tuple[str | None, str | None]:
        self.model_calls += 1
        self._block("reader model")
        return locomo_reader.reader_answer(*args, **kwargs)

    def cli(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self.cli_calls += 1
        self._block("CLI vault")
        return fn(*args, **kwargs)

    def evidence(self) -> dict[str, Any]:
        return {
            "key_loads": self.key_loads,
            "key_pool_constructions": self.pool_builds,
            "reader_model_calls": self.model_calls,
            "cli_vault_calls": self.cli_calls,
            "blocked_in_dry_run": self.dry_run,
            "proof": (
                "The dry-run branch returns after dataset preparation and row "
                "building, before any guarded boundary can be reached."
            ),
        }


# --------------------------------------------------------------------------- #
# Small helpers reused from locomo_run (imported, never edited)
# --------------------------------------------------------------------------- #


def write_json(path: Path, value: Any) -> None:
    locomo_run.write_json(path, value)


def sha256_text(value: Any) -> str:
    return locomo_run.sha256_text(value)


def status_from_error(code: str | None) -> int | None:
    return locomo_run.status_from_error(code)


def entity_name(serial: int) -> str:
    return f"mabcr-{int(serial):06d}"


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                case_id = str(record["case_id"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            done[case_id] = record
    return done


# --------------------------------------------------------------------------- #
# Vault seeding and recall (gold is never involved)
# --------------------------------------------------------------------------- #


def seed_facts(
    guard: CallGuard,
    bin_path: Path | None,
    cli_root: Path | None,
    vault: Path,
    facts: list[dict[str, Any]],
) -> dict[str, Any]:
    """Plant each numbered fact verbatim under a serial-named entity."""
    ok = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=MAX_CLI_WRITE_WORKERS) as executor:
        futures = [
            executor.submit(
                guard.cli,
                locomo_run.vault_remember,
                bin_path,
                cli_root,
                vault,
                entity_name(fact["serial"]),
                str(fact["text"]),
            )
            for fact in facts
        ]
        for future in as_completed(futures):
            try:
                ok += int(bool(future.result()))
            except Exception:
                failed += 1
    return {"facts": len(facts), "seeded": ok, "failed": failed}


def recall_evidence(
    guard: CallGuard,
    bin_path: Path | None,
    cli_root: Path | None,
    vault: Path,
    question: str,
) -> list[str]:
    """Recall by question; rebuild serial-prefixed evidence, oldest first."""
    names = guard.cli(locomo_run.cli_recall_sweep, bin_path, cli_root, vault, question, RECALL_TOP_K)
    collected: dict[int, str] = {}
    for name in names:
        match = _ENTITY_RE.search(str(name))
        if not match:
            continue
        serial = int(match.group(1))
        contents = guard.cli(locomo_run.vault_asof, bin_path, cli_root, vault, str(name))
        for item in contents:
            text = str(item).strip()
            if not text or _SERIAL_LEAK_RE.match(text):
                continue
            collected[serial] = f"{serial}. {text}"
            break
    ordered = sorted(collected.items(), key=lambda pair: pair[0])
    return [text for _, text in ordered[:MAX_EVIDENCE_FACTS]]


# --------------------------------------------------------------------------- #
# Model call + one case chain
# --------------------------------------------------------------------------- #


def reader_with_pool(
    guard: CallGuard, pool: KeyPool, question: str, facts: list[str], reader_model: str
) -> tuple[str | None, str | None]:
    last_error = "reader_unavailable"
    for _ in range(MAX_KEY_RETRIES):
        key = pool.get()
        if key is None:
            return None, "no_key_available"
        answer, error = guard.reader(
            question,
            facts,
            key,
            reader_model=reader_model,
            base_url=BASE_URL,
            system_prompt=READER_PROMPT_CR,
        )
        if error is None and isinstance(answer, str) and answer.strip():
            pool.release(key, True, 200)
            return answer.strip()[:300], None
        pool.release(key, False, status_from_error(error))
        last_error = str(error or "reader_unknown_error")
    return None, last_error


def process_case(
    *,
    case: dict[str, Any],
    bin_path: Path | None,
    cli_root: Path | None,
    vault: Path,
    pool: KeyPool,
    guard: CallGuard,
    reader_model: str,
) -> dict[str, Any]:
    golds = case["answers"]

    def record(pred: str | None, error: str | None) -> dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "subset": case["subset"],
            "context": case["context"],
            "qa_pair_id": case.get("qa_pair_id"),
            "pred": pred,
            "gold_hash_prefix": hashlib.sha256(
                json.dumps(golds, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()[:12],
            "subem": int(adapter.score_subem(pred, golds)) if pred is not None else 0,
            "reader_model": reader_model,
            "error": error,
        }

    try:
        evidence = recall_evidence(guard, bin_path, cli_root, vault, case["question"])
        if not evidence:
            return record(None, "no_recall_observation")
        pred, error = reader_with_pool(guard, pool, case["question"], evidence, reader_model)
        if pred is None:
            return record(None, error or "reader_failed")
        return record(pred, None)
    except Exception as exc:  # a worker error is never a fabricated hit
        return record(None, f"case_error_{type(exc).__name__}")


# --------------------------------------------------------------------------- #
# Result envelopes
# --------------------------------------------------------------------------- #


def base_envelope(*, status: str, dataset_path: Path, plan: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "observed_at": locomo_run.utc_now(),
        "official": False,
        "diagnostic": "derived",
        "comparability_tier": "C",
        "score_asserted": False,
        "note": NOTE,
        "status": status,
        "lane": adapter.LANE,
        "dataset": {
            "local_name": dataset_path.name,
            "sha256": adapter.DATASET_SHA256,
            "bytes": adapter.DATASET_BYTES,
            "url": adapter.DATASET_URL,
            "revision": adapter.HF_REVISION,
            "verified": True,
            "path": str(dataset_path),
        },
        "selection": plan["selection"],
        "plan": {
            "selected_cases": plan["selected_cases"],
            "source_rows_matched": plan["source_rows_matched"],
            "sources": plan["sources"],
            "expected_full_denominator": plan["expected_full_denominator"],
        },
        "reader": {
            "model": args.reader_model,
            "provider": "kilo",
            "base_url": BASE_URL,
            "temperature": 0,
            "system_prompt": READER_PROMPT_CR,
            "reader_transport": "runner.locomo_reader.reader_answer",
            "recall_top_k": RECALL_TOP_K,
            "max_evidence_facts": MAX_EVIDENCE_FACTS,
            "note": "Gateway does not attest model identity; identity is from call arguments.",
        },
        "metric": {
            "name": adapter.METRIC,
            "parse_path": adapter.PARSE_PATH,
            "upstream_commit": adapter.UPSTREAM_COMMIT,
            "judge_free": True,
            "judge_calls": 0,
        },
        "workers": max(1, min(args.workers, MAX_WORKERS_CEILING)),
        "max_key_retries": MAX_KEY_RETRIES,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=("all", "sh", "mh"), default="all")
    parser.add_argument("--context", choices=("all", "6k", "32k", "64k", "262k"), default="all")
    parser.add_argument("--max-cases", type=int, default=0, help="0 = every selected question")
    parser.add_argument("--workers", type=int, default=8, help="case workers, capped at 16")
    parser.add_argument("--keys", type=int, default=0, help="use at most N loaded keys (0 = all)")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="build the exact case plan; make zero key, vault and model calls",
    )
    parser.add_argument(
        "--reader-model", default=locomo_reader.DEFAULT_READER_MODEL,
        help="reader model string (default: frozen harness reader)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_cases < 0 or args.keys < 0 or args.workers < 1:
        print("error: --max-cases/--keys must be >= 0 and --workers >= 1")
        return 2

    started = time.monotonic()
    guard = CallGuard(dry_run=args.dry_run)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    results_dir = Path(args.results_dir).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.subset}_{args.context}_max{args.max_cases or 'all'}"
    checkpoint_path = checkpoint_dir / f"mab_cr_{tag}.jsonl"
    result_path = results_dir / f"mab_cr_{tag}{'_dryrun' if args.dry_run else ''}.json"

    # The only dataset boundary: verified bytes, outside the repo.
    try:
        dataset_path = adapter.prepare(local_cache=LOCAL_CACHE if LOCAL_CACHE.is_file() else None)
        cases, plan = adapter.build_rows(dataset_path, args.subset, args.context, args.max_cases)
    except Exception as exc:
        print(f"blocker: dataset selection failed ({type(exc).__name__}: {exc})")
        return 2
    if not cases:
        print("blocker: empty selection; no denominator to report")
        return 2

    if args.dry_run:
        result = base_envelope(status="dry_run", dataset_path=dataset_path, plan=plan, args=args)
        result["denominators"] = {
            "selected": len(cases),
            "attempted": 0,
            "scored": 0,
            "failed": 0,
            "resumed_skipped": 0,
        }
        result["call_evidence"] = guard.evidence()
        result["subem"] = {"correct": 0, "denominator": 0, "mean_subem": None}
        result["wall_time_s"] = round(time.monotonic() - started, 3)
        write_json(result_path, result)
        print("mab-cr dry-run: official=false diagnostic=derived comparability_tier=C score_asserted=false")
        print(f"dataset: {dataset_path.name} sha256={adapter.DATASET_SHA256} verified=true")
        print(f"selected_cases: {len(cases)} source_rows_matched: {plan['source_rows_matched']}")
        print("call_evidence: " + json.dumps(guard.evidence(), sort_keys=True))
        print(f"result: {result_path}")
        return 0

    keys = guard.keys()
    keys = keys[: args.keys] if args.keys else keys
    print(f"keys_loaded: {len(keys)} (values withheld)")
    if not keys:
        result = base_envelope(status="blocked", dataset_path=dataset_path, plan=plan, args=args)
        result["reason"] = "no_keys_configured"
        result["denominators"] = {"selected": len(cases), "attempted": 0, "scored": 0}
        result["call_evidence"] = guard.evidence()
        write_json(result_path, result)
        print(f"blocker: no_keys_configured; result: {result_path}")
        return 2

    bin_path, cli_root = locomo_run.resolve_cli()
    if bin_path is None:
        result = base_envelope(status="blocked", dataset_path=dataset_path, plan=plan, args=args)
        result["reason"] = "mn_binary_not_found"
        result["denominators"] = {"selected": len(cases), "attempted": 0, "scored": 0}
        result["call_evidence"] = guard.evidence()
        write_json(result_path, result)
        print("blocker: mn binary not found; set MN_BIN to a public CLI build")
        return 2

    done = load_checkpoint(checkpoint_path)
    target = [case for case in cases if case["case_id"] not in done]
    print(f"checkpoint_resume: already_done={len(done)} selected={len(cases)} target={len(target)}")

    pool = guard.pool(keys)
    seed_detail: dict[str, Any] = {"vaults": 0, "facts_total": 0, "facts_seeded": 0}
    case_workers = max(1, min(args.workers, MAX_WORKERS_CEILING))

    if target:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for case in target:
            signature = hashlib.sha256(
                json.dumps(case["facts"], sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            groups[signature].append(case)

        with checkpoint_path.open("a", encoding="utf-8") as checkpoint_file:
            for group in groups.values():
                with tempfile.TemporaryDirectory(prefix="mn-bench-mabcr-") as temporary:
                    vault = Path(temporary) / "vault.db"
                    detail = seed_facts(guard, bin_path, cli_root, vault, group[0]["facts"])
                    seed_detail["vaults"] += 1
                    seed_detail["facts_total"] += detail["facts"]
                    seed_detail["facts_seeded"] += detail["seeded"]
                    if detail["seeded"] == 0:
                        for case in group:
                            record = {
                                "case_id": case["case_id"],
                                "subset": case["subset"],
                                "context": case["context"],
                                "qa_pair_id": case.get("qa_pair_id"),
                                "pred": None,
                                "gold_hash_prefix": hashlib.sha256(
                                    json.dumps(case["answers"], sort_keys=True).encode("utf-8")
                                ).hexdigest()[:12],
                                "subem": 0,
                                "reader_model": args.reader_model,
                                "error": "vault_seed_produced_no_facts",
                            }
                            checkpoint_file.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                            checkpoint_file.flush()
                        continue
                    with ThreadPoolExecutor(max_workers=case_workers) as executor:
                        futures = {
                            executor.submit(
                                process_case,
                                case=case,
                                bin_path=bin_path,
                                cli_root=cli_root,
                                vault=vault,
                                pool=pool,
                                guard=guard,
                                reader_model=args.reader_model,
                            ): case["case_id"]
                            for case in group
                        }
                        for future in as_completed(futures):
                            record = future.result()
                            checkpoint_file.write(
                                json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
                            )
                            checkpoint_file.flush()

    final_checkpoint = load_checkpoint(checkpoint_path)
    final_records = [
        final_checkpoint[case["case_id"]]
        for case in cases
        if case["case_id"] in final_checkpoint
    ]
    metrics = adapter.evaluate(final_records)
    failed = sum(1 for record in final_records if record.get("error"))
    result = base_envelope(status="completed", dataset_path=dataset_path, plan=plan, args=args)
    result["denominators"] = {
        "selected": len(cases),
        "checkpointed": len(final_records),
        "attempted": len(final_records),
        "scored": sum(1 for record in final_records if record.get("pred") is not None),
        "failed": failed,
        "resumed_skipped": len(done),
    }
    result["subem"] = metrics
    result["seed_detail"] = seed_detail
    result["key_pool_stats"] = pool.stats()
    result["call_evidence"] = guard.evidence()
    result["wall_time_s"] = round(time.monotonic() - started, 3)
    result["checkpoint_path"] = str(checkpoint_path)
    result["limitations"] = [
        "Judge-free SubEM still depends on the reader's answer text.",
        "Vault recall is top-k; facts outside the sweep are not shown to the reader.",
        "Our harness and reader are not the benchmark's: no rank, no published comparison.",
    ]
    write_json(result_path, result)

    print("mab-cr: official=false diagnostic=derived comparability_tier=C score_asserted=false")
    print(f"selected={len(cases)} checkpointed={len(final_records)} failed={failed}")
    print(
        f"subem_correct={metrics['correct']} denominator={metrics['denominator']} "
        f"mean={metrics['mean_subem']}"
    )
    print(f"models: reader={args.reader_model} judge=none")
    print(f"wall_time_s: {result['wall_time_s']}")
    print(f"result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
