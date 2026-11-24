"""Standalone, fast-path runner for REAL derived BEAM diagnostics.

This is deliberately separate from ``runner/bench.py``. It wires the already
pinned ``harnesses.beam.adapter`` and the eight-key ``runner.kilo_pool`` pool
directly, so a mini slice can be measured with real network calls and real
keys while the general bench wiring is done separately.

HONESTY CONTRACT
----------------
* Every emitted artifact is stamped official=False, diagnostic="derived",
  comparability_tier="C". No score is asserted as a product result.
* The dataset is downloaded only from the adapter's pinned URLs and is SHA-256
  verified before use; a mismatch aborts the run. Bytes land under
  D:\\memnoth-bench-data\\datasets\\beam, never inside the repository.
* No gold answer and no key value is ever written to a checkpoint, a result,
  or stdout. Checkpoints carry only a gold hash prefix and a rubric hash.
* Reader and judge models default to the adapter's frozen pins. An override is
  recorded as such so it cannot be mistaken for the frozen configuration.

WHY THERE IS A LOCAL DOWNLOADER
-------------------------------
``adapter.prepare(local_dir, tier)`` delegates to ``common.fetch_pinned`` whose
``ensure_data_path`` rejects anything outside this repository's gitignored
``data/`` directory. Passing ``D:\\memnoth-bench-data\\...`` raises PinError
before any byte is written. The runner calls prepare() first and records that
exact blocker, then reuses the adapter's own ``TIERS`` pins (URL, SHA-256, size)
in a streamed download with the same verify-before-move semantics. Pin values
are never invented here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import hmac
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harnesses.beam import adapter  # noqa: E402
from runner.kilo_pool import KeyPool, load_keys  # noqa: E402

VERSION = "0.1.0"
SCHEMA = "memnoth-bench-beam-run/1"
NOTE = (
    "Derived BEAM diagnostic, not an official BEAM run. A mini slice with a "
    "frozen reader/judge pair; raw verdicts are published, no score is claimed "
    "as a product result and no comparison to paper or vendor rows is implied."
)

DATA_ROOT = Path(r"D:\memnoth-bench-data")
DEFAULT_DATASET_DIR = DATA_ROOT / "datasets" / "beam"
DEFAULT_CHECKPOINT_DIR = DATA_ROOT / "checkpoints"
RESULTS_DIR = DATA_ROOT / "results"

TIER_KEYS: dict[str, str] = {"1m": "beam-1m", "10m": "beam-10m"}
MAX_KEY_RETRIES = 3
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_HTTP_STATUS = re.compile(r"_http_(\d{3})\b")


# --------------------------------------------------------------------------- #
# Small honest helpers
# --------------------------------------------------------------------------- #


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def status_from_error(code: str | None) -> int | None:
    """Return 429/5xx from an adapter error code, else None (no cooldown)."""
    match = _HTTP_STATUS.search(str(code or ""))
    if not match:
        return None
    status = int(match.group(1))
    if status == 429 or 500 <= status <= 599:
        return status
    return None


# --------------------------------------------------------------------------- #
# Pinned dataset preparation (verify before use, outside the repo)
# --------------------------------------------------------------------------- #


def fetch_pinned_external(url: str, sha256: str, expected_bytes: int, dest: Path) -> Path:
    """Stream ``url`` to ``dest`` and verify the SHA-256 before moving it in.

    Mirrors ``common.fetch_pinned`` semantics (cache reuse, temp file, atomic
    move, abort on mismatch) without the repository-only path restriction.
    """
    if not _HEX64.match(str(sha256).strip()):
        raise ValueError("dataset SHA-256 must be exactly 64 hex characters")
    if urlparse(str(url)).scheme.lower() not in {"http", "https"}:
        raise ValueError("only http(s) dataset URLs are supported")
    pin = str(sha256).strip().lower()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file():
        if dest.stat().st_size == int(expected_bytes) and hmac.compare_digest(sha256_file(dest), pin):
            print(f"dataset_cache_hit: {dest.name} bytes={dest.stat().st_size} sha256={pin[:12]}...")
            return dest
        dest.unlink()  # stale/corrupt cache: refetch under the same pin

    temp = dest.with_name(f"{dest.name}.part-{os.getpid()}")
    try:
        with urllib.request.urlopen(str(url), timeout=120) as response, temp.open("wb") as output:
            total = 0
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                output.write(block)
                total += len(block)
        observed = sha256_file(temp)
        if not hmac.compare_digest(observed, pin):
            raise RuntimeError(
                f"SHA-256 mismatch for {dest.name}: expected {pin}, got {observed}"
            )
        if total != int(expected_bytes):
            raise RuntimeError(
                f"byte-count mismatch for {dest.name}: expected {expected_bytes}, got {total}"
            )
        os.replace(temp, dest)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    print(f"dataset_verified: {dest.name} bytes={total} sha256={pin[:12]}...")
    return dest


def prepare_dataset(tier_key: str, dataset_dir: Path) -> tuple[list[Path], str, dict[str, Any]]:
    """Exercise adapter.prepare(), then prepare the same pinned shards externally."""
    spec = adapter.TIERS[tier_key]
    try:
        adapter.prepare(local_dir=dataset_dir, tier=tier_key)
        prepare_note = "adapter.prepare() succeeded at the requested external path"
    except Exception as exc:  # expected: ensure_data_path is repo-only
        prepare_note = f"adapter.prepare() declined external path ({type(exc).__name__}: {exc}); used adapter TIERS pins directly"

    paths: list[Path] = []
    manifest: list[dict[str, Any]] = []
    for item in spec["files"]:
        dest = dataset_dir / f"{spec['tier']}-{Path(item['path']).name}"
        paths.append(fetch_pinned_external(item["url"], item["sha256"], item["bytes"], dest))
        manifest.append(
            {
                "path": item["path"],
                "local_name": dest.name,
                "url": item["url"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "verified": True,
            }
        )
    return paths, prepare_note, {"files": manifest, "dir": str(dataset_dir)}


def load_dataset(paths: list[Path]) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the verified BEAM parquet shards") from exc
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(pq.read_table(path).to_pylist())
    return records


# --------------------------------------------------------------------------- #
# Reader / judge plumbing
# --------------------------------------------------------------------------- #


def parse_questions(raw: Any) -> Any:
    return ast.literal_eval(raw) if isinstance(raw, str) else raw


def rubric_map_for(dataset: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Reproduce build_rows' case_id order so each case has its own rubric."""
    mapping: dict[str, list[Any]] = {}
    records = [item for item in dataset if isinstance(item, dict)]
    records.sort(key=lambda item: str(item.get("conversation_id") or ""))
    for record in records:
        if adapter.validate(record):
            continue
        try:
            parsed = parse_questions(record.get("probing_questions"))
        except (ValueError, SyntaxError, TypeError, MemoryError):
            continue
        if not isinstance(parsed, dict):
            continue
        conversation_id = str(record["conversation_id"])
        for ability in adapter.ABILITIES:
            for index, item in enumerate(parsed.get(ability) or []):
                if isinstance(item, dict):
                    mapping[f"{conversation_id}::{ability}::{index + 1}"] = list(item.get("rubric") or [])
    return mapping


def facts_from_chat(chat: Any) -> str:
    """Turn conversation sessions into reader context; no probing labels used."""
    ranked_lists: list[list[dict[str, Any]]] = []
    sessions = chat if isinstance(chat, list) else []
    for session in sessions:
        turns = session if isinstance(session, list) else ([session] if isinstance(session, dict) else [])
        facts: list[dict[str, Any]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            content = " ".join(str(turn.get("content") or "").split())
            if not content:
                continue
            fact: dict[str, Any] = {
                "content": content,
                "speaker": turn.get("role") or turn.get("speaker") or "unknown",
                "time_anchor": turn.get("time_anchor") or turn.get("valid_from") or "",
            }
            if turn.get("valid_to"):
                fact["valid_to"] = turn["valid_to"]
            facts.append(fact)
        if facts:
            ranked_lists.append(facts)
    fused = adapter.rrf_fuse(ranked_lists) if ranked_lists else []
    return adapter.format_context(fused, top_k=adapter.RETRIEVAL_PROFILE["top_k"], chronological=True)


class BudgetedReader:
    """reader_fn for build_rows: spends calls only on selected question IDs."""

    def __init__(
        self,
        pool: KeyPool,
        model: str,
        selected: set[str],
        all_rows: list[dict[str, Any]],
    ) -> None:
        self.pool = pool
        self.model = model
        self.selected = selected
        self.all_rows = all_rows
        self.counter = 0
        self.calls = 0
        self.error_by_case: dict[str, str] = {}

    def __call__(self, question: str, chat: Any) -> str | None:
        index = self.counter
        self.counter += 1
        case_id = str(self.all_rows[index]["case_id"]) if index < len(self.all_rows) else ""
        if case_id not in self.selected:
            return None  # no spend on cases outside this mini slice
        context = facts_from_chat(chat)
        last_error = "reader_unavailable"
        for _ in range(MAX_KEY_RETRIES):
            key = self.pool.get()
            if key is None:
                last_error = "no_key_available"
                break
            answer, error = adapter.reader_call(question, context, api_key=key, model=self.model)
            self.calls += 1
            if answer:
                self.pool.release(key, True, 200)
                self.error_by_case.pop(case_id, None)
                return answer
            self.pool.release(key, False, status_from_error(error))
            last_error = str(error or "reader_unknown_error")
        self.error_by_case[case_id] = last_error
        return None


def judge_with_pool(
    pool: KeyPool,
    question: str,
    answer: str,
    gold: str,
    rubric: list[Any],
    model: str,
) -> dict[str, Any]:
    last_error = "judge_unavailable"
    for _ in range(MAX_KEY_RETRIES):
        key = pool.get()
        if key is None:
            return {"verdict": None, "error": "no_key_available"}
        result = adapter.judge_call(
            question, answer, gold, rubric, api_key=key, model=model
        )
        if not result.get("error"):
            pool.release(key, True, 200)
            return {"verdict": result.get("verdict"), "error": None}
        pool.release(key, False, status_from_error(result.get("error")))
        last_error = str(result.get("error"))
    return {"verdict": None, "error": last_error}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=tuple(TIER_KEYS), default="1m")
    parser.add_argument("--max-cases", type=int, default=5, help="question cases in the mini slice (0 = all)")
    parser.add_argument("--reader-model", default=adapter.READER_MODEL)
    parser.add_argument("--judge-model", default=adapter.JUDGE_MODEL)
    parser.add_argument("--keys", type=int, default=0, help="use at most N loaded keys (0 = all)")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_cases < 0 or args.keys < 0:
        print("error: --max-cases and --keys must be non-negative")
        return 2
    tier_key = TIER_KEYS[args.tier]
    dataset_dir = Path(args.dataset_dir).expanduser()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    result_path = RESULTS_DIR / f"beam_{args.tier}_{args.max_cases}.json"
    checkpoint_path = checkpoint_dir / f"beam_{args.tier}_checkpoint.jsonl"

    all_keys = load_keys()  # counts only; values are never printed or written
    keys = all_keys[: args.keys] if args.keys else all_keys
    print(f"keys_loaded: {len(keys)} (values withheld)")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths, prepare_note, dataset_manifest = prepare_dataset(tier_key, dataset_dir)
    selection = adapter.select(max_cases=args.max_cases, tier=tier_key)
    dataset = load_dataset(paths)

    # Metadata pass: build_rows with no reader_fn makes no model calls.
    all_rows, _attempted_all, build_failures = adapter.build_rows(dataset, reader_fn=None, tier=tier_key)
    rubrics = rubric_map_for(dataset)

    done: set[str] = set()
    if checkpoint_path.is_file():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["question_id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    print(f"checkpoint_resume: already_done={len(done)}")

    # The slice is the canonical first N case IDs, always the same across
    # re-runs. Resume only removes IDs already checkpointed *within that slice*;
    # it never silently advances to a different set of cases.
    slice_ids: list[str] = []
    for row in all_rows:
        if args.max_cases and len(slice_ids) >= args.max_cases:
            break
        slice_ids.append(str(row["case_id"]))
    target = [case_id for case_id in slice_ids if case_id not in done]
    selected = set(target)

    models = {
        "reader": args.reader_model,
        "judge": args.judge_model,
        "reader_provider": adapter.READER_PROVIDER,
        "judge_provider": adapter.JUDGE_PROVIDER,
        "frozen_defaults_used": (
            args.reader_model == adapter.READER_MODEL and args.judge_model == adapter.JUDGE_MODEL
        ),
        "note": adapter.READER["note"] + " " + adapter.JUDGE["note"],
    }

    if not target and keys and result_path.is_file():
        # Full resume: preserve the already-written real result; do not clobber
        # it with a zeros "noop" envelope simply because the IDs are done.
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        den = prior.get("denominators", {})
        print("BEAM diagnostic run: resume no-op; existing result preserved (no model calls)")
        print(
            "denominators: "
            f"attempted={den.get('attempted')} judged={den.get('judged')} "
            f"correct={den.get('correct')} no_verdict={den.get('no_verdict')}"
        )
        print(f"result: {result_path} (unchanged)")
        return 0

    if not keys or not target:
        # Honest blocker or nothing left to do; never invent a score.
        outcome = "no_keys_configured" if not keys else "nothing_to_do_all_cases_checkpointed"
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "version": VERSION,
            "observed_at": utc_now(),
            "official": False,
            "diagnostic": "derived",
            "comparability_tier": "C",
            "score_asserted": False,
            "note": NOTE,
            "status": "blocked" if not keys else "resume_noop",
            "reason": outcome,
            "tier": args.tier,
            "models": models,
            "denominators": {"attempted": 0, "judged": 0, "correct": 0, "no_verdict": 0},
            "dataset": dataset_manifest,
            "dataset_prepare_note": prepare_note,
            "selection": selection,
            "key_pool_stats": pool_stats_for(keys),
            "limitations": ["No score is asserted when no real call was made."],
        }
        write_json(result_path, result)
        print(f"blocker: {outcome}")
        print(f"result: {result_path}")
        return 2 if not keys else 0

    pool = KeyPool(keys)
    reader = BudgetedReader(pool, args.reader_model, selected, all_rows)
    rows, _attempted2, build_failures2 = adapter.build_rows(
        dataset, reader_fn=reader, max_cases=None, tier=tier_key
    )
    by_case = {str(row["case_id"]): row for row in rows}

    case_stats: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint_file:
        for case_id in target:
            row = by_case.get(case_id)
            if row is None:
                case_stats.append(
                    {"question_id": case_id, "verdict": None, "error": "case_missing_after_build"}
                )
                continue
            pred = row.get("pred_answer") if isinstance(row.get("pred_answer"), str) else None
            gold = row.get("gold_answer") if isinstance(row.get("gold_answer"), str) else ""
            reader_error = reader.error_by_case.get(case_id)
            verdict: str | None = None
            judge_error: str | None = None
            if pred and pred.strip():
                judged = judge_with_pool(
                    pool,
                    str(row.get("question") or ""),
                    pred,
                    gold,
                    rubrics.get(case_id, []),
                    args.judge_model,
                )
                verdict = judged["verdict"]
                judge_error = judged["error"]
            elif not reader_error:
                reader_error = "prediction_missing"

            record = {
                "question_id": case_id,
                "tier": args.tier,
                "pred": pred,
                "gold_hash_prefix": sha256_text(gold)[:12] if gold else None,
                "rubric_hash": sha256_text(
                    json.dumps(rubrics.get(case_id, []), sort_keys=True, ensure_ascii=False)
                )[:16],
                "verdict": verdict,
                "reader_model": args.reader_model,
                "judge_model": args.judge_model,
                "error": reader_error or judge_error,
            }
            checkpoint_file.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            checkpoint_file.flush()

            # Only the raw verdict label is retained for the adapter evaluation;
            # the full judge response could echo gold text and is not stored.
            row["judge_raw"] = verdict
            eval_rows.append(row)
            case_stats.append(
                {
                    "question_id": case_id,
                    "ability": row.get("ability"),
                    "verdict": verdict,
                    "error": record["error"],
                    "pred_hash_prefix": sha256_text(pred)[:12] if pred else None,
                }
            )

    attempted = len(case_stats)
    judged = sum(1 for case in case_stats if case["verdict"] is not None)
    correct = sum(1 for case in case_stats if case["verdict"] == "correct")
    denominator = {
        "attempted": attempted,
        "judged": judged,
        "correct": correct,
        "no_verdict": attempted - judged,
        "prediction_missing": sum(1 for case in case_stats if case["pred_hash_prefix"] is None),
        "selected": len(target),
        "resumed_skipped": len(done),
    }
    failure_counts = Counter(case["error"] for case in case_stats if case["error"])
    top_failures = [
        {"reason": reason, "count": count}
        for reason, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    evaluation = adapter.evaluate(eval_rows, tier=tier_key)

    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "observed_at": utc_now(),
        "official": False,
        "diagnostic": "derived",
        "comparability_tier": "C",
        "score_asserted": False,
        "note": NOTE,
        "status": "completed",
        "lane": "beam",
        "tier": args.tier,
        "adapter_tier": tier_key,
        "upstream_commit": adapter.UPSTREAM_COMMIT,
        "hf_commit": adapter.TIERS[tier_key]["hf_commit"],
        "dataset": dataset_manifest,
        "dataset_prepare_note": prepare_note,
        "selection": selection,
        "models": models,
        "denominators": denominator,
        "reader_calls": reader.calls,
        "judge_calls_with_keys": sum(1 for case in case_stats if case["verdict"] is not None),
        "key_pool_stats": pool.stats(),
        "top_failure_reasons": top_failures,
        "cases": case_stats,
        "adapter_evaluate": evaluation,
        "checkpoint_path": str(checkpoint_path),
        "build_failures": (build_failures + build_failures2)[:20],
        "limitations": [
            "Mini slice on a pinned subset; denominators are the real counts above.",
            "Derived SubEM/F1 proxy is not the official BEAM nugget score.",
            "Reader/judge identity is recorded from the call arguments; the gateway does not attest model identity.",
            "No paired same-pin comparison exists, so no score is comparable to paper or vendor rows.",
            "Raw judge responses are not stored; only the verdict label is retained to avoid gold echo.",
        ],
    }
    write_json(result_path, result)

    print("BEAM diagnostic run: official=false diagnostic=derived comparability_tier=C")
    print(f"tier: {args.tier} (adapter {tier_key})  max_cases: {args.max_cases}")
    print(
        "denominators: "
        f"attempted={denominator['attempted']} judged={denominator['judged']} "
        f"correct={denominator['correct']} no_verdict={denominator['no_verdict']}"
    )
    print(f"models: reader={args.reader_model} judge={args.judge_model}")
    print(f"reader_calls: {reader.calls}  keys_available_after: {pool.stats()['keys_available']}")
    print("top_failure_reasons: " + (json.dumps(top_failures, sort_keys=True) if top_failures else "none"))
    print(f"checkpoint: {checkpoint_path}")
    print(f"result: {result_path}")
    return 0


def pool_stats_for(keys: list[str]) -> dict[str, Any]:
    """A stats-shaped dict for the no-keys path, with no key values."""
    return {
        "keys_total": len(keys),
        "keys_available": len(keys),
        "keys_in_use": 0,
        "keys_cooling_down": 0,
        "unknown_release_count": 0,
        "slots": [
            {
                "slot": f"slot-{index}",
                "in_use": False,
                "cooling_down": False,
                "cooldown_remaining_s": 0.0,
                "successes": 0,
                "failures": 0,
                "rate_limited": 0,
            }
            for index in range(len(keys))
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
