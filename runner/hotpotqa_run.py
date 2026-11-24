"""Standalone, multi-key, checkpointed HotpotQA answer-only derived diagnostic.

This runner deliberately does NOT go through ``runner/bench.py``. It wires the
pinned ``harnesses.hotpotqa.adapter``, the eight-key ``runner.kilo_pool`` and the
frozen ``runner.locomo_reader`` transport directly (all imported, none edited).
Answer EM/F1 are computed locally with the official answer-only normaliser; no
LLM judge is used and no score is asserted as a product result.

HONESTY CONTRACT
----------------
* Every artifact is stamped official=False, diagnostic="derived",
  comparability_tier="C", score_asserted=False and "NOT an official score".
  No rank, win or leaderboard comparison is made.
* Gold is used only for local EM/F1 scoring. It is never seeded, never sent to
  the reader, and never written to a checkpoint or a result. Checkpoints carry
  only a gold hash prefix.
* Key values are never printed, logged or written. ``KeyPool.stats()`` reports
  counts/cooldowns only. CLI vault calls are model-free and never use a key.
* Dataset bytes come only from the adapter's pinned resolve/<sha>/ URL and are
  SHA-256 + byte-count verified before use, landing under D:\\memnoth-bench-data,
  never inside the repository. ``--dry-run`` performs no network, no CLI and no
  model call.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harnesses.hotpotqa import adapter  # noqa: E402
from runner import locomo_reader  # noqa: E402
from runner.kilo_pool import KeyPool, load_keys  # noqa: E402
from runner.locomo_run import (  # noqa: E402  (reused conventions, not edited)
    cli_recall_sweep,
    resolve_cli,
    sha256_text,
    status_from_error,
    utc_now,
    vault_remember,
    write_json,
)

VERSION = "0.1.0"
SCHEMA = "memnoth-bench-hotpotqa-run/1"
NOTE = (
    "INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE. Derived diagnostic on 150 "
    "pinned HotpotQA distractor-dev questions with a pinned extractive reader; "
    "judge-free answer-only EM/F1 computed locally. official=false, "
    "diagnostic=derived, comparability_tier=C, score_asserted=false. No rank, "
    "win or comparison to published systems is claimed."
)

DATA_ROOT = Path(r"D:\memnoth-bench-data")
DEFAULT_DATASET_DIR = DATA_ROOT / "datasets" / "hotpotqa"
DEFAULT_CHECKPOINT_DIR = DATA_ROOT / "checkpoints" / "hotpotqa"
DEFAULT_RESULTS_DIR = DATA_ROOT / "results"

DEFAULT_READER_MODEL = "particle-ai/particle-1:free"
BASE_URL = locomo_reader.DEFAULT_BASE_URL
READER_PROVIDER = "kilo"
MAX_KEY_RETRIES = 3
MAX_WORKERS_CEILING = 16
READER_MAX_TOKENS = 256
K_PARAGRAPHS = adapter.K_PARAGRAPHS

# Frozen extractive-reader prompt bytes. Deliberately NOT the LoCoMo 7-step
# prompt: HotpotQA needs a short span, not a date-resolved value. The transport
# (rate gate, bounded retries, short error codes) is locomo_reader._post_json.
HOTPOT_READER_PROMPT = (
    "You are an extractive reader for HotpotQA. Answer the question using ONLY "
    "the paragraphs provided. The answer is a short span copied from the text. "
    "Return ONLY that span on one line: no explanation, no reasoning, no "
    "labels, no quotation marks. If the answer is not present in the paragraphs, "
    "return UNSUPPORTED."
)
PUBLIC_FIELDS = ("question_id", "pred", "gold_hash_prefix", "em", "f1")


# --------------------------------------------------------------------------- #
# Honest helpers
# --------------------------------------------------------------------------- #


def checkpoint_row(record: dict[str, Any]) -> dict[str, Any]:
    """Exactly the five declared checkpoint fields: no gold, no key, no prompt."""
    return {field: record.get(field) for field in PUBLIC_FIELDS}


def load_done(checkpoint_path: Path) -> set[str]:
    done: set[str] = set()
    if checkpoint_path.is_file():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["question_id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


def pooled_stats(keys: list[str]) -> dict[str, Any]:
    """Pre-run count-only stats; no key value, prefix or fingerprint."""
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


def stamped_envelope(status: str, *, selected: int = 0, keys: list[str] | None = None) -> dict[str, Any]:
    """Base result with mandatory honesty stamps and zeroed metrics."""
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "observed_at": utc_now(),
        "official": False,
        "diagnostic": "derived",
        "comparability_tier": "C",
        "score_asserted": False,
        "not_an_official_score": True,
        "note": NOTE,
        "status": status,
        "lane": adapter.LANE,
        "denominators": {
            "selected": selected,
            "attempted": 0,
            "answered": 0,
            "abstained_or_failed": 0,
            "scored": 0,
            "resumed_skipped": 0,
            "em_correct": 0,
            "gold_labels_missing": 0,
        },
        "answer_em": {"correct": 0, "denominator": 0, "display": "0/0 (no denominator)"},
        "answer_f1": {"sum": 0.0, "denominator": 0, "mean": None, "display": "0/0 (no denominator)"},
        "key_pool_stats": pooled_stats(keys or []),
        "wall_time_s": 0.0,
        "top_failure_reasons": [],
        "cases": [],
        "limitations": [
            "Derived diagnostic on a 150-question slice, not the official 7,405-row split.",
            "Answer EM/F1 use the official answer-only normalisation but a generic "
            "LLM reader, not a fine-tuned extractive reader; not comparable to published rows.",
            "The per-case vault holds only that question's 10 paragraphs, so this is "
            "the oracle-context-style arm, not a cross-question vault-recall arm.",
            "Recall is capped at K=10 and the reader is given a bounded context; "
            "published systems use their own protocols.",
            "Gold and key values are never written; checkpoints carry only a gold hash prefix.",
        ],
    }


# --------------------------------------------------------------------------- #
# Model call: KeyPool ONLY here; gold is never in the arguments.
# --------------------------------------------------------------------------- #


def reader_with_pool(
    pool: KeyPool,
    question: str,
    paragraphs: list[str],
    reader_model: str,
) -> tuple[str | None, str | None]:
    """Frozen extractive reader over recalled paragraph text, with key retries."""
    user_content = (
        f"Question: {str(question or '')[:1000]}\n"
        "Paragraphs (source text only):\n"
        + "\n".join(f"[{index + 1}] {str(text)[:2000]}" for index, text in enumerate(paragraphs))
    )
    last_error = "reader_unavailable"
    for _ in range(MAX_KEY_RETRIES):
        key = pool.get()
        if key is None:
            return None, "no_key_available"
        content, error = locomo_reader._post_json(
            [
                {"role": "system", "content": HOTPOT_READER_PROMPT},
                {"role": "user", "content": user_content},
            ],
            reader_model,
            key,
            BASE_URL,
            READER_MAX_TOKENS,
        )
        if error is None and isinstance(content, str) and content.strip():
            pool.release(key, True, 200)
            answer = " ".join(content.strip().splitlines()[0].split())[:300]
            return (answer or None), None
        pool.release(key, False, status_from_error(error))
        last_error = str(error or "reader_unknown_error")
    return None, last_error


# --------------------------------------------------------------------------- #
# One sequential case chain: per-case vault seed -> question-only recall -> reader.
# --------------------------------------------------------------------------- #


def process_case(
    *,
    index: int,
    sample: dict[str, Any],
    pool: KeyPool,
    reader_model: str,
    bin_path: Path | None,
    cli_root: Path | None,
) -> dict[str, Any]:
    """Run one case. Only public fields are returned; gold stays in ``sample``."""
    with tempfile.TemporaryDirectory(prefix="mn-bench-hotpotqa-") as temporary:
        vault = Path(temporary) / "vault.db"

        def seed_fn(entity: str, text: str) -> bool:
            return vault_remember(bin_path, cli_root, vault, entity, text)

        def recall_fn(question: str, k: int) -> list[str]:
            # The query is the question alone; no gold, no id, no labels.
            return cli_recall_sweep(bin_path, cli_root, vault, question, k)

        def reader_fn(question: str, facts: list[str]) -> tuple[str | None, str | None]:
            return reader_with_pool(pool, question, facts, reader_model)

        try:
            built = adapter.build_rows(
                [sample],
                seed_fn=seed_fn,
                recall_fn=recall_fn,
                reader_fn=reader_fn,
                k=K_PARAGRAPHS,
            )
            row = built[0]
        except Exception as exc:  # never let a worker exception fabricate a score
            qid = str(sample.get("id") or f"case-{index}")
            return {
                "question_id": qid,
                "pred": None,
                "gold_hash_prefix": adapter.gold_hash_prefix(sample.get("answer")),
                "em": 0,
                "f1": 0.0,
                "_error": f"case_error_{type(exc).__name__}",
                "_seeded": 0,
                "_recalled": 0,
            }

    published = checkpoint_row(row)
    published["_error"] = row.get("reader_error")
    published["_seeded"] = row.get("seeded_count")
    published["_recalled"] = row.get("recalled_count")
    return published


# --------------------------------------------------------------------------- #
# CLI / argparse
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-cases", type=int, default=adapter.SLICE_SIZE,
        help="cases to run (default 150 = the pinned slice; 0 also means 150)",
    )
    parser.add_argument("--workers", type=int, default=8, help="case workers, capped at 16")
    parser.add_argument("--keys", type=int, default=8, help="use at most N loaded keys (0 = all)")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="verify pins/selection offline; no network, no CLI and no model call",
    )
    parser.add_argument(
        "--reader-model", default=DEFAULT_READER_MODEL,
        help=f"reader model (default {DEFAULT_READER_MODEL}); override is recorded",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Dry run: offline only
# --------------------------------------------------------------------------- #


def run_dry(
    *,
    args: argparse.Namespace,
    selection: dict[str, Any],
    result_path: Path,
    started: float,
) -> int:
    local_path = adapter.DEFAULT_DATA_DIR / adapter.DATASET_FILENAME
    dataset: dict[str, Any] = {
        "local_name": local_path.name,
        "path": str(local_path),
        "present": local_path.is_file(),
        "sha256": None,
        "verified": False,
        "note": "no network access; only a locally prepared file was inspected",
    }
    ids: list[str] = []
    if local_path.is_file():
        observed = adapter.sha256_file(local_path)
        dataset["sha256"] = observed
        dataset["verified"] = observed == adapter.DATASET_SHA256
        if dataset["verified"]:
            rows = adapter.load_rows(local_path, args.max_cases)
            ids = [str(row["id"]) for row in rows]
            dataset["rows_available"] = len(rows)

    result = stamped_envelope("dry_run", selected=len(ids) or selection["effective_denominator"])
    result.update(
        {
            "selection": {key: value for key, value in selection.items() if key != "ids"},
            "selection_ids": ids,
            "dataset": dataset,
            "reader": {
                "model": args.reader_model,
                "provider": READER_PROVIDER,
                "base_url": BASE_URL,
                "max_tokens": READER_MAX_TOKENS,
                "prompt_sha256": adapter.sha256_text(HOTPOT_READER_PROMPT),
                "prompt_bytes_pinned": True,
                "gold_withheld_from_reader": True,
                "frozen_default": args.reader_model == DEFAULT_READER_MODEL,
            },
            "retrieval": {
                "k": K_PARAGRAPHS,
                "vault_scope": "one temp vault per case with that case's 10 paragraphs",
                "recall_query": "question text only",
            },
            "calls_made": {
                "network": False, "cli": False, "model": False, "keys_loaded": False,
            },
            "case_workers": max(1, min(args.workers, MAX_WORKERS_CEILING)),
        }
    )
    result["wall_time_s"] = round(time.monotonic() - started, 3)
    result["limitations"].append("Dry run made no calls; it asserts no score and ran no reader.")
    write_json(result_path, result)
    print("dry_run: no network / no CLI / no model calls")
    print(f"pins: dataset_sha256={adapter.DATASET_SHA256} slice_id_sha={adapter.SLICE_ID_SHA256}")
    print(
        f"selection: n={result['denominators']['selected']} "
        f"id_sha_matches_pin={selection['id_list_sha256_matches_pin']} "
        f"dataset_verified={dataset['verified']}"
    )
    print(f"reader: {args.reader_model} k={K_PARAGRAPHS}")
    print("official=false diagnostic=derived comparability_tier=C score_asserted=false")
    print(f"result: {result_path}")
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_cases < 0 or args.keys < 0 or args.workers < 1:
        print("error: --max-cases/--keys must be >= 0 and --workers >= 1")
        return 2

    started = time.monotonic()
    selection = adapter.select(args.max_cases)
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    results_dir = Path(args.results_dir).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(args.max_cases) if args.max_cases else "all"
    checkpoint_path = checkpoint_dir / f"hotpotqa_max{suffix}_checkpoint.jsonl"
    result_path = results_dir / (
        "hotpotqa_dry_run.json" if args.dry_run else f"hotpotqa_max{suffix}.json"
    )

    if args.dry_run:
        return run_dry(args=args, selection=selection, result_path=result_path, started=started)

    # Network boundary: prepared only in a real run, verifed before any use.
    try:
        dataset_path = adapter.prepare(DEFAULT_DATASET_DIR)
    except Exception as exc:
        print(f"blocker: dataset unavailable ({type(exc).__name__}: {exc})")
        return 2

    try:
        rows = adapter.load_rows(dataset_path, args.max_cases)
    except Exception as exc:
        print(f"blocker: dataset structure/sha ({type(exc).__name__}: {exc})")
        return 2
    if not rows:
        print("blocker: pinned dataset produced no selected rows")
        return 2

    bin_path, cli_root = resolve_cli()
    if bin_path is None:
        print("blocker: mn binary not found; set MN_BIN to a public CLI build")
        return 2

    all_keys = load_keys()  # values are never printed or written
    keys = all_keys[: args.keys] if args.keys else all_keys
    print(f"keys_loaded: {len(keys)} (values withheld)")
    if not keys:
        result = stamped_envelope("blocked", selected=len(rows), keys=keys)
        result["reason"] = "no_keys_configured"
        result["dataset"] = {
            "local_name": dataset_path.name,
            "sha256": adapter.DATASET_SHA256,
            "bytes": adapter.DATASET_BYTES,
            "verified": True,
        }
        result["wall_time_s"] = round(time.monotonic() - started, 3)
        write_json(result_path, result)
        print(f"blocker: no_keys_configured; result: {result_path}")
        return 2

    done = load_done(checkpoint_path)
    target = [
        (index, sample)
        for index, sample in enumerate(rows)
        if str(sample["id"]) not in done
    ]
    print(f"checkpoint_resume: already_done={len(done)} selected={len(rows)} target={len(target)}")
    if not target and result_path.is_file():
        print("resume no-op: all selected cases already checkpointed; prior result preserved")
        return 0

    pool = KeyPool(keys)
    case_workers = max(1, min(args.workers, MAX_WORKERS_CEILING))
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=case_workers) as executor:
        futures = {
            executor.submit(
                process_case,
                index=index,
                sample=sample,
                pool=pool,
                reader_model=args.reader_model,
                bin_path=bin_path,
                cli_root=cli_root,
            ): index
            for index, sample in target
        }
        # The main thread owns the append so concurrent cases cannot interleave bytes.
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint_file:
            for future in as_completed(futures):
                record = future.result()
                checkpoint_file.write(
                    json.dumps(checkpoint_row(record), sort_keys=True, ensure_ascii=False) + "\n"
                )
                checkpoint_file.flush()
                records.append(record)

    # Gold is attached in memory only, for the local official scorer.
    samples_by_id = {str(sample["id"]): sample for sample in rows}
    eval_rows = [
        {
            "question_id": record["question_id"],
            "pred": record["pred"],
            "gold": samples_by_id[record["question_id"]]["answer"],
        }
        for record in records
        if record["question_id"] in samples_by_id
    ]
    evaluation = adapter.evaluate(eval_rows)
    em = evaluation["metrics"]["answer_em"]
    f1 = evaluation["metrics"]["answer_f1"]

    answered = sum(1 for record in records if record["pred"] is not None)
    failure_counts = Counter(str(record["_error"]) for record in records if record["_error"])
    top_failures = [
        {"reason": reason, "count": count}
        for reason, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    result = stamped_envelope("completed", selected=len(rows), keys=keys)
    result.update(
        {
            "dataset": {
                "local_name": dataset_path.name,
                "sha256": adapter.DATASET_SHA256,
                "bytes": adapter.DATASET_BYTES,
                "verified": True,
                "url": adapter.DATASET_URL,
                "config": adapter.CONFIG,
                "split": adapter.SPLIT,
            },
            "selection": {key: value for key, value in selection.items() if key != "ids"},
            "reader": {
                "model": args.reader_model,
                "provider": READER_PROVIDER,
                "base_url": BASE_URL,
                "max_tokens": READER_MAX_TOKENS,
                "prompt_sha256": adapter.sha256_text(HOTPOT_READER_PROMPT),
                "prompt_bytes_pinned": True,
                "gold_withheld_from_reader": True,
                "frozen_default": args.reader_model == DEFAULT_READER_MODEL,
            },
            "retrieval": {
                "k": K_PARAGRAPHS,
                "vault_scope": "one temp vault per case with that case's 10 paragraphs",
                "recall_query": "question text only",
            },
            "denominators": {
                "selected": len(rows),
                "attempted": len(records),
                "answered": answered,
                "abstained_or_failed": len(records) - answered,
                "scored": em["denominator"],
                "resumed_skipped": len(done),
                "em_correct": em["correct"],
                "gold_labels_missing": evaluation["metrics"]["gold_labels_missing"],
                "missing_pred_scored_zero": evaluation["metrics"]["missing_pred_scored_zero"],
            },
            "answer_em": {
                "correct": em["correct"],
                "denominator": em["denominator"],
                "display": em["display"],
            },
            "answer_f1": {
                "sum": f1["sum"],
                "denominator": f1["denominator"],
                "mean": f1["mean"],
                "display": f1["display"],
            },
            "scorer": {
                "url": adapter.SCORER_URL,
                "sha256": adapter.SCORER_SHA256,
                "sha256_note": "not frozen; None rather than an invented hash",
                "semantics": adapter.SCORER_SEMANTICS,
            },
            "key_pool_stats": pool.stats(),
            "case_workers": case_workers,
            "max_key_retries": MAX_KEY_RETRIES,
            "wall_time_s": round(time.monotonic() - started, 3),
            "top_failure_reasons": top_failures,
            "checkpoint_path": str(checkpoint_path),
            "cases": [
                {
                    **checkpoint_row(record),
                    "_error": record.get("_error"),
                    "_seeded": record.get("_seeded"),
                    "_recalled": record.get("_recalled"),
                }
                for record in records
            ],
        }
    )
    # Hard guard: no gold field and no key field may reach a published artifact.
    assert all(
        "gold" not in case and "key" not in case for case in result["cases"]
    ), "gold/key field leaked into published cases"

    write_json(result_path, result)
    print("hotpotqa: official=false diagnostic=derived comparability_tier=C score_asserted=false")
    print(f"selection: n={len(rows)} dataset_sha256={adapter.DATASET_SHA256}")
    print(
        "denominators: "
        f"attempted={len(records)} answered={answered} "
        f"abstained_or_failed={len(records) - answered} scored={em['denominator']}"
    )
    print(
        f"local_scorer: em={em['correct']}/{em['denominator']} "
        f"f1_sum={f1['sum']}/{f1['denominator']} mean_f1={f1['mean']} "
        "(derived diagnostic; NOT an official score)"
    )
    print(f"reader: {args.reader_model} k={K_PARAGRAPHS} workers={case_workers}")
    print("top_failure_reasons: " + (json.dumps(top_failures, sort_keys=True) if top_failures else "none"))
    print(f"checkpoint: {checkpoint_path}")
    print(f"result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
