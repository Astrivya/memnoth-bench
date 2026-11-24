"""Fast, sharded LoCoMo derived diagnostic runner (Tier C; no score asserted).

This is a NEW standalone runner. It does not edit or wrap any existing runner's
behaviour; it imports the frozen pieces from ``runner.locomo_run``,
``runner.locomo_seed`` and ``runner.locomo_reader`` and reuses their seed ->
recall -> asof chain verbatim.

WHY IT IS FASTER (mechanism, not a measured win)
------------------------------------------------
* The PRIMARY metric is LOCAL token F1 against gold, computed with
  ``harnesses.common.token_f1``. A judge model call is therefore not needed for
  F1. ``--skip-judge`` is ON by default; the frozen strict judge runs only when
  ``--no-skip-judge`` (or ``--judge``) is passed explicitly.
* The reader is switched to a cheaper free model (default
  ``particle-ai/particle-1:free``); ``--reader-model`` allows the frozen nemotron
  reader for an A/B. MODEL_PROBE.md found no measured accuracy winner among the
  five probed free readers, so this is a cost/latency choice only.
* Questions are partitioned by ``index % shards == shard_index`` and each shard
  process gets a DISJOINT key slice ``keys[shard_index::shards]``. Each process
  has its own model gate (``MEMNOTH_MODEL_GATE``). If every shard is launched
  concurrently, ``shards * gate`` MUST stay <= 16; lower the per-process gate
  (or launch shards in batches) to respect that budget.

HONESTY CONTRACT
----------------
* official=false, diagnostic=derived, comparability_tier=C, score_asserted=false.
* No cross-shard aggregation, no headline score, no rank, no paper comparison.
* Gold is used ONLY for local F1 and (optionally) the strict judge. It is never
  seeded, never sent to the reader, and never written to a checkpoint/result.
* Checkpoints store the raw prediction plus a gold hash prefix only. No gold
  value and no key value is ever printed, logged or written.
* The pinned, SHA-256-verified dataset is used; bytes stay under
  D:\\memnoth-bench-data, never inside the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Canonical metric: the single source of truth, not a look-alike copy.
from harnesses.common import token_f1 as canonical_token_f1  # noqa: E402
from runner import locomo_reader, locomo_run, locomo_seed  # noqa: E402
from runner.kilo_pool import KeyPool, load_keys  # noqa: E402

VERSION = "0.1.0"
SCHEMA = "memnoth-bench-locomo-fast/1"
COMPARABILITY_TIER = "C"
NOTE = (
    "INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE. Sharded, derived LoCoMo "
    "diagnostic: real CLI recall + frozen 7-step reader, local token F1 against "
    "gold, judge skipped by default. No score is asserted, no rank is implied, "
    "and no shard result is aggregated into one."
)

DEFAULT_READER_MODEL = "particle-ai/particle-1:free"
NEMOTRON_READER_MODEL = locomo_run.READER_MODEL  # frozen reader, for A/B only
KNOWN_READER_MODELS = (DEFAULT_READER_MODEL, NEMOTRON_READER_MODEL)

DATA_ROOT = Path(r"D:\memnoth-bench-data")
DEFAULT_CHECKPOINT_DIR = locomo_run.DEFAULT_CHECKPOINT_DIR.parent / "locomo_fast"
DEFAULT_RESULTS_DIR = locomo_run.DEFAULT_RESULTS_DIR / "locomo_fast"
MAX_CONCURRENCY_BUDGET = 16

PARTITION_RULE = "index % shards == shard_index"
KEY_SLICE_RULE = "keys[shard_index::shards]"


# --------------------------------------------------------------------------- #
# Honest helpers (reuse locomo_run; no key/gold values in any output)
# --------------------------------------------------------------------------- #


def utc_now() -> str:
    return locomo_run.utc_now()


def sha256_text(value: Any) -> str:
    return locomo_run.sha256_text(value)


def write_json(path: Path, value: Any) -> None:
    locomo_run.write_json(path, value)


def load_done(path: Path) -> set[str]:
    return locomo_run.load_done(path)


def f1_parity_ok() -> bool:
    """Guard: the local metric must equal harnesses.common.token_f1 exactly."""
    samples = [
        ("7 May 2023", "7 May 2023"),
        ("7 may 2023", "May 7, 2023"),
        ("UNSUPPORTED", "7 May 2023"),
        ("", ""),
        (None, "answer"),
    ]
    return all(
        canonical_token_f1(pred, ref) == locomo_run.token_f1(pred, ref)
        for pred, ref in samples
    )


def effective_gate() -> int:
    """The gate value the imported locomo_reader actually uses (read at import)."""
    try:
        return int(locomo_reader._MODEL_GATE_LIMIT)
    except (AttributeError, TypeError, ValueError):
        return max(1, min(int(os.environ.get("MEMNOTH_MODEL_GATE", "8")), 16))


def gate_note(shards: int, gate: int) -> dict[str, Any]:
    total = shards * gate
    recommended = max(1, min(gate, MAX_CONCURRENCY_BUDGET // max(1, shards)))
    if shards > MAX_CONCURRENCY_BUDGET:
        advice = (
            f"shards>16: even gate=1 gives {shards} concurrent slots; launch "
            f"at most {MAX_CONCURRENCY_BUDGET} shards at a time (batch them)."
        )
    elif total > MAX_CONCURRENCY_BUDGET:
        advice = (
            f"set MEMNOTH_MODEL_GATE={recommended} in EACH shard process before "
            f"launching it, or run fewer shards concurrently."
        )
    else:
        advice = "within budget; no change needed."
    return {
        "per_process_gate": gate,
        "shards_times_gate": total,
        "budget": MAX_CONCURRENCY_BUDGET,
        "within_budget": total <= MAX_CONCURRENCY_BUDGET,
        "recommended_gate": recommended,
        "advice": advice,
    }


def shard_indices(total: int, shards: int, shard_index: int) -> list[int]:
    """Disjoint, exhaustive partition of range(total) over the shards."""
    return [index for index in range(total) if index % shards == shard_index]


def qid(conversation_index: int, index: int) -> str:
    # Same stable identity as locomo_run, so checkpoints are comparable by id.
    return f"locomo-{conversation_index}-{index + 1}"


def paths_for(
    checkpoint_dir: Path, results_dir: Path, conversation_index: int,
    shards: int, shard_index: int, max_cases: int,
) -> tuple[Path, Path]:
    suffix = str(max_cases) if max_cases else "all"
    stem = f"locomo_fast_conv{conversation_index}_shard{shard_index}of{shards}_max{suffix}"
    return (
        checkpoint_dir / f"{stem}_checkpoint.jsonl",
        results_dir / f"{stem}.json",
    )


def redacted_command(
    args: argparse.Namespace, shard_index: int, gate: int
) -> str:
    """A launch template with NO key values and no dataset contents."""
    env_prefix = f"$env:MEMNOTH_MODEL_GATE='{gate}'"
    parts = [
        "python runner/locomo_fast.py",
        f"--shards {args.shards}",
        f"--shard-index {shard_index}",
        f"--reader-model {args.reader_model}",
        "--skip-judge" if args.skip_judge else "--no-skip-judge",
        f"--max-cases {args.max_cases}",
        f"--keys {args.keys}",
        f"--workers {args.workers}",
        f'--checkpoint-dir "{args.checkpoint_dir}"',
        f'--results-dir "{args.results_dir}"',
    ]
    if args.conversation_index != 0:
        parts.append(f"--conversation-index {args.conversation_index}")
    return f"{env_prefix}; " + " ".join(parts)


# --------------------------------------------------------------------------- #
# Reader: frozen 7-step prompt, configurable model, KeyPool retries only here
# --------------------------------------------------------------------------- #


def reader_with_model(
    pool: KeyPool, question: str, facts: list[str], reader_model: str
) -> tuple[str | None, str | None]:
    last_error = "reader_unavailable"
    for _ in range(locomo_run.MAX_KEY_RETRIES):
        key = pool.get()
        if key is None:
            return None, "no_key_available"
        answer, error = locomo_reader.reader_answer(
            question,
            facts,
            key,
            reader_model=reader_model,
            base_url=locomo_run.BASE_URL,
            system_prompt=locomo_run.READER_PROMPT,
        )
        if error is None and isinstance(answer, str) and answer.strip():
            pool.release(key, True, 200)
            return answer.strip()[:300], None
        pool.release(key, False, locomo_run.status_from_error(error))
        last_error = str(error or "reader_unknown_error")
    return None, last_error


def process_case(
    *,
    index: int,
    question: dict[str, Any],
    conversation_index: int,
    shard_index: int,
    bin_path: Path | None,
    cli_root: Path | None,
    vault: Path,
    pool: KeyPool,
    reader_model: str,
    skip_judge: bool,
) -> dict[str, Any]:
    """One case: seed-free recall -> asof -> reader -> LOCAL F1 -> optional judge."""
    question_id = qid(conversation_index, index)
    question_text = str(question.get("question") or "")
    gold = str(question.get("answer") or "")

    record: dict[str, Any] = {
        "question_id": question_id,
        "index": index,
        "shard_index": shard_index,
        "pred": None,
        "gold_hash_prefix": sha256_text(gold)[:12] if gold else None,
        "f1": 0.0,
        "verdict": None,
        "reader_model": reader_model,
        "judge_model": None if skip_judge else locomo_run.JUDGE_MODEL,
        "error": None,
    }

    try:
        hits = locomo_seed.recall_question(
            lambda query, limit: locomo_run.cli_recall_sweep(
                bin_path, cli_root, vault, query, limit
            ),
            question_text,
            10,
        )
        if not hits:
            record["error"] = "no_recall_observation"
            return record
        facts: list[str] = []
        for name in hits:
            for item in locomo_run.vault_asof(bin_path, cli_root, vault, name):
                text = str(item).strip()
                if text and text not in facts:
                    facts.append(text)
        if not facts:
            record["error"] = "no_recalled_content"
            return record

        pred, reader_error = reader_with_model(pool, question_text, facts, reader_model)
        if pred is None:
            record["error"] = reader_error or "reader_failed"
            return record

        # Gold enters ONLY here (and, optionally, the judge below). It is never
        # given to the reader, never seeded and never written out.
        record["pred"] = pred
        record["f1"] = canonical_token_f1(pred, gold)

        if skip_judge:
            # F1 is local: no judge call at all, not even on failures.
            return record

        verdict, judge_error = locomo_run.judge_with_pool(
            pool, question_text, gold, pred
        )
        record["verdict"] = verdict
        record["error"] = (
            reader_error
            or judge_error
            or ("no_verdict" if verdict is None else None)
        )
        return record
    except Exception as exc:  # no worker exception may become a fabricated score
        record["error"] = f"case_error_{type(exc).__name__}"
        return record


# --------------------------------------------------------------------------- #
# Result envelope (never a score, never cross-shard aggregation)
# --------------------------------------------------------------------------- #


def base_envelope(
    *, reason: str, args: argparse.Namespace, conversation_index: int,
    sample_id: str | None, dataset_manifest: dict[str, Any] | None,
    keys: list[str], gate_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "observed_at": utc_now(),
        "official": False,
        "diagnostic": "derived",
        "comparability_tier": COMPARABILITY_TIER,
        "score_asserted": False,
        "note": NOTE,
        "status": reason,
        "lane": "locomo-small",
        "conversation_index": conversation_index,
        "sample_id": sample_id,
        "reader_model": args.reader_model,
        "judge_model": None if args.skip_judge else locomo_run.JUDGE_MODEL,
        "skip_judge": bool(args.skip_judge),
        "aggregation": "none: single shard only; no cross-shard score is computed",
        "shard": {
            "shards": args.shards,
            "shard_index": args.shard_index,
            "partition_rule": PARTITION_RULE,
            "key_slice_rule": KEY_SLICE_RULE,
            "key_slots": len(keys),
            "concurrency": gate_info,
        },
        "dataset": dataset_manifest,
        "denominators": {
            "attempted": 0, "scored": 0, "judged": 0, "no_verdict": 0,
            "judge_skipped": 0, "failed": 0, "selected": 0, "resumed_skipped": 0,
        },
        "correct_f1_count": 0,
        "f1_sum": 0.0,
        "unsupported_count": 0,
        "top_failure_reasons": [],
        "key_pool_stats": locomo_run.pool_stats_for(keys),
        "seed_detail": {},
        "workers": 0,
        "workers_requested": args.workers,
        "max_key_retries": locomo_run.MAX_KEY_RETRIES,
        "wall_time_s": 0.0,
        "cases": [],
        "limitations": [
            "Derived Tier C diagnostic on one pinned conversation; not an official LoCoMo split or score.",
            "Token F1 is computed locally against gold; no judge smoothing and no asserted aggregate score.",
            "Gold values and key values are never written; checkpoints carry only a gold hash prefix.",
            "Reader/judge identity is recorded from call arguments; the gateway does not attest it.",
            "A shard is not the whole run: denominators are shard-local and are never summed into a score.",
        ],
    }


# --------------------------------------------------------------------------- #
# Arg parsing / dry run
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reader-model", default=DEFAULT_READER_MODEL,
                        help="free reader model id; nemotron allowed for A/B")
    parser.add_argument("--skip-judge", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="default ON: F1 is local, so no judge call. "
                             "--no-skip-judge (or --judge) runs the strict judge")
    parser.add_argument("--judge", dest="skip_judge", action="store_false",
                        help="alias for --no-skip-judge (secondary verdict only)")
    parser.add_argument("--shards", type=int, default=4,
                        help="number of disjoint question shards (default 4)")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="this process's shard, 0..shards-1")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="question cases to consider before partitioning "
                             "(0 = all 199 in the chosen conversation)")
    parser.add_argument("--conversation-index", type=int, default=0,
                        help="pinned conversation to run (default 0; 199 questions)")
    parser.add_argument("--keys", type=int, default=8,
                        help="use at most N loaded keys before sharding (0 = all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="case workers requested (clamped to keys and gate)")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--dry-run", action="store_true",
                        help="print partition sizes and redacted commands; no calls")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> str | None:
    if args.shards < 1:
        return "--shards must be >= 1"
    if not 0 <= args.shard_index < args.shards:
        return "--shard-index must be in 0..shards-1"
    if args.max_cases < 0 or args.keys < 0 or args.workers < 1:
        return "--max-cases/--keys must be >= 0 and --workers >= 1"
    if not str(args.reader_model or "").strip():
        return "--reader-model must not be empty"
    return None


def load_questions(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Pinned dataset, verified before use; never touches model or key values."""
    dataset_path, dataset_note = locomo_run.fetch_pinned_dataset(
        locomo_run.DEFAULT_DATASET_DIR / "locomo10.json"
    )
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("pinned dataset is empty or not a list")
    if not 0 <= args.conversation_index < len(dataset):
        raise ValueError(f"--conversation-index must be 0..{len(dataset) - 1}")
    record = dataset[args.conversation_index]
    if not isinstance(record, dict):
        raise ValueError("conversation record is not an object")
    questions = [item for item in (record.get("qa") or []) if isinstance(item, dict)]
    if not questions:
        raise ValueError("conversation has no question rows")
    if args.max_cases:
        questions = questions[: args.max_cases]
    manifest = {
        "local_name": dataset_path.name,
        "sha256": locomo_run.DATASET_SHA256,
        "bytes": locomo_run.DATASET_BYTES,
        "verified": True,
        "note": dataset_note,
        "url": locomo_run.DATASET_URL,
    }
    return dataset_path, record, questions, manifest


def dry_run(args: argparse.Namespace) -> int:
    _, record, questions, _ = load_questions(args)
    total = len(questions)
    gate = effective_gate()
    gate_info = gate_note(args.shards, gate)

    print("locomo_fast DRY RUN: no model calls, no keys read, no checkpoint writes")
    print(f"official=false diagnostic=derived comparability_tier={COMPARABILITY_TIER} score_asserted=false")
    print(f"conversation={args.conversation_index} sample_id={record.get('sample_id')} "
          f"questions={total} partition={PARTITION_RULE}")
    print(f"reader_model={args.reader_model} skip_judge={args.skip_judge} "
          f"known_models={','.join(KNOWN_READER_MODELS)}")
    print(f"key_slice={KEY_SLICE_RULE} (values withheld; not read in dry-run)")
    print(f"gate: {json.dumps(gate_info, sort_keys=True)}")
    print("shard partition sizes:")
    for shard in range(args.shards):
        indices = shard_indices(total, args.shards, shard)
        first = qid(args.conversation_index, indices[0]) if indices else "-"
        last = qid(args.conversation_index, indices[-1]) if indices else "-"
        print(f"  shard {shard:>3}: n={len(indices):>3}  first={first}  last={last}")
    print("redacted commands (no key values):")
    if args.shards <= 16:
        for shard in range(args.shards):
            print(f"  {redacted_command(args, shard, gate_info['recommended_gate'])}")
    else:
        print(f"  {redacted_command(args, 0, gate_info['recommended_gate'])}")
        print(f"  ... repeat for --shard-index 1..{args.shards - 1} (batch <= 16 concurrently)")
        print(f"  {redacted_command(args, args.shards - 1, gate_info['recommended_gate'])}")
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    problem = validate_args(args)
    if problem:
        print(f"error: {problem}")
        return 2

    started = time.monotonic()
    gate_info = gate_note(args.shards, effective_gate())
    if not gate_info["within_budget"]:
        print("WARNING: " + json.dumps(gate_info, sort_keys=True))

    if args.dry_run:
        return dry_run(args)

    if not f1_parity_ok():
        print("blocker: local token F1 diverged from harnesses.common.token_f1")
        return 2

    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    results_dir = Path(args.results_dir).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path, result_path = paths_for(
        checkpoint_dir, results_dir, args.conversation_index,
        args.shards, args.shard_index, args.max_cases,
    )

    try:
        _, record, questions, manifest = load_questions(args)
    except Exception as exc:
        print(f"blocker: dataset unavailable ({type(exc).__name__}: {exc})")
        return 2
    sample_id = str(record.get("sample_id") or "")
    conversation = (
        record.get("conversation") if isinstance(record.get("conversation"), dict) else record
    )
    total = len(questions)
    indices = shard_indices(total, args.shards, args.shard_index)

    done = load_done(checkpoint_path)
    target = [(index, questions[index]) for index in indices if qid(args.conversation_index, index) not in done]
    print(
        f"shard: index={args.shard_index}/{args.shards} partition_n={len(indices)} "
        f"already_done={len(done)} target={len(target)}"
    )
    if not target and result_path.is_file():
        print("resume no-op: all shard cases already checkpointed; prior result preserved")
        return 0

    bin_path, cli_root = locomo_run.resolve_cli()
    if bin_path is None:
        print("blocker: mn binary not found; set MN_BIN to a public CLI build")
        return 2

    all_keys = load_keys()  # counts only; values are never printed or written
    loaded = all_keys[: args.keys] if args.keys else all_keys
    keys = loaded[args.shard_index :: args.shards]
    print(f"keys_loaded={len(loaded)} shard_key_slots={len(keys)} (values withheld)")
    if not keys:
        result = base_envelope(
            reason="blocked_no_keys_for_shard", args=args,
            conversation_index=args.conversation_index, sample_id=sample_id,
            dataset_manifest=manifest, keys=keys, gate_info=gate_info,
        )
        result["denominators"]["selected"] = len(indices)
        result["wall_time_s"] = round(time.monotonic() - started, 3)
        write_json(result_path, result)
        print(f"blocker: no keys in this shard's slice; result: {result_path}")
        return 2

    gate = effective_gate()
    case_workers = max(1, min(args.workers, locomo_run.MAX_WORKERS_CEILING, len(keys), gate))
    seed_detail: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    seed_note = "not_attempted"

    # One temp vault per shard process: seed ONCE, then recall per question.
    with tempfile.TemporaryDirectory(prefix="mn-bench-locomo-fast-") as temporary:
        vault = Path(temporary) / "vault.db"

        def remember(entity: str, text: str) -> bool:
            return locomo_run.vault_remember(bin_path, cli_root, vault, entity, text)

        all_writes_ok = locomo_seed.seed_conversation(
            remember, conversation, locomo_run.MAX_CLI_WRITE_WORKERS, detail=seed_detail
        )
        seeded = int(seed_detail.get("utterances_seeded", 0)) + int(
            seed_detail.get("date_facts_seeded", 0)
        )
        seed_note = "ok" if all_writes_ok else "partial"
        print(
            f"seed: all_writes_ok={all_writes_ok} utterances={seed_detail.get('utterances_seeded')} "
            f"date_facts={seed_detail.get('date_facts_seeded')}"
        )
        if seeded == 0:
            result = base_envelope(
                reason="blocked_vault_seed_produced_no_facts", args=args,
                conversation_index=args.conversation_index, sample_id=sample_id,
                dataset_manifest=manifest, keys=keys, gate_info=gate_info,
            )
            result["denominators"]["selected"] = len(indices)
            result["seed_detail"] = seed_detail
            result["seed_note"] = seed_note
            result["wall_time_s"] = round(time.monotonic() - started, 3)
            write_json(result_path, result)
            print("blocker: vault_seed_produced_no_facts")
            return 2

        pool = KeyPool(keys)
        if target:
            with ThreadPoolExecutor(max_workers=case_workers) as executor:
                futures = {
                    executor.submit(
                        process_case,
                        index=index,
                        question=question,
                        conversation_index=args.conversation_index,
                        shard_index=args.shard_index,
                        bin_path=bin_path,
                        cli_root=cli_root,
                        vault=vault,
                        pool=pool,
                        reader_model=args.reader_model,
                        skip_judge=args.skip_judge,
                    ): index
                    for index, question in target
                }
                # Main thread owns the append so concurrent cases cannot interleave bytes.
                with checkpoint_path.open("a", encoding="utf-8") as checkpoint_file:
                    for future in as_completed(futures):
                        case_record = future.result()
                        checkpoint_file.write(
                            json.dumps(case_record, sort_keys=True, ensure_ascii=False) + "\n"
                        )
                        checkpoint_file.flush()
                        records.append(case_record)

    wall = round(time.monotonic() - started, 3)
    attempted = len(records)
    scored = sum(1 for row in records if row["pred"] is not None)
    judged = sum(1 for row in records if row["verdict"] is not None)
    failed = sum(1 for row in records if row["error"] is not None)
    correct_f1_count = sum(1 for row in records if row["f1"] == 1.0)
    unsupported_count = sum(
        1 for row in records if locomo_reader.is_unsupported(row["pred"])
    )
    f1_sum = round(sum(float(row["f1"]) for row in records), 6)
    failure_counts: dict[str, int] = {}
    for row in records:
        if row["error"]:
            failure_counts[row["error"]] = failure_counts.get(row["error"], 0) + 1
    top_failures = [
        {"reason": reason, "count": count}
        for reason, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    result = base_envelope(
        reason="completed", args=args, conversation_index=args.conversation_index,
        sample_id=sample_id, dataset_manifest=manifest, keys=keys, gate_info=gate_info,
    )
    result.update(
        {
            "denominators": {
                "attempted": attempted,
                "scored": scored,
                "judged": judged,
                "no_verdict": 0 if args.skip_judge else attempted - judged,
                "judge_skipped": attempted if args.skip_judge else 0,
                "failed": failed,
                "selected": len(indices),
                "resumed_skipped": len(done),
            },
            "correct_f1_count": correct_f1_count,
            "f1_sum": f1_sum,
            "unsupported_count": unsupported_count,
            "top_failure_reasons": top_failures,
            "key_pool_stats": pool.stats(),
            "seed_detail": seed_detail,
            "seed_note": seed_note,
            "workers": case_workers,
            "wall_time_s": wall,
            "checkpoint_path": str(checkpoint_path),
            "result_path": str(result_path),
            "cases": [
                {
                    "question_id": row["question_id"],
                    "pred_hash_prefix": sha256_text(row["pred"])[:12] if row["pred"] else None,
                    "gold_hash_prefix": row["gold_hash_prefix"],
                    "f1": row["f1"],
                    "verdict": row["verdict"],
                    "error": row["error"],
                }
                for row in records
            ],
        }
    )
    write_json(result_path, result)

    print("locomo_fast: official=false diagnostic=derived comparability_tier=C score_asserted=false")
    print(
        f"shard={args.shard_index}/{args.shards} selected={len(indices)} "
        f"reader_model={args.reader_model} skip_judge={args.skip_judge}"
    )
    print(
        "denominators: "
        f"attempted={attempted} scored={scored} judged={judged} failed={failed} "
        f"no_verdict={result['denominators']['no_verdict']} "
        f"judge_skipped={result['denominators']['judge_skipped']}"
    )
    print(f"f1: correct_f1_count={correct_f1_count} sum={f1_sum} over n={scored} scored")
    print(f"wall_time_s: {wall}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
