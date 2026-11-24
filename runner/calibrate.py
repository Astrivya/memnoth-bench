"""Frozen-harness LoCoMo calibration: no-memory and oracle baselines.

Purpose
-------
Before attributing any score to memory, measure the FROZEN reader+judge harness
on the pinned LoCoMo data with NO memory at all:

* ``no_memory`` -- the reader sees ONLY the question (with an explicit
  chain-of-thought instruction in the system prompt; no facts, no evidence, no
  gold). This is the free reader's parametric ceiling under this harness.
* ``oracle`` -- the reader sees the question plus the evidence text that the
  dataset attaches to the gold answer. This is a SANITY CEILING CHECK ONLY. It
  is oracle by construction and is NOT a product result, not a memory result,
  and not comparable to any published LoCoMo number.

Safety / non-goals
------------------
* Gold answers go to ``judge_answer`` and to a SHA-256 hash ONLY. No gold VALUE
  is ever written to a checkpoint, a report, stdout or a log.
* Key values are loaded by name/path through ``kilo_pool`` and are never
  printed, logged or persisted. Output references their count/source only.
* Reader and judge models are the FROZEN module defaults; they are never
  overridden here. No model/version is varied to inflate a score.
* No score is asserted as a product result. ``official:false``, ``derived``,
  and counts always travel with their denominators.

The checkpoint is append-only JSONL and resumable: a case whose ``question_id``
is already present is skipped on later runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_REL = "data/locomo/locomo_conversation.json"
DATA_PATH = ROOT / DATA_REL
# Pinned by harnesses/locomo/adapter.py (upstream commit recorded there).
DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
CHECKPOINT_PATH = Path(r"D:\memnoth-bench-data\checkpoints\calibrate.jsonl")

SCHEMA = "memnoth-bench-calibrate/1"
VERSION = "0.1.0"
NOTE = "INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE"
MODES = ("no_memory", "oracle")
GOLD_HASH_CHARS = 16

# CoT for the no-facts condition. The words "oldest-first" and "unsupported"
# keep locomo_reader._effective_system_prompt from appending its fact-grounding
# clause: with zero facts that clause would be self-contradictory. This is a
# prompt only; the reader model/base_url/max_tokens and the judge stay FROZEN.
NO_MEMORY_COT_PROMPT = (
    "Answer the question using your own knowledge. No supporting facts are "
    "supplied.\n"
    "Work through these steps internally, then output ONLY step 4:\n"
    "1. Identify exactly what the question asks for.\n"
    "2. Recall the relevant knowledge, including any absolute date the question "
    "itself states.\n"
    "3. Derive one short, specific value; check it against the wording of the "
    "question.\n"
    "4. Output ONLY that final value on one line, with no explanation, no "
    "reasoning and no labels.\n"
    "No facts are supplied, so oldest-first fact ordering is not applicable. "
    "Do not answer UNSUPPORTED merely because no facts were provided; use it "
    "only if the question is genuinely unanswerable."
)


def _load_module(name: str, path: Path) -> Any:
    """Load a sibling runner module by file path without sys.path side effects."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_RUNNER = Path(__file__).resolve().parent
_reader = _load_module("_calib_locomo_reader", _RUNNER / "locomo_reader.py")
_pool_mod = _load_module("_calib_kilo_pool", _RUNNER / "kilo_pool.py")

reader_answer = _reader.reader_answer
judge_answer = _reader.judge_answer
DEFAULT_READER_MODEL = _reader.DEFAULT_READER_MODEL
DEFAULT_JUDGE_MODEL = _reader.DEFAULT_JUDGE_MODEL
load_keys = _pool_mod.load_keys
KeyPool = _pool_mod.KeyPool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", "replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_index(conversation: dict[str, Any]) -> tuple[dict[str, tuple[str, dict[str, Any]]], dict[str, str]]:
    """Map verbatim ``dia_id`` values to their real turn text and session dates.

    Only real conversation fields are read: ``session_N`` turn lists carry
    ``dia_id``/``speaker``/``text`` and ``session_N_date_time`` holds the real
    absolute date. Evidence IDs themselves come from the dataset's ``evidence``
    field, which is why the oracle lane is oracle and never a product result.
    """
    by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    dates: dict[str, str] = {}
    for key, value in conversation.items():
        date_match = re.fullmatch(r"(session_[A-Za-z0-9_-]+?)_date_time", str(key), re.IGNORECASE)
        if date_match and isinstance(value, str) and value.strip():
            dates[date_match.group(1)] = value.strip()
        if not (str(key).startswith("session_") and isinstance(value, list)):
            continue
        for turn in value:
            if not isinstance(turn, dict):
                continue
            dia_id = str(turn.get("dia_id") or "").strip()
            if dia_id and dia_id not in by_id:
                by_id[dia_id] = (str(key), turn)
    return by_id, dates


def load_cases(mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the pinned file and build stable, mode-namespaced case records."""
    if not DATA_PATH.is_file():
        raise RuntimeError(f"pinned dataset missing: {DATA_REL}")
    digest = sha256_file(DATA_PATH)
    if digest != DATASET_SHA256:
        raise RuntimeError(f"pinned dataset SHA-256 mismatch: {DATA_REL}")
    dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        raise RuntimeError("pinned dataset is not a list of conversations")

    cases: list[dict[str, Any]] = []
    qa_total = 0
    for conv_index, record in enumerate(dataset):
        if not isinstance(record, dict):
            continue
        conversation = record.get("conversation")
        conversation = conversation if isinstance(conversation, dict) else {}
        by_id, dates = evidence_index(conversation)
        qa = record.get("qa") or []
        if not isinstance(qa, list):
            continue
        for qa_index, question_record in enumerate(qa):
            if not isinstance(question_record, dict):
                continue
            qa_total += 1
            question = str(question_record.get("question") or "").strip()
            gold = str(question_record.get("answer") or "")
            if not question or not gold:
                continue
            facts: list[str] = []
            if mode == "oracle":
                for evidence_id in question_record.get("evidence") or []:
                    hit = by_id.get(str(evidence_id).strip())
                    if hit is None:
                        continue
                    session_id, turn = hit
                    speaker = str(turn.get("speaker") or "").strip()
                    text = str(turn.get("text") or "").strip()
                    if not text:
                        continue
                    body = f"{speaker}: {text}" if speaker else text
                    date = dates.get(session_id)
                    facts.append(f"[{date}] {body}" if date else body)
                if not facts:
                    continue
            cases.append(
                {
                    "question_id": f"{mode}/conv{conv_index + 1:02d}-qa{qa_index + 1:03d}",
                    "question": question,
                    "gold": gold,
                    "facts": facts,
                }
            )
    meta = {
        "path": DATA_REL,
        "sha256": digest,
        "sha256_verified": True,
        "conversations": len(dataset),
        "qa_total": qa_total,
        "cases_available": len(cases),
    }
    return cases, meta


def load_done(path: Path) -> set[str]:
    """Question IDs already checkpointed (any mode)."""
    done: set[str] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        question_id = row.get("question_id") if isinstance(row, dict) else None
        if isinstance(question_id, str) and question_id:
            done.add(question_id)
    return done


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()


def acquire_key(pool: KeyPool) -> str | None:
    """Wait briefly for an available pooled key; never fabricate one."""
    deadline = time.monotonic() + 60.0
    while True:
        key = pool.get()
        if key is not None:
            return key
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


def release_key(pool: KeyPool, key: str, error: str | None) -> None:
    """Release a pooled key; parse ONLY short error codes for cooldown state."""
    status_code: int | None = None
    if error:
        match = re.search(r"http_(\d{3})", error)
        if match:
            status_code = int(match.group(1))
    pool.release(key, success=error is None, status_code=status_code)


def run_case(case: dict[str, Any], mode: str, pool: KeyPool, path: Path) -> dict[str, Any]:
    """One case: reader -> judge -> append one raw-verdict checkpoint line."""
    question = case["question"]
    reader_key = acquire_key(pool)
    if reader_key is None:
        pred, reader_error = None, "reader_no_key_available"
    else:
        if mode == "no_memory":
            pred, reader_error = reader_answer(
                question, [], reader_key, system_prompt=NO_MEMORY_COT_PROMPT
            )
        else:
            pred, reader_error = reader_answer(question, case["facts"], reader_key)
        release_key(pool, reader_key, reader_error)

    judge_key = acquire_key(pool)
    if judge_key is None:
        verdict, judge_error = None, "judge_no_key_available"
    else:
        verdict, judge_error = judge_answer(
            question, case["gold"], pred or "", judge_key
        )
        release_key(pool, judge_key, judge_error)

    row = {
        "question_id": case["question_id"],
        "mode": mode,
        "pred": pred,
        "gold_hash": sha256_text(case["gold"])[:GOLD_HASH_CHARS],
        "verdict": verdict,
        "reader_model": DEFAULT_READER_MODEL,
        "judge_model": DEFAULT_JUDGE_MODEL,
        "reader_error": reader_error,
        "judge_error": judge_error,
        "observed_at": utc_now(),
    }
    append_checkpoint(path, row)
    return row


def denominators(path: Path, mode: str) -> dict[str, Any]:
    """Counts from the whole checkpoint for this mode, with explicit bases."""
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("mode") == mode:
                rows.append(row)
    attempted = len(rows)
    judged_rows = [row for row in rows if row.get("verdict") in (True, False)]
    correct = sum(1 for row in judged_rows if row.get("verdict") is True)
    judged = len(judged_rows)
    no_verdict = attempted - judged
    label = "no_memory_ceiling" if mode == "no_memory" else "oracle_ceiling"
    return {
        "attempted": attempted,
        "judged": judged,
        "correct": correct,
        "no_verdict": no_verdict,
        label: {
            "correct_over_judged": f"{correct}/{judged}" if judged else "0/0",
            "ratio": round(correct / judged, 6) if judged else None,
            "correct_over_attempted": f"{correct}/{attempted}" if attempted else "0/0",
            "ratio_all_attempts": round(correct / attempted, 6) if attempted else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=MODES, default="no_memory")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="cap on the first cases considered this run; 0 means all",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=0,
        help="limit the round-robin pool to the first N loaded keys; 0 means all",
    )
    parser.add_argument(
        "--keys-file", default=None,
        help="path to the KILO key env file (or set MEMNOTH_KEYS_FILE); a path, never a key value",
    )
    args = parser.parse_args(argv)
    mode = str(args.mode)
    if args.max_cases < 0 or args.max_keys < 0:
        parser.error("--max-cases and --max-keys must be non-negative")

    cases, meta = load_cases(mode)
    if args.max_cases:
        cases = cases[: args.max_cases]

    # Pass an explicit --keys-file straight to load_keys(). kilo_pool freezes
    # DEFAULT_KEYS_PATH at import time, so setting the env var here would be too
    # late; the explicit path is the robust route. Values are never disclosed.
    all_keys = load_keys(args.keys_file) if args.keys_file else load_keys()
    if not all_keys:
        raise RuntimeError("no KILO keys loaded; set KILO_API_KEY/KILOCODE_API_KEY or the env file")
    keys = all_keys[: args.max_keys] if args.max_keys else all_keys
    pool = KeyPool(keys)

    done = load_done(CHECKPOINT_PATH)
    pending = [case for case in cases if case["question_id"] not in done]
    run_attempts = 0
    for case in pending:
        run_case(case, mode, pool, CHECKPOINT_PATH)
        run_attempts += 1

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "observed_at": utc_now(),
        "diagnostic": "derived",
        "official": False,
        "note": NOTE,
        "mode": mode,
        "oracle_warning": (
            "Oracle lane uses the dataset evidence field; sanity ceiling only, "
            "NOT a product or memory result."
            if mode == "oracle"
            else "No-memory lane: reader sees the question only, no facts, no evidence, no gold."
        ),
        "dataset": meta,
        "models": {
            "reader": DEFAULT_READER_MODEL,
            "judge": DEFAULT_JUDGE_MODEL,
            "frozen_defaults_used": True,
        },
        "keys": {
            "count": len(keys),
            "source": "env-file KILO_KEY_* + KILO_API_KEY/KILOCODE_API_KEY (names/count only)",
            "round_robin": True,
            "values_disclosed": False,
        },
        "run": {
            "cases_considered": len(cases),
            "skipped_already_checkpointed": len(cases) - len(pending),
            "new_attempts": run_attempts,
        },
        "denominators": denominators(CHECKPOINT_PATH, mode),
        "checkpoint": {
            "path": str(CHECKPOINT_PATH),
            "append_only": True,
            "resumable_by": "question_id (mode-namespaced)",
            "fields": [
                "question_id", "pred", "gold_hash", "verdict",
                "reader_model", "judge_model",
            ],
            "gold_value_written": False,
            "gold_hash": f"sha256 prefix, {GOLD_HASH_CHARS} hex chars",
        },
        "limitations": [
            "Derived frozen-harness baseline; not an official LoCoMo score.",
            "No score is asserted as a product result; ceilings are counts with denominators.",
            "Judge UNSURE and transport failures are counted as no-verdict, never as correct.",
            "Oracle evidence is gold-linked by construction and is never a memory claim.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
