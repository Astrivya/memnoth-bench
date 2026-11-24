"""Pinned LoCoMo adapter: one conversation, five locally pinned questions.

The primary signal is deterministic token-level F1 (no judge, no model call).
An injected ``judge_fn`` can add semantic accuracy, but the judge model/version
is metadata only: the adapter records the pin and cannot verify the callable.

Token counts and latency are reported with their sample size and context, not
as a speed claim. They depend on hardware, provider, context length and caching
and are not comparable to published LoCoMo numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:  # normal package import
    from ..common import (
        apply_judge,
        denominator,
        fetch_pinned,
        fixture_section,
        limiter,
        pinned_ids,
        ratio,
        ratio_text,
        stamps,
        stable_rows,
        stats,
        token_f1,
    )
except (ImportError, ValueError):  # loaded directly by path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import (
        apply_judge,
        denominator,
        fetch_pinned,
        fixture_section,
        limiter,
        pinned_ids,
        ratio,
        ratio_text,
        stamps,
        stable_rows,
        stats,
        token_f1,
    )

LANE = "locomo"
SOURCE_URL = "https://github.com/snap-research/locomo"
CONVERSATIONS = 1
QUESTIONS = 5

# --------------------------------------------------------------------------- #
# PINS -- verified against the official snap-research/locomo repository.
# Provenance: gh api repos/snap-research/locomo/commits/main --jq .sha
# Hash: Get-FileHash -Algorithm SHA256 on a single download of that pinned URL.
# --------------------------------------------------------------------------- #
UPSTREAM_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
DATASET_URL = "https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json"
DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

JUDGE: dict[str, Any] = {
    "model": "gpt-4o-2024-08-06",
    "temperature": 0,
    "pinned": True,
    "enforced_by_adapter": False,
    "note": "Optional. Same pinned judge as the LongMemEval lane for comparability.",
}

_NOTE = (
    "Diagnostic LoCoMo slice: 1 conversation / 5 questions from "
    "fixtures/pinned-ids.json. F1 is deterministic and judge-free; judge "
    "accuracy, tokens and latency are optional/contextual. Not an official "
    "LoCoMo split, not an official score, and not a speed benchmark."
)


def prepare(local_dir: str | Path = "data/locomo") -> Path:
    """Download and SHA-256-verify the pinned conversation into ``data/``.

    Uses only the verified pins above and refuses unverified pins. Never
    called at import time or by ``--smoke``.
    """
    dest = Path(local_dir) / "locomo_conversation.json"
    return fetch_pinned(DATASET_URL, DATASET_SHA256, dest)


def select(max_cases: int | None = None) -> dict[str, Any]:
    """Describe the pinned 1-conversation / 5-question selection."""
    section, found = fixture_section(LANE)
    planned_conversations = int(section.get("conversations", CONVERSATIONS)) if found else CONVERSATIONS
    planned_questions = int(section.get("questions", QUESTIONS)) if found else QUESTIONS
    if planned_conversations < 0 or planned_questions < 0:
        raise ValueError("pinned counts must be non-negative")

    # Only one conversation is in scope; the question budget is the limiter.
    ordinals = [position + 1 for position in limiter(range(planned_questions), max_cases)]
    ids = pinned_ids(section, "ids", max_cases)
    return {
        "lane": LANE,
        "conversations": min(planned_conversations, 1),
        "planned_questions": planned_questions,
        "effective_denominator": denominator(ordinals),
        "ordinals": ordinals,
        "ids": ids,
        "ids_pinned": bool(ids),
        "source": SOURCE_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "fixture_found": found,
        "official_split": bool(section.get("official_split", False)),
        "note": _NOTE,
    }


def evaluate(
    results: list[Any],
    judge_fn: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate supplied rows with deterministic token F1.

    Each row is a dict with ``question_id``, ``gold_answer``, ``pred_answer``,
    and optionally ``tokens``, ``latency_ms`` and ``context_tokens``.
    ``judge_fn(record) -> bool | {'correct': bool} | 'yes'/'no'`` is optional.

    Top-level ``denominator`` is the number of rows supplied. Top-level
    ``correct`` is the number of rows whose F1 is exactly 1.0 (perfect token
    match) unless a judge ran, in which case it is the judge's correct count;
    ``primary_metric`` names which is used.
    """
    source_rows = stable_rows(results or [])
    f1_values: list[float] = []
    exact_token_matches = 0
    rows: list[dict[str, Any]] = []

    for index, record in enumerate(source_rows):
        if not isinstance(record, dict):
            rows.append(
                {
                    "index": index,
                    "valid": False,
                    "note": "row was not a dict; excluded from all denominators",
                }
            )
            continue
        score = token_f1(record.get("pred_answer"), record.get("gold_answer"))
        f1_values.append(score)
        if score == 1.0:
            exact_token_matches += 1
        rows.append(
            {
                "index": index,
                "question_id": record.get("question_id") or record.get("id"),
                "gold_answer": record.get("gold_answer"),
                "pred_answer": record.get("pred_answer"),
                "f1": score,
                "exact_token_match": score == 1.0,
                "tokens": record.get("tokens"),
                "latency_ms": record.get("latency_ms"),
                "context_tokens": record.get("context_tokens"),
                "valid": True,
            }
        )

    judged = apply_judge(source_rows, judge_fn)
    for judge_row in judged["rows"]:
        position = judge_row["index"]
        if position < len(rows):
            rows[position]["judge_verdict"] = judge_row["verdict"]
            rows[position]["judge_reason"] = judge_row["reason"]

    f1_denom = len(f1_values)
    f1_sum = round(sum(f1_values), 6)
    primary_name = "judge_accuracy" if judged["ran"] else "f1_exact_token_match"
    primary_correct = judged["correct"] if judged["ran"] else exact_token_matches
    primary_denom = judged["denominator"] if judged["ran"] else f1_denom

    note_suffix = " (diagnostic; not an official score)"
    context_stats = {
        "tokens": stats(record.get("tokens") for record in source_rows if isinstance(record, dict)),
        "latency_ms": stats(
            record.get("latency_ms") for record in source_rows if isinstance(record, dict)
        ),
        "context_tokens": stats(
            record.get("context_tokens") for record in source_rows if isinstance(record, dict)
        ),
        "note": (
            "Descriptive only, over this 5-question slice. Latency and token "
            "counts depend on hardware, provider, context length and caching; "
            "no speed or efficiency claim is made."
        ),
    }
    summary_parts = [
        ratio_text("rows supplied", len(source_rows), len(source_rows), ""),
        ratio_text(primary_name, primary_correct, primary_denom, note_suffix),
        f"sum F1 = {f1_sum} over n={f1_denom} rows, mean F1 = {ratio(f1_sum, f1_denom)}",
        f"judge ran: {judged['ran']}",
        f"tokens n={context_stats['tokens']['n']}, latency n={context_stats['latency_ms']['n']}",
    ]

    return {
        "lane": LANE,
        "denominator": denominator(source_rows),
        "correct": primary_correct,
        "primary_metric": primary_name,
        "primary_denominator": primary_denom,
        "rows": rows,
        "metrics": {
            "primary": {
                "name": primary_name,
                "correct": primary_correct,
                "denominator": primary_denom,
                "display": ratio_text(primary_name, primary_correct, primary_denom, note_suffix),
                "judge_free": not judged["ran"],
            },
            "f1": {
                "sum": f1_sum,
                "denominator": f1_denom,
                "mean": ratio(f1_sum, f1_denom),
                "exact_token_matches": exact_token_matches,
                "display": f"sum F1 {f1_sum}/{f1_denom} = {ratio(f1_sum, f1_denom)} (judge-free)",
            },
            "judge": {
                "ran": judged["ran"],
                "correct": judged["correct"],
                "denominator": judged["denominator"],
                "note": judged["note"],
            },
            "context": context_stats,
        },
        "summary": "; ".join(summary_parts),
        "source_url": SOURCE_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        **stamps(_NOTE, judge=JUDGE),
    }


def self_test() -> dict[str, Any]:
    """Offline smoke with synthetic rows. No network, no keys, no spend."""
    rows = [
        {
            "question_id": "q1",
            "gold_answer": "the blue notebook",
            "pred_answer": "the blue notebook",
            "tokens": 512,
            "latency_ms": 120,
            "context_tokens": 4096,
        },
        {
            "question_id": "q2",
            "gold_answer": "Paris",
            "pred_answer": "paris",
            "tokens": 498,
            "latency_ms": 133,
            "context_tokens": 4096,
        },
        {
            "question_id": "q3",
            "gold_answer": "Tuesday morning",
            "pred_answer": "Wednesday evening",
            "tokens": 530,
            "latency_ms": 118,
            "context_tokens": 8192,
        },
    ]
    verdicts = {"q1": True, "q2": False, "q3": False}
    result = evaluate(rows, lambda record: verdicts.get(record["question_id"]))
    assert result["official"] is False
    assert result["diagnostic"] == "derived"
    assert result["denominator"] == 3
    assert result["metrics"]["f1"]["exact_token_matches"] == 2  # token F1 is case-insensitive
    assert result["metrics"]["context"]["tokens"]["n"] == 3
    assert len(result["rows"]) == 3
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="offline synthetic smoke test")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("pass --smoke; dataset preparation is a separate network-enabled step")
    print(json.dumps(self_test(), indent=2, sort_keys=True))
