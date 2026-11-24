"""Pinned LongMemEval adapter: oracle variant, temporal-reasoning subset.

Two independent signals are reported:

1. Model-free retrieval metrics (Recall@k / NDCG@k) computed directly from the
   upstream ``answer_session_ids`` labels. No judge and no model is involved.
2. Answer accuracy via an injected ``judge_fn``. The judge model/version is
   pinned in :data:`JUDGE`; the adapter records the pin but cannot verify what
   the injected callable actually called, so ``enforced_by_adapter`` is False.

The slice is small and locally constructed (fixtures/pinned-ids.json). It is a
diagnostic, not an official LongMemEval split or score.
"""

from __future__ import annotations

import math
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
    )

LANE = "longmemeval"
SOURCE_URL = "https://github.com/xiaowu0162/LongMemEval"
PAPER_URL = "https://arxiv.org/abs/2410.10813"

# --------------------------------------------------------------------------- #
# PINS -- verified 2026-11-24 against the official upstream sources.
# Never guess a hash: compute it from the exact artifact you downloaded.
# --------------------------------------------------------------------------- #
# The official cleaned dataset is hosted on HuggingFace. `longmemeval_oracle.json`
# is the only released variant carrying the `answer_session_ids` labels this
# adapter reads; the `_s` and `_m` variants are not used here. The URL pins the
# exact HF dataset commit that serves the artifact, so content cannot drift
# with `main`. Provenance:
#   HF commits API:
#     https://huggingface.co/api/datasets/xiaowu0162/longmemeval-cleaned/commits/main
#   HF tree at the pinned commit (same `lfs.oid` as the SHA-256 below):
#     https://huggingface.co/api/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f
DATASET_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
    "98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json"
)
# SHA-256 of that exact file (15,388,478 bytes). Verified two independent ways:
#   (a) the HF tree API reports this value as `lfs.oid` for the file at the
#       pinned commit (Git LFS oid is the SHA-256 of the file contents);
#   (b) `Get-FileHash -Algorithm SHA256` on the single downloaded artifact.
# Both produced the digest below. No hash was invented.
DATASET_SHA256 = "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"
# Upstream GitHub repository HEAD at pin time (the source linked by SOURCE_URL),
# from `gh api repos/xiaowu0162/LongMemEval/commits/main --jq .sha`. This is a
# resolved commit, not a branch name, and is distinct from the HF dataset
# commit embedded in DATASET_URL above.
UPSTREAM_COMMIT = "9e0b455f4ef0e2ab8f2e582289761153549043fc"

VARIANT = "oracle"
QUESTION_TYPE_FILTER = "temporal-reasoning"
KS: tuple[int, ...] = (1, 3, 5, 10)

# Judge pin verified against the pinned upstream `src/evaluation/evaluate_qa.py`:
# `model_zoo['gpt-4o']` resolves to `gpt-4o-2024-08-06` and the completion kwargs
# set `temperature=0`. Upstream's README invokes `evaluate_qa.py gpt-4o ...`;
# `gpt-4o` is the alias that resolves to the pinned snapshot below.
JUDGE: dict[str, Any] = {
    "model": "gpt-4o-2024-08-06",
    "temperature": 0,
    "pinned": True,
    "enforced_by_adapter": False,
    "note": (
        "Pinned for comparability only. The caller must inject a judge_fn that "
        "actually uses this model/version and temperature; the adapter cannot "
        "verify the remote call."
    ),
}

_NOTE = (
    "Diagnostic LongMemEval oracle temporal-reasoning slice (30 questions by "
    "fixture). Not an official split, not an official score, and not comparable "
    "to published LongMemEval numbers unless the same data, labels, judge and "
    "prompt are used. Denominators count only rows actually supplied."
)


def prepare(local_dir: str | Path = "data/longmemeval") -> Path:
    """Download and SHA-256-verify the pinned oracle dataset into ``data/``.

    The URL and SHA-256 pins above are real; ``fetch_pinned`` still refuses
    placeholders or a byte mismatch. This is a network boundary and is never
    invoked at import time or by ``--smoke``.
    """
    dest = Path(local_dir) / "longmemeval_oracle.json"
    return fetch_pinned(DATASET_URL, DATASET_SHA256, dest)


def select(max_cases: int | None = None) -> dict[str, Any]:
    """Describe the deterministic pinned selection without downloading data.

    Counts come from ``fixtures/pinned-ids.json``. If the fixture later gains a
    literal ID array (``ids``), those IDs are used; otherwise IDs are resolved
    deterministically from the prepared dataset by filtering to
    ``question_type == temporal-reasoning`` and sorting by ``question_id``.
    """
    section, found = fixture_section(LANE)
    planned = int(section.get("count", 0)) if found else 0
    if planned < 0:
        raise ValueError("pinned count must be non-negative")

    ordinals = [position + 1 for position in limiter(range(planned), max_cases)]
    ids = pinned_ids(section, "ids", max_cases)
    return {
        "lane": LANE,
        "variant": VARIANT,
        "filter": {"question_type": QUESTION_TYPE_FILTER},
        "source": SOURCE_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "fixture_found": found,
        "planned_denominator": planned,
        "effective_denominator": denominator(ordinals),
        "ordinals": ordinals,
        "ids": ids,
        "ids_pinned": bool(ids),
        "id_resolution": (
            "sort filtered rows by question_id and take the first "
            "effective_denominator entries"
        ),
        "official_split": bool(section.get("official_split", False)),
        "note": _NOTE,
    }


def _dedupe(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    ordered: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in ordered:
            ordered.append(text)
    return ordered


def _ndcg_at_k(gold: list[str], ranked: list[str], k: int) -> float:
    gold_set = set(gold)
    dcg = sum(
        1.0 / math.log2(position + 2)
        for position, item in enumerate(ranked[:k])
        if item in gold_set
    )
    ideal = sum(1.0 / math.log2(position + 2) for position in range(min(len(gold_set), k)))
    return dcg / ideal if ideal else 0.0


def evaluate(
    results: list[Any],
    judge_fn: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate supplied result rows.

    Each row is a dict with ``question_id``, ``question_type``,
    ``answer_session_ids`` (gold labels) and ``retrieved_session_ids`` (ranked
    session IDs from the system under test). ``answer`` (model answer text) is
    used by ``judge_fn`` when injected.

    ``judge_fn(record) -> bool | {'correct': bool} | 'yes'/'no'``. Indeterminate
    verdicts are excluded from the judge denominator rather than guessed.

    Top-level ``denominator`` is the number of rows supplied. Top-level
    ``correct`` is the numerator of the primary metric (judge accuracy when a
    judge ran, otherwise a label-derived retrieval hit at the largest k).
    """
    source_rows = stable_rows(results or [])
    label_missing = 0
    labeled = 0
    recall_hits = 0
    k_totals: dict[int, dict[str, float]] = {
        k: {"recall_num": 0.0, "ndcg_sum": 0.0} for k in KS
    }
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

        gold = _dedupe(record.get("answer_session_ids"))
        ranked = _dedupe(record.get("retrieved_session_ids"))
        row: dict[str, Any] = {
            "index": index,
            "question_id": record.get("question_id") or record.get("id"),
            "question_type": record.get("question_type"),
            "gold_session_ids": gold,
            "retrieved_session_ids": ranked,
            "valid_label": bool(gold),
            "recall": {},
            "ndcg": {},
        }
        if not gold:
            label_missing += 1
            row["note"] = "no answer_session_ids label; excluded from recall/NDCG denominators"
        else:
            labeled += 1
            gold_set = set(gold)
            for k in KS:
                hits = len(gold_set.intersection(ranked[:k]))
                recall = hits / len(gold_set)
                ndcg = _ndcg_at_k(gold, ranked, k)
                row["recall"][f"@{k}"] = round(recall, 6)
                row["ndcg"][f"@{k}"] = round(ndcg, 6)
                k_totals[k]["recall_num"] += hits / len(gold_set)
                k_totals[k]["ndcg_sum"] += ndcg
            if len(gold_set.intersection(ranked[: max(KS)])) > 0:
                recall_hits += 1
        rows.append(row)

    judged = apply_judge(source_rows, judge_fn)
    for judge_row in judged["rows"]:
        position = judge_row["index"]
        if position < len(rows):
            rows[position]["judge_verdict"] = judge_row["verdict"]
            rows[position]["judge_reason"] = judge_row["reason"]

    primary_name = "judge_accuracy" if judged["ran"] else "retrieval_hit_at_k"
    primary_correct = judged["correct"] if judged["ran"] else recall_hits
    primary_denom = judged["denominator"] if judged["ran"] else labeled

    recall_at_k: dict[str, Any] = {}
    ndcg_at_k: dict[str, Any] = {}
    for k in KS:
        recall_at_k[f"@{k}"] = {
            "numerator_sum": round(k_totals[k]["recall_num"], 6),
            "denominator": labeled,
            "mean_recall": ratio(round(k_totals[k]["recall_num"], 6), labeled),
            "display": ratio_text(
                f"mean recall@{k}",
                round(k_totals[k]["recall_num"], 6),
                labeled,
                " (count-sum / labeled rows; diagnostic)",
            ),
        }
        ndcg_at_k[f"@{k}"] = {
            "sum": round(k_totals[k]["ndcg_sum"], 6),
            "denominator": labeled,
            "mean_ndcg": ratio(round(k_totals[k]["ndcg_sum"], 6), labeled),
        }

    metric_note = " (diagnostic; not an official score)"
    summary_parts = [
        ratio_text("rows supplied", len(source_rows), len(source_rows), ""),
        ratio_text("labeled rows", labeled, len(source_rows), ""),
        f"label missing: {label_missing}",
        ratio_text(primary_name, primary_correct, primary_denom, metric_note),
        recall_at_k["@5"]["display"],
        f"judge ran: {judged['ran']}",
    ]

    return {
        "lane": LANE,
        "variant": VARIANT,
        "question_type_filter": QUESTION_TYPE_FILTER,
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
                "display": ratio_text(primary_name, primary_correct, primary_denom, metric_note),
                "judge_free": not judged["ran"],
            },
            "recall_at_k": recall_at_k,
            "ndcg_at_k": ndcg_at_k,
            "judge": {
                "ran": judged["ran"],
                "correct": judged["correct"],
                "denominator": judged["denominator"],
                "indeterminate_excluded": (
                    len(source_rows) - judged["denominator"] if judged["ran"] else 0
                ),
                "note": judged["note"],
            },
            "labeled_rows": labeled,
            "label_missing": label_missing,
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
            "question_type": QUESTION_TYPE_FILTER,
            "answer_session_ids": ["s2", "s7"],
            "retrieved_session_ids": ["s7", "s9", "s2", "s1"],
            "answer": "on tuesday",
        },
        {
            "question_id": "q2",
            "question_type": QUESTION_TYPE_FILTER,
            "answer_session_ids": ["s4"],
            "retrieved_session_ids": ["s8", "s4"],
            "answer": "last week",
        },
        {
            "question_id": "q3",
            "question_type": QUESTION_TYPE_FILTER,
            "answer_session_ids": ["s1"],
            "retrieved_session_ids": ["s9"],
            "answer": "never",
        },
    ]
    verdicts = {"q1": True, "q2": False, "q3": False}
    result = evaluate(rows, lambda record: verdicts.get(record["question_id"]))
    assert result["official"] is False
    assert result["diagnostic"] == "derived"
    assert result["denominator"] == 3
    assert result["correct"] == 1
    assert result["metrics"]["labeled_rows"] == 3
    assert len(result["rows"]) == 3
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="offline synthetic smoke test")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("pass --smoke; dataset preparation is a verified network boundary")
    print(json.dumps(self_test(), indent=2, sort_keys=True))
