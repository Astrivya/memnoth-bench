"""Pinned MemoryAgentBench adapter -- FactConsolidation (highest-ROI lane).

This is the most complete of the three harness adapters because it carries the
structured conflict-resolution check that matters most for a bi-temporal vault:

* Primary metric: **structured serial-number exact match**. Each case is built
  from an MQUAKE true + rewrite pair; the gold serial is the serial belonging
  to the newer rewrite fact. A system that paraphrases, hedges, or keeps the
  superseded serial fails. No judge and no fuzzy matching are involved.
* Secondary metric: plain answer exact match after whitespace/case
  normalization. This is reported second, never promoted over the primary.
* Optional judge: an injected ``judge_fn`` can add a semantic verdict, but it
  is kept separate from the two exact-match metrics and its pin is metadata
  only (the adapter cannot verify the remote call).

Both subsets (SH = 24, MH = 16) run at 4096 and 8192 context tokens. Every
grouping reports its own denominator, and invalid rows are excluded with an
explicit count rather than silently dropped or padded.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

try:  # normal package import
    from ..common import (
        PIN_BEFORE_USE,
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
        PIN_BEFORE_USE,
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

LANE = "memoryagentbench"
SOURCE_URL = "https://github.com/Yukun-Cui/MemoryAgentBench"
MQUAKE_URL = "https://github.com/stanfordnlp/MQUAKE"
TASK = "FactConsolidation"
SUBSETS: tuple[str, ...] = ("SH", "MH")
CONTEXT_TOKENS: tuple[int, ...] = (4096, 8192)
VALID_UPDATE_TYPES: tuple[str, ...] = ("true", "rewrite")

# --------------------------------------------------------------------------- #
# PINS -- replace every PIN_BEFORE_USE with a verified value before use.
# Never invent a hash: compute it from the exact artifact you downloaded.
# --------------------------------------------------------------------------- #
DATASET_URL = PIN_BEFORE_USE  # PIN_BEFORE_USE: pinned FactConsolidation artifact URL
DATASET_SHA256 = PIN_BEFORE_USE  # PIN_BEFORE_USE: SHA-256 of that exact artifact
UPSTREAM_COMMIT = PIN_BEFORE_USE  # PIN_BEFORE_USE: upstream git commit for the harness
MQUAKE_COMMIT = PIN_BEFORE_USE  # PIN_BEFORE_USE: upstream git commit for MQUAKE pairs

JUDGE: dict[str, Any] = {
    "model": "gpt-4o-2024-08-06",
    "temperature": 0,
    "pinned": True,
    "enforced_by_adapter": False,
    "note": (
        "Optional semantic verdict only. Primary and secondary metrics are "
        "judge-free exact matches. The caller must inject a matching judge_fn; "
        "the adapter cannot verify the remote call."
    ),
}

_NOTE = (
    "Diagnostic MemoryAgentBench FactConsolidation slice: SH 24 + MH 16 cases "
    "at 4096/8192 tokens from MQUAKE true+rewrite pairs (fixtures/pinned-ids.json). "
    "Primary metric is structured serial-number exact match; plain accuracy is "
    "secondary. Not an official split, not an official score. Denominators are "
    "the rows actually supplied, never the planned fixture counts."
)


def prepare(local_dir: str | Path = "data/memoryagentbench") -> Path:
    """Download and SHA-256-verify the pinned FactConsolidation artifact.

    The upstream build combines MQUAKE true + rewrite pairs; the pairs are not
    fetched separately here so there is a single pinned, checksummed artifact.
    Refuses to run while pins are placeholders. Never called at import time or
    by ``--smoke``.
    """
    dest = Path(local_dir) / "fact_consolidation.json"
    return fetch_pinned(DATASET_URL, DATASET_SHA256, dest)


def select(max_cases: int | None = None) -> dict[str, Any]:
    """Describe the deterministic pinned SH/MH x context-token selection.

    Case IDs come from an optional literal ID array in the fixture when one is
    present (``fact_consolidation_sh_ids`` / ``fact_consolidation_mh_ids``).
    Otherwise the prepared dataset is filtered to FactConsolidation and sorted
    deterministically by ``(subset, case_id)`` before slicing. The slice is
    capped by ``max_cases`` across the whole cross-product; it is never padded.
    """
    section, found = fixture_section(LANE)
    counts = {
        "SH": int(section.get("fact_consolidation_sh", 24)) if found else 24,
        "MH": int(section.get("fact_consolidation_mh", 16)) if found else 16,
    }
    contexts = list(section.get("context_tokens", list(CONTEXT_TOKENS))) if found else list(CONTEXT_TOKENS)
    if any(value < 0 for value in counts.values()):
        raise ValueError("pinned subset counts must be non-negative")
    if not contexts:
        raise ValueError("at least one context-token setting is required")

    configurations: list[dict[str, Any]] = []
    for subset in SUBSETS:
        for context in contexts:
            for position in range(counts[subset]):
                configurations.append(
                    {
                        "subset": subset,
                        "context_tokens": int(context),
                        "ordinal": position + 1,
                    }
                )
    configurations = limiter(configurations, max_cases)

    ids_by_subset = {
        "SH": pinned_ids(section, "fact_consolidation_sh_ids", None),
        "MH": pinned_ids(section, "fact_consolidation_mh_ids", None),
    }
    planned = sum(counts.values()) * len(contexts)
    return {
        "lane": LANE,
        "task": TASK,
        "source": SOURCE_URL,
        "mquake_source": MQUAKE_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "mquake_commit": MQUAKE_COMMIT,
        "fixture_found": found,
        "counts": counts,
        "context_tokens": [int(value) for value in contexts],
        "planned_denominator": planned,
        "effective_denominator": denominator(configurations),
        "configurations": configurations,
        "ids_by_subset": ids_by_subset,
        "ids_pinned": any(ids_by_subset.values()),
        "id_resolution": (
            "filter to FactConsolidation, sort by (subset, case_id), then take "
            "the configurations listed above"
        ),
        "official_split": bool(section.get("official_split", False)),
        "note": _NOTE,
    }


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #


def _coerce_context(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate(record: dict[str, Any]) -> list[str]:
    """Return validation errors; a row with any error is excluded explicitly."""
    errors: list[str] = []
    subset = str(record.get("subset", "")).upper()
    if subset not in SUBSETS:
        errors.append(f"subset must be one of {SUBSETS}, got {record.get('subset')!r}")
    if _coerce_context(record.get("context_tokens")) not in CONTEXT_TOKENS:
        errors.append(f"context_tokens must be one of {CONTEXT_TOKENS}, got {record.get('context_tokens')!r}")
    if record.get("gold_serial") is None or record.get("pred_serial") is None:
        errors.append("gold_serial and pred_serial are both required for the primary metric")
    return errors


def serial_exact_match(gold: Any, predicted: Any) -> bool:
    """Primary metric: strict exact match after ``str().strip()`` only.

    No case folding, separator removal, substring matching or fuzzy credit.
    This intentionally rejects a correct serial wrapped in prose.
    """
    if gold is None or predicted is None:
        return False
    return str(gold).strip() == str(predicted).strip()


def canonical_serial(value: Any) -> str | None:
    """Diagnostic-only canonical form; never used for the primary metric."""
    if value is None:
        return None
    text = re.sub(r"[\s_\-]+", "", str(value).strip().lower())
    return text or None


def normalized_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).casefold()
    return text or None


def answer_exact_match(gold: Any, predicted: Any) -> bool:
    """Secondary metric: whitespace/case-normalized exact match."""
    gold_text = normalized_answer(gold)
    pred_text = normalized_answer(predicted)
    return gold_text is not None and pred_text is not None and gold_text == pred_text


def _group(gold_rows: list[dict[str, Any]], *, subset: str | None, context: int | None) -> dict[str, Any]:
    selected = [
        row
        for row in gold_rows
        if (subset is None or row["subset"] == subset)
        and (context is None or row["context"] == context)
    ]
    serial_correct = sum(1 for row in selected if row["serial_exact"])
    answer_correct = sum(1 for row in selected if row["answer_exact"])
    n = len(selected)
    return {
        "denominator": n,
        "serial_exact_match": {
            "correct": serial_correct,
            "denominator": n,
            "display": ratio_text("serial exact", serial_correct, n, " (primary, judge-free)"),
        },
        "answer_exact_match": {
            "correct": answer_correct,
            "denominator": n,
            "display": ratio_text("answer exact", answer_correct, n, " (secondary, judge-free)"),
        },
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate(
    results: list[Any],
    judge_fn: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Evaluate supplied FactConsolidation rows.

    Each row is a dict with ``case_id``, ``subset`` (``SH``/``MH``),
    ``context_tokens`` (4096/8192), ``update_type`` (``true``/``rewrite``),
    ``gold_serial``, ``pred_serial``, and optionally ``gold_answer`` /
    ``pred_answer``. Invalid rows are reported and excluded from denominators.

    Top-level ``denominator`` is the number of *valid* rows; ``correct`` is the
    number of serial-number exact matches (the primary metric). Planned fixture
    counts are echoed as non-authoritative context only.
    """
    source_rows = stable_rows(results or [])
    rows: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []
    valid_outputs: list[dict[str, Any]] = []
    invalid_rows = 0

    for index, record in enumerate(source_rows):
        if not isinstance(record, dict):
            invalid_rows += 1
            rows.append(
                {
                    "index": index,
                    "valid": False,
                    "validation_errors": ["row was not a dict"],
                }
            )
            continue
        errors = validate(record)
        if errors:
            invalid_rows += 1
            rows.append(
                {
                    "index": index,
                    "case_id": record.get("case_id"),
                    "valid": False,
                    "validation_errors": errors,
                }
            )
            continue

        subset = str(record["subset"]).upper()
        context = _coerce_context(record["context_tokens"])
        assert context is not None  # guaranteed by validate()

        serial_hit = serial_exact_match(record.get("gold_serial"), record.get("pred_serial"))
        answer_hit = answer_exact_match(record.get("gold_answer"), record.get("pred_answer"))
        output: dict[str, Any] = {
            "index": index,
            "case_id": record.get("case_id"),
            "subset": subset,
            "context_tokens": context,
            "update_type": record.get("update_type"),
            "gold_serial": record.get("gold_serial"),
            "pred_serial": record.get("pred_serial"),
            "serial_exact": serial_hit,
            "serial_canonical_match": (
                canonical_serial(record.get("gold_serial"))
                == canonical_serial(record.get("pred_serial"))
                and canonical_serial(record.get("gold_serial")) is not None
            ),
            "gold_answer": record.get("gold_answer"),
            "pred_answer": record.get("pred_answer"),
            "answer_exact": answer_hit,
            "valid": True,
        }
        rows.append(output)
        valid_records.append(record)
        valid_outputs.append(output)

    gold_rows: list[dict[str, Any]] = [
        {
            "subset": row["subset"],
            "context": row["context_tokens"],
            "serial_exact": row["serial_exact"],
            "answer_exact": row["answer_exact"],
        }
        for row in valid_outputs
    ]

    judged = apply_judge(valid_records, judge_fn, key="case_id")
    for judge_row in judged["rows"]:
        position = judge_row["index"]
        if position < len(valid_outputs):
            valid_outputs[position]["judge_verdict"] = judge_row["verdict"]
            valid_outputs[position]["judge_reason"] = judge_row["reason"]

    n = len(gold_rows)
    serial_correct = sum(1 for row in gold_rows if row["serial_exact"])
    answer_correct = sum(1 for row in gold_rows if row["answer_exact"])

    by_subset = {subset: _group(gold_rows, subset=subset, context=None) for subset in SUBSETS}
    by_context = {
        str(context): _group(gold_rows, subset=None, context=context) for context in CONTEXT_TOKENS
    }
    by_group = {
        f"{subset}@{context}": _group(gold_rows, subset=subset, context=context)
        for subset in SUBSETS
        for context in CONTEXT_TOKENS
    }

    section, found = fixture_section(LANE)
    planned_sh = int(section.get("fact_consolidation_sh", 24)) if found else 24
    planned_mh = int(section.get("fact_consolidation_mh", 16)) if found else 16
    planned_contexts = list(section.get("context_tokens", list(CONTEXT_TOKENS))) if found else list(CONTEXT_TOKENS)
    planned_denominators = {
        "SH": planned_sh,
        "MH": planned_mh,
        "context_tokens": [int(value) for value in planned_contexts],
        "planned_total": (planned_sh + planned_mh) * len(planned_contexts),
        "observed_valid_denominator": n,
        "authoritative": False,
        "note": (
            "Planned counts are context from the pinned fixture only. The observed "
            "valid denominator is authoritative; no denominator is inflated to "
            "match the plan."
        ),
    }

    update_types = sorted(
        {
            str(record.get("update_type"))
            for record in valid_records
            if record.get("update_type") is not None
        }
    )
    conflict_guardrail = {
        "primary_metric": "structured_serial_exact_match",
        "description": (
            "The gold serial is the serial of the newer MQUAKE rewrite fact. "
            "Returning the superseded true-fact serial, a paraphrase, or prose "
            "instead of the structured value all fail exact match."
        ),
        "fuzzy_matching": False,
        "normalization": "str().strip() only; no case folding or separator removal",
        "update_types_observed": update_types,
        "provenance_check": (
            "The adapter can only check the supplied gold_serial label. It "
            "cannot verify offline that the label really came from the pinned "
            "MQUAKE true+rewrite pair; prepare() must use the pinned artifact."
        ),
    }

    metric_note = " (diagnostic; not an official score)"
    summary_parts = [
        ratio_text("rows supplied", len(source_rows), len(source_rows), ""),
        f"invalid rows excluded: {invalid_rows}",
        ratio_text("serial exact (primary)", serial_correct, n, metric_note),
        ratio_text("answer exact (secondary)", answer_correct, n, metric_note),
        ratio_text("serial exact SH", by_subset["SH"]["serial_exact_match"]["correct"], by_subset["SH"]["denominator"], ""),
        ratio_text("serial exact MH", by_subset["MH"]["serial_exact_match"]["correct"], by_subset["MH"]["denominator"], ""),
        f"judge ran: {judged['ran']}",
    ]

    return {
        "lane": LANE,
        "task": TASK,
        "denominator": n,
        "correct": serial_correct,
        "primary_metric": "structured_serial_exact_match",
        "received_rows": len(source_rows),
        "invalid_rows": invalid_rows,
        "rows": rows,
        "metrics": {
            "primary": {
                "name": "structured_serial_exact_match",
                "correct": serial_correct,
                "denominator": n,
                "display": ratio_text("serial exact", serial_correct, n, metric_note),
                "judge_free": True,
                "by_subset": by_subset,
                "by_context_tokens": by_context,
                "by_subset_and_context": by_group,
            },
            "secondary": {
                "name": "plain_answer_exact_match",
                "correct": answer_correct,
                "denominator": n,
                "display": ratio_text("answer exact", answer_correct, n, metric_note),
                "judge_free": True,
                "by_subset": {
                    subset: by_subset[subset]["answer_exact_match"] for subset in SUBSETS
                },
                "by_context_tokens": {
                    context: by_context[context]["answer_exact_match"] for context in by_context
                },
            },
            "judge": {
                "ran": judged["ran"],
                "correct": judged["correct"],
                "denominator": judged["denominator"],
                "note": judged["note"],
            },
            "context_effect": {
                "descriptive": by_context,
                "note": (
                    "4096 vs 8192 token rows are descriptive counts on this "
                    "local slice; any difference is not a causal context-length "
                    "claim and is not comparable to published results."
                ),
            },
            "conflict_guardrail": conflict_guardrail,
            "planned_denominators": planned_denominators,
        },
        "summary": "; ".join(summary_parts),
        "source_url": SOURCE_URL,
        "mquake_url": MQUAKE_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "mquake_commit": MQUAKE_COMMIT,
        **stamps(_NOTE, judge=JUDGE),
    }


def self_test() -> dict[str, Any]:
    """Offline smoke with synthetic SH/MH x 4096/8192 rows. No network/spend."""
    rows = [
        {
            "case_id": "fc-sh-1",
            "subset": "SH",
            "context_tokens": 4096,
            "update_type": "rewrite",
            "gold_serial": "SR-1042",
            "pred_serial": "SR-1042",
            "gold_answer": "Ada Lovelace",
            "pred_answer": "ada lovelace",
        },
        {
            "case_id": "fc-sh-2",
            "subset": "SH",
            "context_tokens": 8192,
            "update_type": "rewrite",
            "gold_serial": "SR-1043",
            "pred_serial": "SR-1042",
            "gold_answer": "Grace Hopper",
            "pred_answer": "Grace Murray",
        },
        {
            "case_id": "fc-mh-1",
            "subset": "MH",
            "context_tokens": 4096,
            "update_type": "rewrite",
            "gold_serial": "SR-2051",
            "pred_serial": "SR-2051",
            "gold_answer": "Alan Turing",
            "pred_answer": "Alan Turing",
        },
        {
            "case_id": "fc-mh-2",
            "subset": "MH",
            "context_tokens": 8192,
            "update_type": "true",
            "gold_serial": "SR-2052",
            "pred_serial": "serial SR-2052",
            "gold_answer": "Katherine Johnson",
            "pred_answer": "Katherine Johnson",
        },
        {
            "case_id": "fc-bad",
            "subset": "XX",
            "context_tokens": 999,
            "gold_serial": "SR-1",
            "pred_serial": "SR-1",
        },
    ]
    verdicts = {"fc-sh-1": True, "fc-sh-2": False, "fc-mh-1": True, "fc-mh-2": False}
    result = evaluate(rows, lambda record: verdicts.get(record["case_id"]))
    assert result["official"] is False
    assert result["diagnostic"] == "derived"
    assert result["received_rows"] == 5
    assert result["invalid_rows"] == 1
    assert result["denominator"] == 4
    assert result["correct"] == 2
    assert result["metrics"]["primary"]["by_subset"]["SH"]["denominator"] == 2
    assert result["metrics"]["primary"]["by_subset"]["MH"]["denominator"] == 2
    assert result["metrics"]["secondary"]["correct"] == 3
    assert result["metrics"]["planned_denominators"]["authoritative"] is False
    assert len(result["rows"]) == 5
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="offline synthetic smoke test")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("pass --smoke; dataset preparation requires real PIN_BEFORE_USE values")
    print(json.dumps(self_test(), indent=2, sort_keys=True))
