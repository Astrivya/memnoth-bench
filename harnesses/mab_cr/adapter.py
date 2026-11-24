"""Official-style MemoryAgentBench Conflict_Resolution SubEM lane (judge-free).

This is a NEW lane. It is deliberately separate from ``harnesses/memoryagentbench``,
whose serial-number exact match is a DERIVED gate and is NOT the published metric.

What this adapter provides
--------------------------
* ``prepare()``      - download + SHA-256-verify the pinned Conflict_Resolution
                       parquet into D:\\memnoth-bench-data (outside the repo).
* ``build_rows()``   - turn each pinned parquet row into one case per question:
                       numbered facts (verbatim text, serial in the entity name),
                       a natural-language question, and short gold answers.
* scorer             - the published ``substring_exact_match`` from the pinned
                       upstream commit, including ``normalize_answer`` and the
                       official ``default_post_process`` max(raw, parsed) rule.
* ``evaluate()``     - SubEM denominators by subset (SH/MH) and context length
                       (6k/32k/64k/262k). No judge is called anywhere.

HONESTY CONTRACT
----------------
Every result is stamped ``official: false`` even though the *scoring function*
is the published one: the reader, prompts, vault seeding, and run envelope are
ours, not the benchmark's. ``diagnostic: derived``, ``comparability_tier: C``,
``score_asserted: false``. No rank, no win, no comparison to a published row.
Gold is used only for scoring, never seeded into the vault or sent to a reader.

PINS (public provenance only; verified 2026-11-24)
--------------------------------------------------
HF dataset revision: ``7ea066982b140a19337e17e60d45d4076e042faf``
  https://huggingface.co/api/datasets/ai-hyz/MemoryAgentBench/revision/7ea066982b140a19337e17e60d45d4076e042faf
CR parquet: 1,491,588 bytes, SHA-256 below (HF Git-LFS oid == recomputed bytes)
  https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/7ea066982b140a19337e17e60d45d4076e042faf/data/Conflict_Resolution-00000-of-00001.parquet?download=true
  https://huggingface.co/api/datasets/ai-hyz/MemoryAgentBench/tree/7ea066982b140a19337e17e60d45d4076e042faf/data
Upstream code commit: ``538026089d1a8a8eff05121d0db89b388f360eba``
  https://github.com/HUST-AI-HYZ/MemoryAgentBench/commit/538026089d1a8a8eff05121d0db89b388f360eba
Metric source (pinned commit): ``utils/eval_other_utils.py`` -- normalize_answer,
  substring_exact_match_score, drqa_metric_max_over_ground_truths, parse_output,
  default_post_process. See the README note in the official repo:
  https://github.com/HUST-AI-HYZ/MemoryAgentBench/blob/538026089d1a8a8eff05121d0db89b388f360eba/README.md
Paper: https://arxiv.org/abs/2507.05257 (Table 5, Table 6, Appendix B.4.2).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
import string
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

LANE = "mab-cr-subem"

DATASET_FILE = "Conflict_Resolution-00000-of-00001.parquet"
DATASET_URL = (
    "https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/"
    "7ea066982b140a19337e17e60d45d4076e042faf/"
    "data/Conflict_Resolution-00000-of-00001.parquet?download=true"
)
DATASET_SHA256 = "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45"
DATASET_BYTES = 1_491_588
HF_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
UPSTREAM_COMMIT = "538026089d1a8a8eff05121d0db89b388f360eba"
METRIC = "substring_exact_match"
PARSE_PATH = "default_post_process_max_raw_and_parsed"

# Bench data lives outside the checkout. Hard storage cap for the whole tree.
DATA_ROOT = Path(r"D:\memnoth-bench-data")
DATASET_DIR = DATA_ROOT / "datasets" / "mab"
STORAGE_CAP_BYTES = 80 * 1024**3  # 80 GiB hard cap

SUBSETS: tuple[str, ...] = ("SH", "MH")
CONTEXTS: tuple[str, ...] = ("6k", "32k", "64k", "262k")
SOURCE_RE = re.compile(r"^factconsolidation_(sh|mh)_(6k|32k|64k|262k)$")
FACT_RE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$", re.MULTILINE)
SERIAL_LEAK_RE = re.compile(r"^\d{1,6}\.\s+\S")
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_ANSWER_PREFIX = re.compile(r"Answer:", re.IGNORECASE)

NOTE = (
    "Official-style MemoryAgentBench Conflict_Resolution SubEM on the pinned "
    "dataset. The scorer is the published substring_exact_match (judge-free), "
    "but the reader, prompts, vault seeding and envelope are ours. "
    "Not an official benchmark result; no rank and no score asserted."
)


# --------------------------------------------------------------------------- #
# Storage / download boundary (no network at import time)
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_size_bytes(root: Path) -> int:
    """Best-effort total bytes under ``root``; unreadable entries are skipped."""
    total = 0
    if not root.is_dir():
        return 0
    for base, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except OSError:
                continue
    return total


def _check_storage_headroom(dest: Path, incoming_bytes: int) -> dict[str, Any]:
    """Refuse a download that would cross the 80 GiB bench-data cap."""
    used = directory_size_bytes(DATA_ROOT)
    if dest.is_file():
        incoming_bytes = 0  # a verified cache hit adds nothing
    if used + incoming_bytes > STORAGE_CAP_BYTES:
        raise RuntimeError(
            f"bench-data cap exceeded: {used + incoming_bytes} bytes > {STORAGE_CAP_BYTES}"
        )
    return {"used_bytes": used, "cap_bytes": STORAGE_CAP_BYTES, "incoming_bytes": incoming_bytes}


def _validate_pins() -> None:
    if not _HEX64.fullmatch(str(DATASET_SHA256).strip()):
        raise ValueError("dataset SHA-256 pin must be 64 hex characters")
    if urlparse(DATASET_URL).scheme.lower() not in {"http", "https"}:
        raise ValueError("dataset URL must be http(s)")


def prepare(
    local_dir: str | Path | None = None,
    *,
    local_cache: str | Path | None = None,
) -> Path:
    """Download and SHA-256-verify the pinned CR parquet outside the repo.

    ``local_cache`` may point to an already-downloaded copy of the SAME pinned
    artifact (verified by SHA-256 before it is copied). It removes the network
    requirement for offline dry runs; it is never a source of unverified bytes.
    """
    _validate_pins()
    dest = Path(local_dir).expanduser() if local_dir is not None else DATASET_DIR
    dest = (dest / DATASET_FILE) if dest.is_dir() or not str(dest).endswith(DATASET_FILE) else dest
    dest = dest.resolve()
    root = DATA_ROOT.resolve()
    try:
        dest.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"dataset destination must stay under {root}: {dest}") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    pin = DATASET_SHA256.strip().lower()

    if dest.is_file():
        if dest.stat().st_size == DATASET_BYTES and hmac.compare_digest(sha256_file(dest), pin):
            _check_storage_headroom(dest, 0)
            return dest
        dest.unlink()  # stale cache: refetch under the same immutable pin

    headroom = _check_storage_headroom(dest, DATASET_BYTES)

    if local_cache is not None:
        cached = Path(local_cache).expanduser().resolve()
        if cached.is_file() and cached.stat().st_size == DATASET_BYTES \
                and hmac.compare_digest(sha256_file(cached), pin):
            temp = dest.with_name(f"{dest.name}.part-{os.getpid()}")
            try:
                shutil.copyfile(cached, temp)
                if not hmac.compare_digest(sha256_file(temp), pin):
                    raise RuntimeError("local cache copy failed SHA-256 verification")
                os.replace(temp, dest)
            finally:
                try:
                    temp.unlink()
                except FileNotFoundError:
                    pass
            return dest

    temp = dest.with_name(f"{dest.name}.part-{os.getpid()}")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=120) as response, temp.open("wb") as output:
            total = 0
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                output.write(block)
                total += len(block)
        if total != DATASET_BYTES:
            raise RuntimeError(f"byte-count mismatch: expected {DATASET_BYTES}, got {total}")
        if not hmac.compare_digest(sha256_file(temp), pin):
            raise RuntimeError("SHA-256 mismatch for pinned CR parquet; refusing unverified bytes")
        os.replace(temp, dest)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    # Keep the headroom value referenced so the cap check cannot be optimised away.
    assert headroom["cap_bytes"] == STORAGE_CAP_BYTES
    return dest


# --------------------------------------------------------------------------- #
# Row construction (parquet -> cases)
# --------------------------------------------------------------------------- #


def parse_numbered_facts(context: Any) -> list[dict[str, Any]]:
    """Parse the ``0. fact`` lines; keep fact text verbatim, serial separate."""
    text = "" if context is None else str(context)
    facts = [
        {"serial": int(serial), "text": body.strip()}
        for serial, body in FACT_RE.findall(text)
        if body.strip()
    ]
    serials = [fact["serial"] for fact in facts]
    if serials != list(range(len(serials))):
        raise ValueError("pinned context does not contain contiguous 0-based numbered facts")
    return facts


def _flatten_golds(answers: Any) -> list[str]:
    """Flatten the parquet's list-of-lists answers into strings (official shape)."""
    if isinstance(answers, str):
        return [answers]
    if not isinstance(answers, (list, tuple)):
        return []
    flat: list[str] = []
    for item in answers:
        if isinstance(item, str):
            flat.append(item)
        elif isinstance(item, (list, tuple)):
            flat.extend(str(value) for value in item)
    return [value for value in flat if str(value).strip()]


def build_rows(
    dataset_path: str | Path,
    subset: str = "all",
    context: str = "all",
    max_cases: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialise one case per question from the pinned parquet.

    Selection is a deterministic filter + slice. It never pads, samples or
    invents a case, so the reported denominator is the number of cases returned.
    """
    subset = str(subset).lower()
    context = str(context).lower()
    if subset not in {"all", *[s.lower() for s in SUBSETS]}:
        raise ValueError(f"--subset must be all/sh/mh, got {subset!r}")
    if context not in {"all", *CONTEXTS}:
        raise ValueError(f"--context must be all/6k/32k/64k/262k, got {context!r}")
    if int(max_cases) < 0:
        raise ValueError("max_cases must be >= 0")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pyarrow is required to read the pinned parquet") from exc

    table = pq.read_table(str(dataset_path))
    required = {"context", "questions", "answers", "metadata"}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"pinned parquet missing columns: {sorted(missing)}")

    cases: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for row_index in range(table.num_rows):
        row = {name: table[name][row_index].as_py() for name in table.column_names}
        metadata = row.get("metadata") or {}
        source = str(metadata.get("source") or "")
        match = SOURCE_RE.match(source)
        if not match:
            continue
        sub, ctx = match.group(1).upper(), match.group(2)
        if subset != "all" and sub.lower() != subset:
            continue
        if context != "all" and ctx != context:
            continue

        facts = parse_numbered_facts(row.get("context"))
        questions = list(row.get("questions") or [])
        answers = list(row.get("answers") or [])
        if len(questions) != len(answers):
            raise ValueError(f"{source}: questions/answers length mismatch")
        sources.append(
            {
                "source": source,
                "subset": sub,
                "context": ctx,
                "facts": len(facts),
                "questions": len(questions),
            }
        )
        for index, question in enumerate(questions):
            golds = _flatten_golds(answers[index])
            cases.append(
                {
                    "case_id": f"mab-cr-{sub.lower()}-{ctx}-{index:04d}",
                    "subset": sub,
                    "context": ctx,
                    "source": source,
                    "question": str(question),
                    "answers": golds,
                    "facts": facts,
                    "qa_pair_id": str((metadata.get("qa_pair_ids") or [None] * len(questions))[index])
                    if metadata.get("qa_pair_ids") else None,
                }
            )

    cases.sort(key=lambda case: (case["subset"], CONTEXTS.index(case["context"]), case["case_id"]))
    sources.sort(key=lambda item: (item["subset"], CONTEXTS.index(item["context"])))
    if max_cases:
        cases = cases[:int(max_cases)]

    plan = {
        "lane": LANE,
        "dataset": DATASET_FILE,
        "dataset_sha256": DATASET_SHA256,
        "revision": HF_REVISION,
        "upstream_commit": UPSTREAM_COMMIT,
        "metric": METRIC,
        "selection": {"subset": subset, "context": context, "max_cases": max_cases},
        "sources": sources,
        "selected_cases": len(cases),
        "source_rows_matched": len(sources),
        "expected_full_denominator": 800,
        "expected_full_denominator_note": (
            "8 pinned rows x 100 questions = 800; the observed denominator is "
            "always len(selected_cases), never the expectation."
        ),
    }
    return cases, plan


# --------------------------------------------------------------------------- #
# Official scorer (verbatim semantics from the pinned commit)
# --------------------------------------------------------------------------- #


def normalize_answer(value: Any) -> str:
    """Official normalize_answer: lower -> strip ASCII punctuation -> articles -> ws."""
    text = "" if value is None else str(value)
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_output(output_text: Any, answer_prefix: str = "Answer:") -> str | None:
    """Official parse_output: first Answer: line, else first line."""
    text = "" if output_text is None else str(output_text)
    pattern = re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return re.sub(f"^{re.escape(answer_prefix)}", "", match[1].strip(), flags=re.IGNORECASE).strip()
    first = re.compile(r"(?:^)(.*)(?:\n|$)").search(text)
    return first[1].strip() if first else None


def substring_exact_match_score(prediction: Any, ground_truth: Any) -> bool:
    """Official SubEM: normalized gold contained in normalized prediction."""
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def _flatten_gold_scores(golds: Any) -> list[Any]:
    if isinstance(golds, str):
        return [golds]
    if not golds or not isinstance(golds, (list, tuple)):
        return []
    if isinstance(golds[0], (list, tuple)):
        return [value for part in golds for value in part]
    return list(golds)


def subem_max_over_golds(prediction: Any, golds: Any) -> bool:
    """Official max-over-ground-truths for SubEM (empty gold list -> False)."""
    items = _flatten_gold_scores(golds)
    if not items:
        return False
    return max(substring_exact_match_score(prediction, gold) for gold in items)


def score_subem(prediction: Any, golds: Any) -> bool:
    """Official default_post_process for factconsolidation_* rows.

    CR is not a special-cased dataset in the pinned ``post_process``; it falls
    through to ``default_post_process``: score the raw output, score the
    ``parse_output`` result when present, and take the per-metric maximum.
    """
    raw = subem_max_over_golds(prediction, golds)
    parsed = parse_output(prediction)
    if parsed is None:
        return bool(raw)
    return bool(raw or subem_max_over_golds(parsed, golds))


def validate(record: dict[str, Any]) -> list[str]:
    """Validation errors for a scored row; any error excludes that row."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["row was not a dict"]
    if not record.get("case_id"):
        errors.append("case_id is required")
    if str(record.get("subset", "")).upper() not in SUBSETS:
        errors.append(f"subset must be one of {SUBSETS}")
    if str(record.get("context", "")).lower() not in CONTEXTS:
        errors.append(f"context must be one of {CONTEXTS}")
    value = record.get("subem")
    if isinstance(value, bool):
        pass
    elif isinstance(value, int) and value in (0, 1):
        pass
    else:
        errors.append("subem must be a bool or 0/1")
    if not record.get("pred") and not record.get("error"):
        errors.append("pred or error is required for a checkpoint row")
    if not record.get("gold_hash_prefix"):
        errors.append("gold_hash_prefix is required for a scored checkpoint row")
    if SERIAL_LEAK_RE.match(str(record.get("pred") or "").strip()):
        errors.append("prediction embeds a serial marker in answer text")
    return errors


def _group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if bool(row["subem"]))
    mean = round(correct / n, 6) if n else None
    return {
        "correct": correct,
        "denominator": n,
        "mean_subem": mean,
        "display": f"substring_exact_match: {correct}/{n} = {mean:.3f}" if n
        else "substring_exact_match: 0/0 = no denominator",
    }


def evaluate(results: Iterable[Any]) -> dict[str, Any]:
    """Aggregate official-style SubEM. No judge is called or consulted."""
    source_rows = list(results or [])
    rows: list[dict[str, Any]] = []
    invalid = 0
    for index, record in enumerate(source_rows):
        errors = validate(record)
        if errors:
            invalid += 1
            rows.append(
                {
                    "index": index,
                    "case_id": record.get("case_id") if isinstance(record, dict) else None,
                    "valid": False,
                    "validation_errors": errors,
                }
            )
            continue
        rows.append(
            {
                "index": index,
                "case_id": record["case_id"],
                "subset": str(record["subset"]).upper(),
                "context": str(record["context"]).lower(),
                "subem": bool(record["subem"]),
                "valid": True,
            }
        )
    valid = [row for row in rows if row["valid"]]
    by_subset = {subset: _group([r for r in valid if r["subset"] == subset]) for subset in SUBSETS}
    by_context = {ctx: _group([r for r in valid if r["context"] == ctx]) for ctx in CONTEXTS}
    by_group = {
        f"{subset}@{ctx}": _group(
            [r for r in valid if r["subset"] == subset and r["context"] == ctx]
        )
        for subset in SUBSETS
        for ctx in CONTEXTS
    }
    overall = _group(valid)
    return {
        "lane": LANE,
        "metric": METRIC,
        "parse_path": PARSE_PATH,
        "upstream_commit": UPSTREAM_COMMIT,
        "received_rows": len(source_rows),
        "invalid_rows": invalid,
        "denominator": overall["denominator"],
        "correct": overall["correct"],
        "mean_subem": overall["mean_subem"],
        "by_subset": by_subset,
        "by_context": by_context,
        "by_subset_and_context": by_group,
        "judge_free": True,
        "judge_calls": 0,
        "rows": rows,
        "stamps": {
            "official": False,
            "diagnostic": "derived",
            "comparability_tier": "C",
            "score_asserted": False,
        },
        "official": False,
        "diagnostic": "derived",
        "comparability_tier": "C",
        "score_asserted": False,
        "note": NOTE,
    }


# --------------------------------------------------------------------------- #
# Offline self test (no network, no keys, no model, no vault)
# --------------------------------------------------------------------------- #


def self_test() -> dict[str, Any]:
    """Synthetic short-string answers: scorer + denominators + stamps."""
    synthetic = [
        # (prediction, gold answers, expected SubEM)
        ("The answer is Paris.", ["Paris"], True),          # case-insensitive containment
        ("PARIS!", ["paris"], True),                        # punctuation + case
        ("The Eiffel Tower", ["Paris"], False),             # no partial credit
        ("Answer: Mount Everest", ["Mount Everest"], True),  # parsed path
        # SubEM containment still accepts a label prefix (unlike exact_match).
        ("label: 43", ["43"], True),
        ("the chanel was founded by andy warhol", ["Andy Warhol"], True),  # article strip
        ("pesäpallo", ["pesäpallo"], True),                  # unicode gold, verbatim
        ("nothing relevant", [], False),                    # empty gold list
        ("The capital is Rome", ["the capital"], True),     # article-normalized gold substring
    ]
    checks = [
        {
            "prediction": pred,
            "golds": golds,
            "expected": expected,
            "observed": score_subem(pred, golds),
        }
        for pred, golds, expected in synthetic
    ]
    for check in checks:
        assert check["observed"] == check["expected"], check
    assert normalize_answer("The A, An, THE, Foo!") == "foo"
    assert parse_output("Answer: Rome\nreason") == "Rome"
    assert substring_exact_match_score("abc", "b") is True
    assert substring_exact_match_score("abc", "d") is False

    scored: list[dict[str, Any]] = []
    layout = [
        ("SH", "6k", 0, 1, 1, 0),
        ("SH", "32k", 1, 1, 1, 0),
        ("MH", "6k", 1, 0, 1, 0),
        ("MH", "262k", 0, 1, 0, 1),
    ]
    for subset, ctx, first, second, third, bad in layout:
        for index, value in enumerate((first, second, third)):
            scored.append(
                {
                    "case_id": f"test-{subset.lower()}-{ctx}-{index}",
                    "subset": subset,
                    "context": ctx,
                    "pred": f"synthetic-{index}",
                    "gold_hash_prefix": hashlib.sha256(str(index).encode()).hexdigest()[:12],
                    "subem": bool(value),
                }
            )
        if bad:
            scored.append(
                {
                    "case_id": f"test-{subset.lower()}-{ctx}-bad",
                    "subset": subset,
                    "context": ctx,
                    "pred": "3. leaked-serial",
                    "gold_hash_prefix": "deadbeef0000",
                    "subem": True,
                }
            )
    result = evaluate(scored)
    assert result["official"] is False
    assert result["diagnostic"] == "derived"
    assert result["comparability_tier"] == "C"
    assert result["score_asserted"] is False
    assert result["judge_calls"] == 0
    assert result["received_rows"] == 13
    assert result["invalid_rows"] == 1
    assert result["denominator"] == 12
    assert result["correct"] == 8
    assert result["by_subset"]["SH"]["denominator"] == 6
    assert result["by_subset"]["MH"]["denominator"] == 6
    assert result["by_context"]["6k"]["denominator"] == 6
    assert result["by_context"]["32k"]["denominator"] == 3
    assert result["by_context"]["262k"]["denominator"] == 3
    assert result["by_subset_and_context"]["SH@6k"]["correct"] == 2
    result["scorer_checks"] = checks
    result["self_test"] = "ok"
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="offline synthetic self test")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("pass --smoke; dataset preparation needs the pinned artifact and network access")
    print(json.dumps(self_test(), indent=2, sort_keys=True, ensure_ascii=False))
