"""Pinned HotpotQA (Yang et al., 2018) answer-only EM/F1 harness adapter.

This lane is a **judge-free derived diagnostic**, not an official HotpotQA run
and not a memory benchmark. The official distractor dev set is a closed set of
10 paragraphs per question; here they are seeded into a per-case vault and the
question alone drives recall. Answer Exact Match (EM) and token-level F1 use the
official answer-only normalisation from ``hotpot_evaluate_v1.py`` (SQuAD style:
lowercase -> strip ASCII punctuation -> remove articles -> collapse whitespace),
with the official yes/no guard. No LLM judge, no score assertion, no rank.

Public provenance (verified 2026-11-24, no private document is a source of record):
HF revision ``1908d6afbbead072334abe2965f91bd2709910ab``, file
``distractor/validation-00000-of-00001.parquet``, SHA-256
``c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6``.
The bytes were independently re-verified locally: 7,405 rows, columns
``id, question, answer, type, level, supporting_facts, context``, and the
sorted-ID-list SHA-256 of the first 150 rows matches the pinned slice.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
import string
import sys
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

try:  # normal package import
    from ..common import denominator, ratio, ratio_text, stable_rows, stamps, utc_now
except (ImportError, ValueError):  # loaded directly by path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import denominator, ratio, ratio_text, stable_rows, stamps, utc_now

LANE = "hotpotqa"
SOURCE_URL = "https://github.com/hotpotqa/hotpot"
PAPER_URL = "https://arxiv.org/abs/1809.09600"

# --------------------------------------------------------------------------- #
# PINNED ARTIFACT -- every value below is verified, never guessed.
# Public provenance: HF revision 1908d6afbbead072334abe2965f91bd2709910ab,
# file distractor/validation-00000-of-00001.parquet, SHA-256 c20b638ca82b21d04
# fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6, verified 2026-11-24.
# The exact resolve/<sha>/ URL is used; resolve/main/ is never used. Hash and
# row count were re-checked locally against the 27,452,575-byte parquet file.
# --------------------------------------------------------------------------- #
HF_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
DATASET_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/"
    + HF_REVISION
    + "/distractor/validation-00000-of-00001.parquet"
)
DATASET_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
DATASET_BYTES = 27_452_575
DATASET_FILENAME = "validation-00000-of-00001.parquet"
CONFIG = "distractor"
SPLIT = "validation"
N_DEV_ROWS = 7_405

# Approved external data root. Dataset bytes never enter the repository.
BENCH_DATA_ROOT = Path(r"D:\memnoth-bench-data")
DEFAULT_DATA_DIR = BENCH_DATA_ROOT / "datasets" / "hotpotqa"
CAP_BYTES = 80_000_000_000  # hard 80 GB cap for the whole bench-data root

# Deterministic n=150 slice: sort all 7,405 ids lexicographically, take first
# 150. The id-list digest below is SHA-256 of those 150 ids joined by '\n'.
SLICE_SIZE = 150
SLICE_ID_SHA256 = "5e3e6eb0e03ff6a866aa0ace6b0eaf569d1b32f89f6ad87c7cdda99e2ee960e9"
SLICE_FIRST_FIVE = (
    "5a70eee85542994082a3e3f0",
    "5a70f0a75542994082a3e403",
    "5a70f0e75542994082a3e408",
    "5a70f11a5542994082a3e40b",
    "5a70f1685542994082a3e40f",
)
SLICE_LAST_FIVE = (
    "5a727e1b5542991f9a20c497",
    "5a7280105542994cef4bc2e4",
    "5a7280925542991f9a20c4a1",
    "5a7281075542994cef4bc2e9",
    "5a72814f5542994cef4bc2eb",
)
SLICE_TYPE_MIX = {"bridge": 117, "comparison": 33}
SLICE_LEVEL = "hard"

# Official scorer: algorithm transcribed verbatim from the pinned script read on
# 2026-11-24. No hash is invented -- a scorer copy has not been frozen, so the
# hash is None and is reported as such. Do not present it as verified.
SCORER_URL = "https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py"
SCORER_SHA256 = None
SCORER_SEMANTICS = (
    "Answer-only EM and token F1 from hotpot_evaluate_v1.py: normalize_answer = "
    "white_space_fix(remove_articles(remove_punc(lower(s)))); EM = normalized "
    "equality; F1 = Counter overlap on normalised tokens with the official "
    "yes/no/noanswer guard. Sup and Joint are not scored."
)

K_PARAGRAPHS = 10  # distractor setting: exactly 10 paragraphs per question
_YES_NO = frozenset({"yes", "no", "noanswer"})
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_ENTITY_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")

_NOTE = (
    "Derived diagnostic on a pinned 150-row HotpotQA distractor-dev slice with "
    "a pinned extractive reader; judge-free answer-only EM/F1 computed locally. "
    "official=false, diagnostic=derived, comparability_tier=C, "
    "score_asserted=false. NOT an official score. No ranking or win against "
    "published systems is claimed."
)

__all__ = [
    "LANE", "SOURCE_URL", "PAPER_URL", "HF_REVISION", "DATASET_URL",
    "DATASET_SHA256", "DATASET_BYTES", "DATASET_FILENAME", "CONFIG", "SPLIT",
    "N_DEV_ROWS", "BENCH_DATA_ROOT", "DEFAULT_DATA_DIR", "CAP_BYTES",
    "SLICE_SIZE", "SLICE_ID_SHA256", "SLICE_FIRST_FIVE", "SLICE_LAST_FIVE",
    "SLICE_TYPE_MIX", "SLICE_LEVEL", "SCORER_URL", "SCORER_SHA256",
    "SCORER_SEMANTICS", "K_PARAGRAPHS", "normalize_answer", "em_score",
    "f1_score", "score_pair", "sha256_text", "sha256_file", "gold_hash_prefix",
    "prepare", "load_rows", "select", "paragraph_records", "build_rows",
    "public_row", "evaluate", "self_test",
]


# --------------------------------------------------------------------------- #
# Official answer-only scorer -- exact semantics, copied from the pinned script.
# Order matters: lower -> strip string.punctuation -> remove a/an/the -> collapse.
# --------------------------------------------------------------------------- #


def normalize_answer(s: Any) -> str:
    """The official HotpotQA/SQuAD normaliser. No stemming, no locale tricks."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower("" if s is None else str(s)))))


def em_score(prediction: Any, gold: Any) -> int:
    """Official answer Exact Match: 1 when normalised strings are equal."""
    return int(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: Any, gold: Any) -> tuple[float, float, float]:
    """Official answer F1 -> ``(f1, precision, recall)``.

    Mirrors ``hotpot_evaluate_v1.py:f1_score`` including the symmetric
    yes/no/noanswer guard; a guard hit or zero overlap returns ``(0, 0, 0)``.
    """
    zero = (0.0, 0.0, 0.0)
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(gold)
    if norm_pred in _YES_NO and norm_pred != norm_gold:
        return zero
    if norm_gold in _YES_NO and norm_pred != norm_gold:
        return zero
    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def score_pair(prediction: Any, gold: Any) -> tuple[int, float]:
    """Return ``(em, f1)``; an abstention/missing prediction scores (0, 0)."""
    if prediction is None or not str(prediction).strip():
        return 0, 0.0
    f1, _, _ = f1_score(prediction, gold)
    return em_score(prediction, gold), f1


# --------------------------------------------------------------------------- #
# Small hashing / download helpers (no gold or key values are ever emitted).
# --------------------------------------------------------------------------- #


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gold_hash_prefix(gold: Any, length: int = 12) -> str | None:
    """Only a hash prefix of gold is ever published, never the gold string."""
    if gold is None or not str(gold):
        return None
    return sha256_text(gold)[:length]


def _directory_bytes(root: Path) -> int:
    total = 0
    if root.is_dir():
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    return total


def fetch_pinned_external(
    url: str,
    sha256: str,
    expected_bytes: int,
    dest: Path,
    *,
    timeout: float = 120.0,
) -> tuple[Path, str]:
    """Stream, verify, then move a pinned artifact under the approved data root.

    Unlike ``common.fetch_pinned`` (which rejects anything outside the repo's
    ``data/``), this accepts only a destination under ``D:\\memnoth-bench-data``,
    enforces the 80 GB cap and a disk-space margin, and verifies both the SHA-256
    pin and the recorded byte count before ``os.replace``. This is the single
    network boundary and is never invoked at import time or by ``--dry-run``.
    """
    if not _HEX64.fullmatch(str(sha256).strip()):
        raise ValueError("dataset SHA-256 pin must be exactly 64 hex characters")
    if urlparse(str(url)).scheme.lower() not in {"http", "https"}:
        raise ValueError("dataset URL must be http(s)")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        raise ValueError("expected byte count must be a positive int")

    target = Path(dest).expanduser().resolve()
    try:
        target.relative_to(BENCH_DATA_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"refusing to write outside approved data root {BENCH_DATA_ROOT}: {target}"
        ) from exc

    current = _directory_bytes(BENCH_DATA_ROOT)
    if current + expected_bytes > CAP_BYTES:
        raise RuntimeError(
            f"80 GB bench-data cap would be exceeded: current={current} "
            f"adding={expected_bytes} cap={CAP_BYTES}"
        )
    if shutil.disk_usage(target.anchor).free < expected_bytes * 2:
        raise RuntimeError("insufficient free disk space for the pinned artifact")

    pin = str(sha256).strip().lower()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if target.stat().st_size == expected_bytes and hmac.compare_digest(
            sha256_file(target), pin
        ):
            return target, "dataset_cache_hit"
        target.unlink()  # stale/corrupt cache: refetch under the same pin

    temp = target.with_name(f"{target.name}.part-{os.getpid()}")
    try:
        total = 0
        with urllib.request.urlopen(str(url), timeout=timeout) as response, temp.open("wb") as out:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                total += len(block)
        observed = sha256_file(temp)
        if not hmac.compare_digest(observed, pin):
            raise RuntimeError(f"SHA-256 mismatch for {target.name}: expected {pin}, got {observed}")
        if total != expected_bytes:
            raise RuntimeError(
                f"byte-count mismatch for {target.name}: expected {expected_bytes}, got {total}"
            )
        os.replace(temp, target)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return target, "dataset_verified"


def prepare(local_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    """Download and verify the pinned dev parquet into ``local_dir`` (default D:\\...).

    Pass an explicit path to redirect the download; it must stay under the
    approved bench-data root, outside the repository. No network at import time.
    """
    dest = Path(local_dir).expanduser() / DATASET_FILENAME
    return fetch_pinned_external(DATASET_URL, DATASET_SHA256, DATASET_BYTES, dest)[0]


# --------------------------------------------------------------------------- #
# Deterministic selection and row loading (offline after preparation).
# --------------------------------------------------------------------------- #


def _slice_limit(max_cases: int | None) -> int:
    """The pinned slice is n=150; 0/None keeps that default, never the full set."""
    if max_cases is None or max_cases == 0:
        return SLICE_SIZE
    if isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 0:
        raise ValueError("max_cases must be a non-negative int or None")
    return min(max_cases, N_DEV_ROWS)


def load_rows(
    dataset_path: str | Path | None = None,
    max_cases: int | None = SLICE_SIZE,
) -> list[dict[str, Any]]:
    """Read the pinned parquet, sort by ``id`` ascending, take the first N rows.

    This is the only place gold fields enter memory. Callers must hand only the
    question text to recall and only context text to the reader; gold and
    supporting-fact labels are for scoring/diagnostics, never reader input.
    """
    path = Path(dataset_path).expanduser() if dataset_path else DEFAULT_DATA_DIR / DATASET_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"pinned dataset not prepared: {path}")
    if not hmac.compare_digest(sha256_file(path), DATASET_SHA256):
        raise RuntimeError(f"dataset SHA-256 mismatch for {path}; refusing to use bytes")

    import pyarrow.parquet as pq  # lazy: importing this module stays stdlib-only

    columns = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]
    table = pq.read_table(path, columns=columns)
    data = table.to_pydict()
    order = sorted(range(table.num_rows), key=lambda i: str(data["id"][i]))
    limit = min(_slice_limit(max_cases), len(order))
    rows: list[dict[str, Any]] = []
    for position in order[:limit]:
        rows.append(
            {
                "id": str(data["id"][position]),
                "question": str(data["question"][position]),
                "answer": str(data["answer"][position]),
                "type": str(data["type"][position]),
                "level": str(data["level"][position]),
                "supporting_facts": data["supporting_facts"][position],
                "context": data["context"][position],
            }
        )
    return rows


def select(max_cases: int | None = SLICE_SIZE) -> dict[str, Any]:
    """Describe the deterministic pinned 150-row slice without any network.

    If the prepared parquet is already present locally, actual IDs and their
    digest are computed offline (no download). Otherwise only the frozen pin
    metadata (rule, denominator, first/last five, recorded digest) is returned.
    """
    limit = _slice_limit(max_cases)
    path = DEFAULT_DATA_DIR / DATASET_FILENAME
    ids: list[str] = []
    verified_sha: str | None = None
    if path.is_file() and hmac.compare_digest(sha256_file(path), DATASET_SHA256):
        try:
            ids = [str(row["id"]) for row in load_rows(path, limit)]
            verified_sha = sha256_text("\n".join(ids))
        except (OSError, RuntimeError, ValueError):
            ids, verified_sha = [], None

    if verified_sha is not None:
        id_list_sha256: str | None = verified_sha
    elif limit == SLICE_SIZE:
        id_list_sha256 = SLICE_ID_SHA256
    else:
        id_list_sha256 = None

    return {
        "lane": LANE,
        "source": SOURCE_URL,
        "paper": PAPER_URL,
        "hf_revision": HF_REVISION,
        "dataset_url": DATASET_URL,
        "dataset_sha256": DATASET_SHA256,
        "dataset_bytes": DATASET_BYTES,
        "config": CONFIG,
        "split": SPLIT,
        "rule": "sort all 7405 ids ascending (lexicographic) and take the first N",
        "planned_denominator": SLICE_SIZE,
        "effective_denominator": denominator(ids) if ids else limit,
        "ordinals": list(range(1, limit + 1)),
        "ids": ids,
        "ids_resolved_offline": bool(ids),
        "id_list_sha256": id_list_sha256,
        "id_list_sha256_matches_pin": verified_sha == SLICE_ID_SHA256,
        "slice_first_five": list(SLICE_FIRST_FIVE),
        "slice_last_five": list(SLICE_LAST_FIVE),
        "slice_type_mix": dict(SLICE_TYPE_MIX),
        "slice_level": SLICE_LEVEL,
        "all_dev_rows_hard": True,
        "official_split": False,
        "k_paragraphs": K_PARAGRAPHS,
        "note": _NOTE,
    }


# --------------------------------------------------------------------------- #
# Gold-free ingestion / recall / reader wiring + official local scoring.
# --------------------------------------------------------------------------- #


def _entity_name(value: Any) -> str:
    return _ENTITY_UNSAFE.sub("-", str(value)).strip("-")[:120] or "entity"


def paragraph_records(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Return all 10 distractor paragraphs as ``(entity, title, text)`` records.

    Reads ``context`` only. Never reads ``answer`` or ``supporting_facts``.
    A structural mismatch raises with the exact observed shape so a caller can
    report a precise blocker rather than silently dropping paragraphs.
    """
    context = sample.get("context")
    if not isinstance(context, dict):
        raise ValueError(f"context must be a struct/dict, got {type(context).__name__}")
    titles = context.get("title")
    sentences = context.get("sentences")
    if not isinstance(titles, (list, tuple)) or not isinstance(sentences, (list, tuple)):
        raise ValueError(
            f"context.title/sentences must be lists, got "
            f"{type(titles).__name__}/{type(sentences).__name__}"
        )
    if len(titles) != K_PARAGRAPHS or len(sentences) != K_PARAGRAPHS:
        raise ValueError(
            f"expected {K_PARAGRAPHS} paragraphs, got titles={len(titles)} "
            f"sentences={len(sentences)}"
        )
    # The title is the node name (per the pinned plan): a question can then
    # actually recall a paragraph by its title. Opaque IDs are unfindable.
    records: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for index, (title, group) in enumerate(zip(titles, sentences)):
        if not isinstance(group, (list, tuple)):
            raise ValueError(f"context.sentences[{index}] must be a list, got {type(group).__name__}")
        clean = [str(item).strip() for item in group if str(item).strip()]
        text = f"{str(title)}\n" + "\n".join(clean)
        base = _entity_name(title) or f"paragraph-{index:02d}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        entity = base if count == 0 else _entity_name(f"{base}-p{index:02d}")
        records.append(
            {
                "entity": entity,
                "title": str(title),
                "text": text[:2000],
                "index": index,
            }
        )
    return records


def build_rows(
    samples: Iterable[dict[str, Any]],
    *,
    seed_fn: Callable[[str, str], bool],
    recall_fn: Callable[[str, int], list[str]],
    reader_fn: Callable[[str, list[str]], tuple[str | None, str | None]],
    k: int = K_PARAGRAPHS,
    detail: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Seed -> recall -> read -> score for each sample, with zero gold leakage.

    Contracts:
      ``seed_fn(entity, text) -> bool``        (one vault note per paragraph)
      ``recall_fn(question, k) -> [entity]``   (the query is the question only)
      ``reader_fn(question, [text]) -> (pred, error)`

    The reader receives only the question and the text of paragraphs that were
    seeded from ``context``; it never receives ``answer``, ``supporting_facts``,
    ``id``, ``type`` or any hint about the gold span. Missing predictions are
    scored 0 explicitly and reported, never skipped.
    """
    counters = {
        "cases": 0,
        "paragraphs_seeded": 0,
        "recall_empty": 0,
        "reader_failed": 0,
        "missing_pred": 0,
    }
    rows: list[dict[str, Any]] = []
    for sample in samples or []:
        question = str(sample.get("question") or "")
        gold = str(sample.get("answer") or "")
        paragraphs = paragraph_records(sample)
        entity_to_text = {item["entity"]: item["text"] for item in paragraphs}

        seeded = 0
        for item in paragraphs:
            if seed_fn(item["entity"], item["text"]):
                seeded += 1

        recalled = [str(name) for name in (recall_fn(question, k) or [])]
        recalled_texts: list[str] = []
        for name in recalled:
            text = entity_to_text.get(name)
            if text is not None and text not in recalled_texts:
                recalled_texts.append(text)

        error: str | None = None
        pred: str | None = None
        if seeded == 0:
            error = "no_paragraphs_seeded"
        elif not recalled_texts:
            error = "no_recalled_content"
        else:
            pred, error = reader_fn(question, recalled_texts)

        if pred is None or not str(pred).strip():
            counters["missing_pred"] += 1
            if error is None:
                error = "missing_prediction"
        em, f1 = score_pair(pred, gold)
        rows.append(
            {
                "question_id": str(sample.get("id") or sample.get("question_id") or ""),
                "question": question,
                "pred": pred.strip()[:300] if isinstance(pred, str) else None,
                "gold": gold,  # in-memory only; never written to a public artifact
                "gold_hash_prefix": gold_hash_prefix(gold),
                "em": em,
                "f1": f1,
                "recalled_entities": recalled,
                "recalled_count": len(recalled_texts),
                "seeded_count": seeded,
                "reader_error": error,
                "k": k,
            }
        )
        counters["cases"] += 1
        counters["paragraphs_seeded"] += seeded
        counters["recall_empty"] += int(not recalled_texts)
        counters["reader_failed"] += int(error is not None)
    if detail is not None:
        detail.update(counters)
    return rows


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    """Exactly the checkpoint fields: no gold, no key, no question text."""
    return {
        "question_id": row.get("question_id"),
        "pred": row.get("pred"),
        "gold_hash_prefix": row.get("gold_hash_prefix"),
        "em": int(row.get("em") or 0),
        "f1": float(row.get("f1") or 0.0),
    }


def evaluate(results: list[Any]) -> dict[str, Any]:
    """Answer-only EM/F1 with real denominators and all honesty stamps.

    EM is the primary metric; its numerator is the count of normalised-equal
    answers and its denominator is the number of rows with a gold label. A
    missing/abstaining prediction is scored 0 (published as ``missing_pred``),
    never excluded. Gold is used only here and in :func:`score_pair`.
    """
    source_rows = stable_rows(results or [])
    scored = 0
    em_correct = 0
    f1_sum = 0.0
    missing_pred = 0
    label_missing = 0
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
        gold = record.get("gold")
        pred = record.get("pred")
        if gold is None or not str(gold).strip():
            label_missing += 1
            rows.append(
                {
                    "index": index,
                    "question_id": record.get("question_id"),
                    "valid": False,
                    "note": "no gold label; excluded from EM/F1 denominators",
                }
            )
            continue
        if pred is None or not str(pred).strip():
            missing_pred += 1
        em, f1 = score_pair(pred, gold)
        scored += 1
        em_correct += em
        f1_sum += f1
        rows.append(
            {
                "index": index,
                "question_id": record.get("question_id"),
                "em": em,
                "f1": round(f1, 6),
                "gold_hash_prefix": gold_hash_prefix(gold),
                "missing_pred": pred is None or not str(pred).strip(),
                "valid": True,
            }
        )

    f1_sum_rounded = round(f1_sum, 6)
    em_display = ratio_text("answer_em", em_correct, scored, " (diagnostic; not an official score)")
    f1_display = ratio_text(
        "answer_f1_sum", f1_sum_rounded, scored, " (diagnostic; not an official score)"
    )
    summary = "; ".join(
        [
            ratio_text("rows supplied", len(source_rows), len(source_rows), ""),
            ratio_text("rows with gold", scored, len(source_rows), ""),
            f"missing_predictions_scored_zero: {missing_pred}",
            f"gold_labels_missing: {label_missing}",
            em_display,
            f"mean_answer_f1: {ratio(f1_sum_rounded, scored)}",
        ]
    )
    extra: dict[str, Any] = {
        "comparability_tier": "C",
        "score_asserted": False,
        "not_an_official_score": True,
        "dataset_sha256": DATASET_SHA256,
        "hf_revision": HF_REVISION,
        "slice_id_sha256": SLICE_ID_SHA256,
        "scorer_url": SCORER_URL,
        "scorer_sha256": SCORER_SHA256,
        "scorer_note": (
            "Algorithm transcribed from the pinned official script; a scorer "
            "copy/hash has NOT been frozen, so None is reported rather than "
            "an invented hash."
        ),
        "metric_definition": SCORER_SEMANTICS,
        "gold_withheld_from_reader": True,
    }
    return {
        "lane": LANE,
        "denominator": denominator(source_rows),
        "correct": em_correct,
        "primary_metric": "answer_em",
        "primary_denominator": scored,
        "rows": rows,
        "metrics": {
            "primary": {
                "name": "answer_em",
                "correct": em_correct,
                "denominator": scored,
                "display": em_display,
                "judge_free": True,
            },
            "answer_em": {
                "correct": em_correct,
                "denominator": scored,
                "display": em_display,
            },
            "answer_f1": {
                "sum": f1_sum_rounded,
                "denominator": scored,
                "mean": ratio(f1_sum_rounded, scored),
                "display": f1_display,
            },
            "missing_pred_scored_zero": missing_pred,
            "gold_labels_missing": label_missing,
        },
        "summary": summary,
        "source_url": SOURCE_URL,
        **stamps(_NOTE, extra=extra),
    }


# --------------------------------------------------------------------------- #
# Offline self-test: scorer vectors, gold-leakage guard, full synthetic build.
# --------------------------------------------------------------------------- #


def self_test() -> dict[str, Any]:
    """Offline checks: no network, no keys, no model calls, no dataset I/O."""
    vectors = [
        ("The Cat, sat!", "cat sat"),
        ("A dog and the cat", "dog and cat"),
        ("  Hello   WORLD!!  ", "hello world"),
        ("yes", "yes"),
    ]
    normalisations = {src: normalize_answer(src) for src, _ in vectors}
    assert all(normalize_answer(src) == want for src, want in vectors)

    assert em_score("The Eiffel Tower", "eiffel tower") == 1
    assert em_score("Paris", "London") == 0
    assert score_pair("the blue notebook", "blue notebook") == (1, 1.0)
    assert abs(f1_score("Barack Obama", "Barack Hussein Obama")[0] - 0.8) < 1e-9
    assert f1_score("yes", "no") == (0.0, 0.0, 0.0)
    assert f1_score("noanswer", "Paris") == (0.0, 0.0, 0.0)
    assert score_pair(None, "Paris") == (0, 0.0)

    def _synthetic(qid: str, gold: str, answer_paragraph: str) -> dict[str, Any]:
        titles = [f"Paragraph {i}" for i in range(K_PARAGRAPHS)]
        groups = [[f"Sentence {i}.a."] for i in range(K_PARAGRAPHS)]
        groups[3] = [f"{answer_paragraph} is the answer span."]
        return {
            "id": qid,
            "question": f"What is named in {qid}?",
            "answer": gold,
            "type": "bridge",
            "level": "hard",
            "supporting_facts": {"title": ["Paragraph 3"], "sent_id": [0]},
            "context": {"title": titles, "sentences": groups},
        }

    samples = [
        _synthetic("syn-1", "Aurora", "Aurora"),
        _synthetic("syn-2", "blue notebook", "blue notebook"),
        _synthetic("syn-3", "Nimbus", "Different"),
        _synthetic("syn-4", "Harbor", "Harbor"),
    ]
    called: dict[str, Any] = {"seeds": [], "queries": [], "reader_inputs": []}
    answers = {"syn-1": "Aurora", "syn-2": "the blue notebook", "syn-3": "Wrong", "syn-4": None}
    entities_by_qid = {
        str(sample["id"]): [item["entity"] for item in paragraph_records(sample)]
        for sample in samples
    }

    def seed_fn(entity: str, text: str) -> bool:
        called["seeds"].append((entity, text))
        return True

    def recall_fn(question: str, k: int) -> list[str]:
        called["queries"].append((question, k))
        assert k == K_PARAGRAPHS
        match = re.search(r"(syn-\d+)", question)
        assert match is not None
        entities = list(entities_by_qid[match.group(1)])
        # Fake recall must name only entities that ingestion actually seeded;
        # this keeps the synthetic test in lockstep with title-based naming.
        seeded = {entity for entity, _ in called["seeds"]}
        assert all(entity in seeded for entity in entities)
        return entities

    def reader_fn(question: str, facts: list[str]) -> tuple[str | None, str | None]:
        called["reader_inputs"].append((question, list(facts)))
        # Guard: the reader receives only (question, seeded context text).
        match = re.search(r"(syn-\d+)", question)
        assert match is not None
        qid = match.group(1)
        assert qid in answers
        assert len(facts) == K_PARAGRAPHS
        assert all("supporting_facts" not in fact for fact in facts)
        return answers[qid], None

    detail: dict[str, Any] = {}
    built = build_rows(
        samples, seed_fn=seed_fn, recall_fn=recall_fn, reader_fn=reader_fn, detail=detail
    )
    assert detail["cases"] == 4
    assert detail["paragraphs_seeded"] == 4 * K_PARAGRAPHS
    assert detail["missing_pred"] == 1

    result = evaluate(built)
    assert result["official"] is False
    assert result["diagnostic"] == "derived"
    assert result["comparability_tier"] == "C"
    assert result["score_asserted"] is False
    assert result["not_an_official_score"] is True
    assert result["denominator"] == 4
    assert result["correct"] == 2  # syn-1 and syn-2 are normalised-equal
    assert result["metrics"]["answer_f1"]["denominator"] == 4
    assert result["metrics"]["missing_pred_scored_zero"] == 1
    assert len(called["reader_inputs"]) == 4
    assert all(q == question for (q, _), question in zip(called["queries"], [s["question"] for s in samples]))

    checkpoint = [public_row(row) for row in built]
    assert all(set(row) == {"question_id", "pred", "gold_hash_prefix", "em", "f1"} for row in checkpoint)
    assert all("gold" not in row and "key" not in row for row in checkpoint)

    return {
        "lane": LANE,
        "checks": {
            "normalisation_vectors": len(vectors),
            "em_f1_vectors": 7,
            "synthetic_cases": len(samples),
            "paragraphs_seeded": detail["paragraphs_seeded"],
            "reader_calls": len(called["reader_inputs"]),
            "missing_pred_scored_zero": detail["missing_pred"],
            "checkpoint_fields": sorted(checkpoint[0]),
            "gold_leak_guard": True,
        },
        "evaluation_summary": result["summary"],
        "official": result["official"],
        "diagnostic": result["diagnostic"],
        "comparability_tier": result["comparability_tier"],
        "score_asserted": result["score_asserted"],
        "not_an_official_score": result["not_an_official_score"],
        "note": "Offline synthetic self-test only. No score is asserted.",
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="offline synthetic smoke test")
    parser.add_argument("--self-test", action="store_true", help="alias for --smoke")
    args = parser.parse_args()
    if not (args.smoke or args.self_test):
        parser.error("pass --smoke/--self-test; dataset preparation is a separate network step")
    print(json.dumps(self_test(), indent=2, sort_keys=True))
