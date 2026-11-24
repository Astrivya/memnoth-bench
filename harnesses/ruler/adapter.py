"""Pinned, judge-free RULER retrieval diagnostic adapter.

This adapter is a *derived local harness*, not RULER's official runner. It uses
RULER's deterministic scoring rule (case-insensitive substring containment,
``string_match_all`` / ``string_match_part`` from the pinned upstream
``scripts/eval/synthetic/constants.py``) but replaces RULER's LLM reader with
this product's own vault recall. No model and no judge are called anywhere in
this file: there is no injected judge callable, no key import, no network at
import time, and no score assertion.

The only usable data pin today is the 4K ``niah_single_1`` parquet whose
SHA-256 was independently recomputed. 8K-128K pins are deliberately marked
``verified=False`` with no invented digest: :func:`prepare` refuses them.

Honesty stamps on every :func:`evaluate` result: official=False,
diagnostic="derived", comparability_tier="C", score_asserted=False.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

try:  # normal package import
    from ..common import (
        PinError,
        denominator,
        limiter,
        ratio,
        ratio_text,
        sha256_file,
        stamps,
        stable_rows,
        utc_now,
    )
except (ImportError, ValueError):  # loaded directly by path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import (  # type: ignore[no-redef]
        PinError,
        denominator,
        limiter,
        ratio,
        ratio_text,
        sha256_file,
        stamps,
        stable_rows,
        utc_now,
    )

LANE = "ruler"
SOURCE_URL = "https://github.com/NVIDIA/RULER"
PAPER_URL = "https://arxiv.org/abs/2404.06654"
# Pinned classic script pipeline on main (PGP-verified by GitHub at research time).
UPSTREAM_COMMIT = "db43f92ff634bbce1c33d1fb15858aec4bba509a"
DATASET_REPO = "self-long/RULER-llama3-1M"
DATASET_REVISION = "5da49e4000224d3289ffb14fc32cbd20211765ed"
DATASET_ROOT = (
    "https://huggingface.co/datasets/self-long/RULER-llama3-1M/resolve/"
    + DATASET_REVISION
)
# External data root. Nothing from a dataset is ever written into the repo.
DEFAULT_DATA_DIR = Path(r"D:\memnoth-bench-data\datasets\ruler")
DATASET_DIR_CAP_BYTES = 80 * 1024**3  # workspace hard cap for D:\memnoth-bench-data

# Judge-freeness is structural, not a claim: no key pool, no judge callable,
# no reader model, no completion endpoint appears in this module.
JUDGE_FREE = True
API_KEYS_REQUIRED = False
MODEL_CALLS = 0
COMPARABILITY_TIER = "C"

# Upstream source pins. Computed from raw.githubusercontent.com at the pinned
# commit; kept as provenance only (this adapter never downloads or executes
# upstream code). Provenance is fully public — no private path is referenced.
SOURCE_FILE_PINS: dict[str, dict[str, Any]] = {
    "README.md": {
        "bytes": 18187,
        "sha256": "f3aae5bbd58fcd7de32aab099c455dd2bbc1da23a2efee9db3071eb9fbf894a3",
    },
    "scripts/synthetic.yaml": {
        "bytes": 2356,
        "sha256": "34bc71dcacdc41a829a170f04b528fbf48d62c616338005ab4991680fbf8cb0b",
    },
    "scripts/config_tasks.sh": {
        "bytes": 1167,
        "sha256": "bd2b228b303951ff6e1dba2dd551270744b50d78f5d51493b0c4abfb41a0c544",
    },
    "scripts/config_models.sh": {
        "bytes": 3039,
        "sha256": "f59f5dd0a2e61ba430840f156af534f32a011db640881642d0093a885130cf34",
    },
    "scripts/data/synthetic/constants.py": {
        "bytes": 3052,
        "sha256": "6296e901d495ec6200dc3f68993ea13d8282e3c0dbe1a8c47967f111105d1fde",
    },
    "scripts/data/synthetic/niah.py": {
        "bytes": 12743,
        "sha256": "53f2e2e0c61607d3814303c7f22b2f419fd004eb3bfe92a00ecf811e31aa8451",
    },
    "scripts/data/prepare.py": {
        "bytes": 6730,
        "sha256": "f2d210860fbf5c640cb41ed104c7a923c9f4043f6e6c354277daacc73afc643d",
    },
    "scripts/eval/synthetic/constants.py": {
        "bytes": 1531,
        "sha256": "6740467c17b8dc06b6b30f4f97e54ce8de81db0dd879f1538d0b6b5727f4bd5f",
    },
    "scripts/eval/evaluate.py": {
        "bytes": 6934,
        "sha256": "b9a5fcbded7209663d97670496c74dbcd6358ddbe3d9887a0c5f0c4f974bfe1c",
    },
}

# --------------------------------------------------------------------------- #
# Data pins. A pin is usable only when verified=True AND sha256 is a real
# locally recomputed digest. The 8K-128K rows keep HF LFS metadata oids only as
# a note; an LFS oid quoted from metadata is NOT an independent verification.
# --------------------------------------------------------------------------- #


def _unverified(length: str, lfs_oid: str | None = None) -> dict[str, Any]:
    note = "UNVERIFIED — SHA-256 not independently recomputed; prepare() refuses this pin"
    if lfs_oid:
        note += "; HF metadata LFS oid (not a verified digest): " + lfs_oid
    return {
        "verified": False,
        "sha256": None,
        "bytes": None,
        "url": DATASET_ROOT + f"/niah_single_1_{length}/validation-00000-of-00001.parquet",
        "note": note,
    }


PINS: dict[tuple[str, str], dict[str, Any]] = {
    ("niah_single_1", "4k"): {
        "verified": True,
        # Verified: downloaded from the revision-pinned resolve URL and hashed
        # locally; equals the HF LFS oid. 416,867 bytes.
        "sha256": "03b988e7d1eb9b380b7a658641caf5692dc8d99be89712a87e65003626f417f2",
        "bytes": 416867,
        "url": DATASET_ROOT + "/niah_single_1_4k/validation-00000-of-00001.parquet",
        "note": "VERIFIED 2026-11-24: local SHA-256 recomputation equals HF LFS oid",
    },
    ("niah_single_1", "8k"): _unverified(
        "8k", "55b517c2de2605397e812fdd83c54dcd5b7616b9f500bfbbb2de8ce27fcf4ba1"
    ),
    ("niah_single_1", "16k"): _unverified(
        "16k", "c720221eba49a74ba482c92bce78df4d1aac2bae11fd93ec31c96bbd5fc37b10"
    ),
    ("niah_single_1", "32k"): _unverified(
        "32k", "abc2e32a75159d4bc80f6eb99ac7cbf300b22ecf390553894492f0cda38ccd25"
    ),
    ("niah_single_1", "64k"): _unverified(
        "64k", "1956bbf83f3b3f591a18a688fb7f488d56c659a67fef987b1bce6db206eb7c4c"
    ),
    ("niah_single_1", "128k"): _unverified(
        "128k", "6bd410541936a1dfdb980c5bb123293c771f19a229ec6ba781252a0879087ab9"
    ),
}

# Task names whose upstream scorer is a pure string function; qa_* excluded
# because their gold derives from SQuAD/HotpotQA and is not fully synthetic.
JUDGE_FREE_TASKS: tuple[str, ...] = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
)
LENGTHS: tuple[str, ...] = ("4k", "8k", "16k", "32k", "64k", "128k")
CHUNK_LINES = 5  # deterministic lines per vault entity; never splits a line
RECALL_LIMIT = 10
MAX_PRED_CHARS = 4000

_NOTE = (
    "Derived, judge-free RULER retrieval diagnostic. RULER's official reader "
    "generates an answer from the full context in one prompt; here the context "
    "is seeded into a temporary vault and its recall is scored with RULER's "
    "case-insensitive substring rule. NOT an official RULER score, not "
    "comparable to published rows, no rank or win claimed."
)

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_QUESTION_RE = re.compile(
    r"What is the special magic number for (?P<key>[^?]+?) mentioned in the provided text\?"
)
_INSTRUCTION = (
    "A special magic number is hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the number afterwards."
)
_ASSISTANT_SPLIT = "<|eot_id|><|start_header_id|>assistant"
_NUMERIC_GOLD = re.compile(r"^\d{4,}$")
_NUMERIC_CANDIDATE = re.compile(r"\b\d{4,}\b")


# --------------------------------------------------------------------------- #
# Pins, validation, preparation
# --------------------------------------------------------------------------- #


def pin_for(task: str, length: str) -> dict[str, Any]:
    """Return the pin for ``(task, length)`` or raise a precise ValueError."""
    key = (str(task).strip(), str(length).strip().lower())
    if key not in PINS:
        raise ValueError(
            f"no RULER pin for task={key[0]!r} length={key[1]!r}; "
            f"available: {sorted(t for t, _ in PINS)}"
        )
    return dict(PINS[key])


def validate() -> dict[str, Any]:
    """Offline structural check of every pin. No network, no keys."""
    checked = 0
    for (task, length), pin in PINS.items():
        if pin.get("verified") is True:
            digest = str(pin.get("sha256") or "")
            if not _HEX64.fullmatch(digest) or not isinstance(pin.get("bytes"), int):
                raise PinError(f"verified pin {task}:{length} lacks a real SHA-256/byte count")
        else:
            if pin.get("sha256") is not None:
                raise PinError(f"unverified pin {task}:{length} must not carry a digest")
            if urlparse(str(pin.get("url"))).scheme.lower() not in {"http", "https"}:
                raise PinError(f"pin {task}:{length} URL must be http(s)")
        checked += 1
    for path, pin in SOURCE_FILE_PINS.items():
        if not _HEX64.fullmatch(str(pin.get("sha256") or "")):
            raise PinError(f"source pin {path} has an invalid SHA-256")
    if not (JUDGE_FREE and not API_KEYS_REQUIRED and MODEL_CALLS == 0):
        raise PinError("RULER harness must remain judge-free and key-free")
    return {
        "lane": LANE,
        "pins_checked": checked,
        "verified_pins": sorted(f"{t}:{n}" for (t, n), p in PINS.items() if p["verified"]),
        "refused_pins": sorted(f"{t}:{n}" for (t, n), p in PINS.items() if not p["verified"]),
        "source_files_pinned": len(SOURCE_FILE_PINS),
        "upstream_commit": UPSTREAM_COMMIT,
        "judge_free": True,
        "api_keys_required": False,
    }


def _guard_budget(data_dir: Path, new_bytes: int) -> None:
    """Refuse a write that would pass the external 80 GB dataset cap."""
    if new_bytes < 0:
        raise PinError("negative byte count")
    if data_dir.is_dir():
        present = sum(
            path.stat().st_size for path in data_dir.rglob("*") if path.is_file()
        )
    else:
        present = 0
    if present + new_bytes > DATASET_DIR_CAP_BYTES:
        raise PinError(
            f"refusing write: dataset dir would hold {present + new_bytes} bytes "
            f"(hard cap {DATASET_DIR_CAP_BYTES})"
        )
    free = shutil.disk_usage(str(data_dir.anchor if data_dir.anchor else Path.cwd().anchor)).free
    if free < new_bytes + 5 * 1024**3:
        raise PinError("refusing write: less than 5 GB free after the dataset write")


def prepare(
    task: str = "niah_single_1",
    length: str = "4k",
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> Path:
    """Download and SHA-256-verify the pinned parquet into an EXTERNAL directory.

    Never writes into the repository's ``data/``. An unverified pin (8K-128K)
    is refused before any network call: no digest is invented from HF metadata.
    """
    pin = pin_for(task, length)
    if not pin.get("verified"):
        raise PinError(
            f"refusing unverified pin {task}:{length}: {pin['note']}. "
            "Recompute the SHA-256 from a downloaded artifact and update the pin first."
        )
    digest = str(pin["sha256"]).lower()
    expected_bytes = int(pin["bytes"])
    if not _HEX64.fullmatch(digest):
        raise PinError("verified pin SHA-256 must be exactly 64 hex characters")
    url = str(pin["url"])
    if urlparse(url).scheme.lower() not in {"http", "https"}:
        raise PinError("dataset URL must be http(s)")

    dest_dir = Path(data_dir).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        dest_dir.relative_to(repo_root)
    except ValueError:
        pass
    else:
        raise PinError(f"refusing to write dataset inside the repository: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    _guard_budget(dest_dir, expected_bytes)
    dest = dest_dir / f"{task}_{length}.parquet"

    if dest.is_file() and hmac.compare_digest(sha256_file(dest), digest):
        if dest.stat().st_size == expected_bytes:
            return dest
        dest.unlink()  # hash matched but size did not: never trust this cache

    temp = dest.with_name(f"{dest.name}.part-{os.getpid()}")
    try:
        with urlopen(url, timeout=120) as response, temp.open("wb") as output:
            total = 0
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                output.write(block)
                total += len(block)
        observed = sha256_file(temp)
        if not hmac.compare_digest(observed, digest):
            raise RuntimeError(
                f"SHA-256 mismatch for {dest.name}: expected {digest}, got {observed}"
            )
        if total != expected_bytes:
            raise RuntimeError(
                f"byte-count mismatch for {dest.name}: expected {expected_bytes}, got {total}"
            )
        os.replace(temp, dest)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return dest


# --------------------------------------------------------------------------- #
# Official metric (verbatim rule from the pinned upstream evaluator)
# --------------------------------------------------------------------------- #


def postprocess_pred(value: Any) -> str:
    """Upstream ``postprocess_pred``: strip, control chars -> newline, strip."""
    return re.sub(r"[\x00-\x1f]", "\n", "" if value is None else str(value).strip()).strip()


def string_match_part(prediction: Any, references: Sequence[Any]) -> float:
    """RULER ``string_match_part``: 100.0 if ANY reference is contained."""
    pred = postprocess_pred(prediction).lower()
    refs = [str(item).lower() for item in references if item is not None]
    if not refs:
        return 0.0
    return 100.0 if any(ref in pred for ref in refs) else 0.0


def string_match_all(prediction: Any, references: Sequence[Any]) -> float:
    """RULER ``string_match_all``: 100 * fraction of ALL references contained."""
    pred = postprocess_pred(prediction).lower()
    refs = [str(item).lower() for item in references if item is not None]
    if not refs:
        return 0.0
    hits = sum(ref in pred for ref in refs)
    return round(100.0 * hits / len(refs), 2)


def substring_match_f1(prediction: Any, references: Sequence[Any]) -> float:
    """Derived per-sample F1 from exact substring containment (no model/judge).

    Recall = contained gold strings / all gold strings. Precision = contained
    gold strings / distinct answer-shaped candidate strings found in the
    prediction: for numeric-gold tasks, distinct runs of 4+ digits; otherwise
    only the contained gold strings are countable (precision is 1 by
    construction, so F1 reduces to containment recall). This is a derived
    diagnostic, NOT an official RULER metric; the official rule is
    :func:`string_match_all` above.
    """
    pred = postprocess_pred(prediction).lower()
    refs = [str(item).lower() for item in references if item is not None]
    if not refs:
        return 0.0
    hits = sum(ref in pred for ref in refs)
    if hits == 0:
        return 0.0
    recall = hits / len(refs)
    if all(_NUMERIC_GOLD.fullmatch(ref) for ref in refs):
        candidates = set(_NUMERIC_CANDIDATE.findall(pred))
        precision = hits / len(candidates) if candidates else 0.0
    else:
        precision = 1.0
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


def score_one(prediction: Any, gold: Sequence[Any]) -> dict[str, Any]:
    """Score one sample locally. Gold is used only here, never in a query."""
    refs = [str(item) for item in (gold or []) if item is not None and str(item) != ""]
    pred = postprocess_pred(prediction)
    all_score = string_match_all(pred, refs)
    part_score = string_match_part(pred, refs)
    return {
        "string_match_all": all_score,
        "string_match_part": part_score,
        "contained_count": sum(str(item).lower() in pred.lower() for item in refs),
        "gold_count": len(refs),
        "f1": substring_match_f1(pred, refs) if refs else 0.0,
        "valid": bool(refs),
        "match_all": all_score == 100.0,
        "match_part": part_score == 100.0,
    }


# --------------------------------------------------------------------------- #
# Row construction: parquet -> deterministic haystack chunks + gold-free query
# --------------------------------------------------------------------------- #


def _parse_niah_row(record: dict[str, Any]) -> dict[str, Any] | None:
    """Extract haystack, question and gold from a pinned parquet row.

    Verified 4K schema: ``index:int64``, ``input:string`` (Llama-3 template:
    system header + instruction + haystack + question + assistant answer
    prefix), ``answers:list<string>`` (short gold strings), ``length:int64``,
    ``predictions:struct`` (third-party model outputs, never read here).

    Returns None (with a reason) when the template cannot be parsed. No value
    is guessed, padded, or copied from ``predictions``.
    """
    text = record.get("input")
    if not isinstance(text, str) or not text:
        return None
    user_part = text.split(_ASSISTANT_SPLIT, 1)[0]
    match = _QUESTION_RE.search(user_part)
    if match is None:
        return None
    question = match.group(0).strip()
    key = match.group("key").strip()
    before = user_part[: match.start()]
    lines = [line.strip() for line in before.splitlines()]
    try:
        start = next(i for i, line in enumerate(lines) if line == _INSTRUCTION) + 1
    except StopIteration:
        return None
    haystack_lines = [line for line in lines[start:] if line]
    if not haystack_lines:
        return None
    raw_gold = record.get("answers")
    gold = [str(item) for item in raw_gold if item is not None and str(item) != ""] if isinstance(
        raw_gold, (list, tuple)
    ) else []
    if not gold:
        return None
    # No-gold-leakage guard: the recall query must never contain an answer.
    if any(value.lower() in question.lower() for value in gold):
        return None
    return {
        "index": record.get("index"),
        "key": key,
        "question": question,
        "haystack_lines": haystack_lines,
        "context": "\n".join(haystack_lines),
        "gold": gold,
        "answer": gold[0],
        "source_length": record.get("length"),
    }


def build_rows(
    source: str | Path | Iterable[dict[str, Any]],
    *,
    task: str = "niah_single_1",
    length: str = "4k",
    max_cases: int | None = None,
    chunk_lines: int = CHUNK_LINES,
) -> list[dict[str, Any]]:
    """Turn parquet rows (or synthetic dict rows) into vault-seedable cases.

    Each case carries ``chunks`` (deterministic groups of intact haystack
    lines) and ``entities`` (stable ids ``ruler_<len>_<index>_<chunk>``). The
    ``query`` is the RULER question (key only); gold is never included.
    """
    if not isinstance(chunk_lines, int) or chunk_lines < 1:
        raise ValueError("chunk_lines must be a positive int")
    if isinstance(source, (str, Path)):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("pyarrow is required to read the pinned parquet") from exc
        path = Path(source)
        table = pq.read_table(path, columns=["index", "input", "answers", "length"])
        records: list[dict[str, Any]] = table.to_pylist()
    else:
        records = [row for row in source if isinstance(row, dict)]
    records = limiter(records, max_cases)

    rows: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        parsed = _parse_niah_row(record)
        if parsed is None:
            skipped += 1
            continue
        lines = parsed["haystack_lines"]
        chunks = ["\n".join(lines[i : i + chunk_lines]) for i in range(0, len(lines), chunk_lines)]
        index = parsed["index"]
        rows.append(
            {
                "case_id": f"ruler_{length}_{index}",
                "index": index,
                "task": task,
                "length": length,
                "key": parsed["key"],
                "query": parsed["question"],
                "context": parsed["context"],
                "context_lines": len(lines),
                "chunks": chunks,
                "entities": [f"ruler_{length}_{index}_c{n:03d}" for n in range(len(chunks))],
                "gold": parsed["gold"],
                "answer": parsed["answer"],
                "source_length": parsed["source_length"],
                "skipped_rows": skipped,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Aggregate evaluation: denominators, official:false, derived stamps
# --------------------------------------------------------------------------- #


def evaluate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate result rows. No judge_fn parameter exists by design.

    Each row needs ``pred`` (recalled text) and ``gold`` (gold strings). Gold
    is consumed here only; emitted rows carry a gold *hash prefix*, never gold
    itself. Denominators are real counts of supplied/scored/excluded rows.
    """
    materialized = stable_rows(list(results or []))
    rows: list[dict[str, Any]] = []
    all_correct = 0
    part_correct = 0
    f1_sum = 0.0
    scored = 0

    for position, record in enumerate(materialized):
        if not isinstance(record, dict):
            rows.append(
                {
                    "index": position,
                    "valid": False,
                    "note": "row was not a dict; excluded from all denominators",
                }
            )
            continue
        gold = record.get("gold")
        gold = [str(item) for item in gold if item is not None and str(item) != ""] if isinstance(
            gold, (list, tuple)
        ) else []
        verdict = score_one(record.get("pred"), gold)
        gold_hash = hashlib.sha256("\x1f".join(gold).encode("utf-8")).hexdigest()[:12] if gold else None
        row: dict[str, Any] = {
            "index": position,
            "case_id": record.get("case_id"),
            "task": record.get("task"),
            "length": record.get("length"),
            "gold_hash_prefix": gold_hash,
            "query_hash_prefix": hashlib.sha256(
                str(record.get("query") or "").encode("utf-8")
            ).hexdigest()[:12],
            "valid": verdict["valid"],
            **verdict,
        }
        if not verdict["valid"]:
            row["note"] = "no gold strings; excluded from all denominators"
        else:
            scored += 1
            all_correct += int(verdict["match_all"])
            part_correct += int(verdict["match_part"])
            f1_sum += verdict["f1"]
        rows.append(row)

    f1_sum = round(f1_sum, 6)
    f1_mean = round(f1_sum / scored, 6) if scored else None
    score_all = round(100.0 * all_correct / scored, 2) if scored else None
    metrics = {
        "primary": {
            "name": "ruler_string_match_all_case_insensitive_substring",
            "correct": all_correct,
            "denominator": scored,
            "score": score_all,
            "display": ratio_text(
                "RULER string_match_all (case-insensitive substring)",
                all_correct,
                scored,
                " (derived harness; not an official RULER score)",
            ),
            "judge_free": True,
        },
        "string_match_part": {
            "correct": part_correct,
            "denominator": scored,
            "score": round(100.0 * part_correct / scored, 2) if scored else None,
            "display": ratio_text("string_match_part", part_correct, scored, ""),
        },
        "substring_match_f1": {
            "sum": f1_sum,
            "denominator": scored,
            "mean": f1_mean,
            "ratio": ratio(f1_sum, scored),
        },
    }
    summary_parts = [
        ratio_text("rows supplied", len(materialized), len(materialized), ""),
        ratio_text("scored", scored, len(materialized), ""),
        ratio_text("excluded", len(materialized) - scored, len(materialized), ""),
        metrics["primary"]["display"],
        metrics["string_match_part"]["display"],
        f"substring_match_f1_mean: {f1_mean} over n={scored}",
        "judge: none (no model call, no key)",
    ]
    result: dict[str, Any] = {
        "lane": LANE,
        "denominator": denominator(materialized),
        "scored": scored,
        "excluded": len(materialized) - scored,
        "correct": all_correct,
        "part_correct": part_correct,
        "f1_sum": f1_sum,
        "f1_mean": f1_mean,
        "primary_metric": metrics["primary"]["name"],
        "rows": rows,
        "metrics": metrics,
        "summary": "; ".join(summary_parts),
        "source_url": SOURCE_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "observed_at": utc_now(),
    }
    result.update(
        stamps(
            _NOTE,
            extra={
                "comparability_tier": COMPARABILITY_TIER,
                "score_asserted": False,
                "judge_free": True,
                "api_keys_required": False,
                "model_calls": MODEL_CALLS,
                "metric_source": (
                    "NVIDIA/RULER scripts/eval/synthetic/constants.py at "
                    + UPSTREAM_COMMIT
                    + ": string_match_all / string_match_part (case-insensitive containment)"
                ),
                "reader": "local mn vault recall (not RULER's LLM reader)",
            },
        )
    )
    return result


# --------------------------------------------------------------------------- #
# Offline validation and smoke
# --------------------------------------------------------------------------- #


def _synthetic_parquet_rows() -> list[dict[str, Any]]:
    """Two synthetic niah-shaped rows; no network, no file, no gold in query."""
    haystack = "\n".join(
        ["The grass is green. The sky is blue."] * 8
        + ["One of the special magic numbers for quiet-lantern is: 4826173."]
        + ["The grass is green. The sky is blue."] * 4
    )
    rows = []
    for index, (key, number) in enumerate((("quiet-lantern", "4826173"), ("brisk-orbit", "1059203"))):
        body = haystack.replace("quiet-lantern", key).replace("4826173", number)
        rows.append(
            {
                "index": index,
                "input": (
                    "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: "
                    "December 2023\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                    + _INSTRUCTION
                    + "\n"
                    + body
                    + f"\nWhat is the special magic number for {key} mentioned in the provided text?"
                    + _ASSISTANT_SPLIT
                    + "\n\n The special magic number mentioned in the provided text is"
                ),
                "answers": [number],
                "length": 4096,
            }
        )
    return rows


def self_test() -> dict[str, Any]:
    """Offline smoke: pin validation, synthetic haystack parse, deterministic scoring."""
    validation = validate()
    assert validation["pins_checked"] == len(PINS)
    assert validation["verified_pins"] == ["niah_single_1:4k"]
    assert len(validation["refused_pins"]) == 5

    synthetic = _synthetic_parquet_rows()
    rows = build_rows(synthetic, max_cases=2)
    assert len(rows) == 2, "both synthetic rows must parse"
    for row in rows:
        assert row["query"].startswith("What is the special magic number for ")
        assert row["answer"] not in row["query"], "gold leaked into the recall query"
        assert any(row["answer"] in chunk for chunk in row["chunks"]), "needle lost in chunking"
        assert len(row["entities"]) == len(row["chunks"])
        assert len(set(row["entities"])) == len(row["entities"])

    verdict_sets = [
        ("t1", ["4826173"], "One of the special magic numbers for quiet-lantern is: 4826173."),
        ("t2", ["1059"], "\x00value: 1059\n"),  # control char + case-fold path
        ("t3", ["7777777"], "the number is 1234567"),
        ("t4", ["12345", "67890"], "67890 only"),
        ("t5", [], "no gold here"),  # excluded from all denominators
    ]
    results = [
        {"case_id": case_id, "gold": gold, "pred": pred, "task": "niah_single_1", "length": "4k"}
        for case_id, gold, pred in verdict_sets
    ]
    verdict = evaluate(results)
    assert verdict["official"] is False
    assert verdict["diagnostic"] == "derived"
    assert verdict["comparability_tier"] == "C"
    assert verdict["score_asserted"] is False
    assert verdict["judge_free"] is True
    assert verdict["denominator"] == 5
    assert verdict["scored"] == 4
    assert verdict["excluded"] == 1
    assert verdict["correct"] == 2
    assert verdict["part_correct"] == 3
    assert verdict["f1_sum"] == 2.666667
    assert verdict["f1_mean"] == 0.666667
    by_id = {row.get("case_id"): row for row in verdict["rows"]}
    assert by_id["t1"]["string_match_all"] == 100.0
    assert by_id["t2"]["string_match_all"] == 100.0
    assert by_id["t3"]["string_match_all"] == 0.0
    assert by_id["t4"]["string_match_all"] == 50.0
    assert by_id["t4"]["string_match_part"] == 100.0
    assert by_id["t5"]["valid"] is False
    assert all("gold" not in row for row in verdict["rows"]), "raw gold must not be emitted"

    return {
        "lane": LANE,
        "offline": True,
        "network_calls": 0,
        "keys_loaded": 0,
        "model_calls": 0,
        "pins_checked": validation["pins_checked"],
        "verified_pins": validation["verified_pins"],
        "refused_pins": validation["refused_pins"],
        "synthetic_rows_parsed": len(rows),
        "chunks_per_row": [len(row["chunks"]) for row in rows],
        "rows_supplied": verdict["denominator"],
        "rows_scored": verdict["scored"],
        "rows_excluded": verdict["excluded"],
        "match_all_correct": verdict["correct"],
        "match_part_correct": verdict["part_correct"],
        "f1_sum": verdict["f1_sum"],
        "f1_mean": verdict["f1_mean"],
        "official": verdict["official"],
        "diagnostic": verdict["diagnostic"],
        "note": "Synthetic offline smoke only; no dataset, no score, no win claimed.",
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="offline synthetic self-test")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("pass --smoke; dataset preparation is a verified network boundary")
    print(json.dumps(self_test(), indent=2, sort_keys=True))
