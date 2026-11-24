"""Shared, stdlib-only helpers for the pinned external benchmark harnesses.

Design constraints (do not relax):

* Nothing here performs network I/O or reads a dataset at import time.
* Datasets are downloaded only by :func:`fetch_pinned`, only from an explicit
  pinned URL, and only into this repository's gitignored ``data/`` directory.
* A SHA-256 pin is verified against the bytes *before* those bytes are put at
  their final destination. A placeholder pin refuses to download.
* Selection only slices. It never pads, samples, or invents cases, so a
  reported denominator is always the number of rows that were actually used.

Upstream dataset bytes are never vendored or redistributed. Only pins, IDs,
and derived diagnostic metrics belong in this repository.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.parse import urlparse
from urllib.request import urlopen

PIN_BEFORE_USE = "PIN_BEFORE_USE"
"""Sentinel for a pin that must be filled in before any network call."""

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
FIXTURE_PATH = REPO_ROOT / "fixtures" / "pinned-ids.json"


class PinError(RuntimeError):
    """A pin is missing, a placeholder, or structurally invalid."""


class HashMismatch(RuntimeError):
    """Downloaded bytes did not match the pinned SHA-256."""


# --------------------------------------------------------------------------- #
# Pins, paths, downloads
# --------------------------------------------------------------------------- #


def is_placeholder(value: Any) -> bool:
    """True for unset pins. Never treat a placeholder as a usable value."""
    if value is None:
        return True
    if isinstance(value, str):
        text = value.strip()
        return not text or PIN_BEFORE_USE in text.upper()
    return False


def utc_now() -> str:
    """Real UTC timestamp, second resolution, Z suffix."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_data_path(dest: str | Path) -> Path:
    """Resolve ``dest`` and refuse anything outside gitignored ``data/``."""
    candidate = Path(dest)
    candidate = (REPO_ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        candidate.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise PinError(f"refusing to write outside gitignored data/: {candidate}") from exc
    return candidate


def fetch_pinned(
    url: str,
    sha256: str,
    dest: str | Path,
    *,
    timeout: float = 60.0,
    force: bool = False,
) -> Path:
    """Download ``url`` to ``dest`` under ``data/`` after verifying ``sha256``.

    The artifact is streamed to a temporary file, hashed, and only moved into
    place when the digest matches the pin. A cached artifact is reused only if
    it still matches; a stale cached artifact is refetched under the same pin.

    Raises :class:`PinError` for placeholder/invalid pins and
    :class:`HashMismatch` when bytes do not match. This function is the only
    network boundary in the harnesses; it is never called at import time.
    """
    if is_placeholder(url):
        raise PinError(
            "PIN_BEFORE_USE: dataset URL is still a placeholder; set the real pinned URL first"
        )
    if is_placeholder(sha256):
        raise PinError(
            "PIN_BEFORE_USE: dataset SHA-256 is still a placeholder; compute it from the "
            "pinned artifact and record it. Never invent a hash."
        )

    scheme = urlparse(str(url)).scheme.lower()
    if scheme not in {"http", "https"}:
        raise PinError(f"only http(s) dataset URLs are supported, got {scheme!r}")

    pin = str(sha256).strip().lower()
    if not _HEX64.match(pin):
        raise PinError("dataset SHA-256 must be exactly 64 hex characters")

    target = ensure_data_path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        if hmac.compare_digest(sha256_file(target), pin):
            return target
        target.unlink()  # stale/corrupt cache: refetch under the same pin

    temp = target.with_name(f"{target.name}.part-{os.getpid()}")
    try:
        with urlopen(str(url), timeout=timeout) as response, open(temp, "wb") as output:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                output.write(block)
        observed = sha256_file(temp)
        if not hmac.compare_digest(observed, pin):
            raise HashMismatch(
                f"SHA-256 mismatch for {url}: expected {pin}, got {observed}"
            )
        os.replace(temp, target)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return target


# --------------------------------------------------------------------------- #
# Fixtures and deterministic selection
# --------------------------------------------------------------------------- #


def load_ids(path: str | Path = FIXTURE_PATH) -> dict[str, Any]:
    """Load the committed pinned-ID fixture (small, never a dataset)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("pinned-id fixture must be a JSON object")
    return data


def fixture_section(lane: str) -> tuple[dict[str, Any], bool]:
    """Return ``(section, found)``; missing/malformed fixture is not fatal."""
    try:
        data = load_ids()
    except (OSError, ValueError):
        return {}, False
    section = data.get(lane)
    if not isinstance(section, dict):
        return {}, False
    return section, True


def limiter(cases: Iterable[Any], max_cases: int | None) -> list[Any]:
    """Deterministically slice ``cases``; never pad, sample, or reshuffle.

    ``None`` or ``0`` means no truncation. A non-negative integer keeps only
    the first ``max_cases`` items in the caller's existing order. The caller
    must compute its denominator from ``len(returned)`` -- this function never
    returns more items than it was given and never reports a denominator.
    """
    items = list(cases)
    if max_cases is None:
        return items
    if isinstance(max_cases, bool) or not isinstance(max_cases, int):
        raise TypeError("max_cases must be an int or None")
    if max_cases < 0:
        raise ValueError("max_cases must be non-negative")
    if max_cases == 0:
        return items
    return items[:max_cases]


def denominator(rows: Sequence[Any]) -> int:
    """The only honest denominator: how many rows are actually present."""
    return len(rows)


def pinned_ids(section: dict[str, Any], key: str, max_cases: int | None = None) -> list[str]:
    """Read an optional pinned ID array from the fixture section, if present."""
    raw = section.get(key, [])
    if not isinstance(raw, list):
        return []
    return [str(value) for value in limiter(raw, max_cases) if value is not None]


def stable_rows(results: Iterable[Any]) -> list[Any]:
    """Order result rows by a stable identifier, falling back to input index."""
    def key(pair: tuple[int, Any]) -> tuple[int, str]:
        index, record = pair
        if isinstance(record, dict):
            for field in ("question_id", "case_id", "id", "item_id"):
                if record.get(field) is not None:
                    return (0, str(record[field]))
        return (1, f"{index:08d}")

    return [record for _, record in sorted(enumerate(results), key=key)]


# --------------------------------------------------------------------------- #
# Honest reporting helpers
# --------------------------------------------------------------------------- #


def ratio(numerator: float, denom: int) -> float | None:
    """A labelled fraction, or None when there is no denominator to divide by.

    Numerator may be a count or a sum of per-item fractions; the denominator
    is always an integer count. Booleans are rejected so a judge verdict can
    never be mistaken for a count.
    """
    if isinstance(numerator, bool) or isinstance(denom, bool):
        raise TypeError("ratio needs numeric counts, not bool")
    if not isinstance(numerator, (int, float)) or not isinstance(denom, int):
        raise TypeError("ratio needs a numeric numerator and integer denominator")
    if denom <= 0:
        return None
    return round(numerator / denom, 6)


def ratio_text(name: str, numerator: float, denom: int, suffix: str = "") -> str:
    """Human-readable count/fraction with context -- never a bare percentage."""
    if denom <= 0:
        return f"{name}: 0/0 (no denominator; not estimated){suffix}"
    value = ratio(numerator, denom)
    return f"{name}: {numerator}/{denom} = {value:.3f}{suffix}"


def stamps(note: str, *, judge: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Mandatory honesty stamps for every evaluate() result."""
    result: dict[str, Any] = {
        "official": False,
        "diagnostic": "derived",
        "note": note,
        "observed_at": utc_now(),
    }
    if judge is not None:
        result["judge"] = judge
    if extra:
        result.update(extra)
    return result


def normalize_verdict(raw: Any) -> tuple[bool | None, Any, str | None]:
    """Coerce a judge_fn return value into a determinate boolean verdict."""
    if isinstance(raw, bool):
        return raw, raw, None
    if isinstance(raw, dict):
        for field in ("correct", "is_correct", "ok"):
            if isinstance(raw.get(field), bool):
                return raw[field], raw, None
        return None, raw, "judge mapping had no boolean correct/is_correct/ok field"
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"yes", "true", "correct", "1"}:
            return True, raw, None
        if text in {"no", "false", "incorrect", "0"}:
            return False, raw, None
        return None, raw, f"unrecognized judge string: {raw!r}"
    return None, raw, f"unrecognized judge return type: {type(raw).__name__}"


def apply_judge(
    rows: Iterable[Any],
    judge_fn: Callable[[Any], Any] | None,
    *,
    key: str = "question_id",
) -> dict[str, Any]:
    """Run an injected judge over rows, counting only determinate verdicts."""
    materialized = list(rows)
    if judge_fn is None:
        return {
            "ran": False,
            "correct": 0,
            "denominator": 0,
            "rows": [],
            "note": "judge_fn was not injected; no judge accuracy was computed",
        }
    correct = 0
    counted = 0
    per_item: list[dict[str, Any]] = []
    for index, record in enumerate(materialized):
        verdict, raw, reason = normalize_verdict(judge_fn(record))
        if verdict is not None:
            counted += 1
            correct += int(verdict)
        per_item.append(
            {
                "index": index,
                "id": record.get(key) if isinstance(record, dict) else None,
                "verdict": verdict,
                "reason": reason,
                "raw": raw,
            }
        )
    return {
        "ran": True,
        "correct": correct,
        "denominator": counted,
        "rows": per_item,
        "note": "indeterminate verdicts are excluded from the judge denominator",
    }


def stats(values: Iterable[Any]) -> dict[str, Any]:
    """Descriptive stats with an explicit n; non-numeric values are counted."""
    materialized = list(values)
    usable = [
        float(value)
        for value in materialized
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0
    ]
    result: dict[str, Any] = {
        "n": len(usable),
        "missing_or_invalid": len(materialized) - len(usable),
    }
    if usable:
        result.update(
            {
                "sum": round(sum(usable), 6),
                "mean": round(sum(usable) / len(usable), 6),
                "min": round(min(usable), 6),
                "max": round(max(usable), 6),
            }
        )
    return result


def token_f1(prediction: Any, reference: Any) -> float:
    """Deterministic SQuAD-style token F1; no model and no judge involved."""
    pred_tokens = re.findall(r"\w+", "" if prediction is None else str(prediction).lower())
    ref_tokens = re.findall(r"\w+", "" if reference is None else str(reference).lower())
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


__all__ = [
    "PIN_BEFORE_USE",
    "PinError",
    "HashMismatch",
    "REPO_ROOT",
    "DATA_ROOT",
    "FIXTURE_PATH",
    "is_placeholder",
    "utc_now",
    "sha256_file",
    "ensure_data_path",
    "fetch_pinned",
    "load_ids",
    "fixture_section",
    "limiter",
    "denominator",
    "pinned_ids",
    "stable_rows",
    "ratio",
    "ratio_text",
    "stamps",
    "normalize_verdict",
    "apply_judge",
    "stats",
    "token_f1",
]
