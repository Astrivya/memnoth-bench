"""Standalone, fast, multi-key, checkpointed LoCoMo derived diagnostic runner.

This file deliberately does NOT go through ``runner/bench.py``. ``bench.py``
reads a single ``KILO_API_KEY``; this runner fans model calls across the eight
keys loaded by ``runner.kilo_pool`` (env file + KILOCODE_API_KEY) while keeping
the frozen harness of ``locomo_reader``/``locomo_seed`` unchanged.

HONESTY CONTRACT
----------------
* Every artifact is stamped official=False, diagnostic="derived". No score is
  asserted as a product result and no comparison to a paper/vendor row is made.
* The reader receives ONLY the question and CLI-recalled facts. Gold is used
  only for the strict judge and for locally computed token F1; it is never
  seeded, never sent to the reader, and never written to a checkpoint/result.
* Key values are never printed, logged, or written. Checkpoints carry only a
  gold hash prefix. ``KeyPool.stats()`` is counts/cooldowns only.
* The dataset is downloaded from the pinned adapter URL and SHA-256 verified
  before use. Bytes land under D:\\memnoth-bench-data, never inside the repo.
* CLI vault calls are model-free (isolated HOME via adapters.mn_json). KeyPool
  is used ONLY for reader/judge model calls, never for CLI calls.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import mn_json  # noqa: E402
from runner import locomo_reader, locomo_seed  # noqa: E402
from runner.kilo_pool import KeyPool, load_keys  # noqa: E402

VERSION = "0.1.0"
SCHEMA = "memnoth-bench-locomo-run/1"
NOTE = (
    "INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE. Derived LoCoMo diagnostic: "
    "real CLI recall + frozen reader/judge, token F1 computed locally. No score "
    "is asserted as a product result and no ranking is implied."
)

DATA_ROOT = Path(r"D:\memnoth-bench-data")
DEFAULT_DATASET_DIR = DATA_ROOT / "datasets" / "locomo"
DEFAULT_CHECKPOINT_DIR = DATA_ROOT / "checkpoints" / "locomo"
DEFAULT_RESULTS_DIR = DATA_ROOT / "results"
REPORT_PATH = DATA_ROOT / "reports" / "LOCOMO_RUN.md"

# Pins copied verbatim from harnesses/locomo/adapter.py (verified against
# snap-research/locomo commit 3eb6f2c5). Kept here so this runner is standalone.
DATASET_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/"
    "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json"
)
DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
DATASET_BYTES = 2805274

MAX_WORKERS_CEILING = 16        # one case per worker; case chain is sequential internally
MAX_CLI_WRITE_WORKERS = 8       # locomo_seed caps its own fan-out at this ceiling
MAX_KEY_RETRIES = 3             # retries with the NEXT key after a failed model call
_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_HTTP_STATUS = re.compile(r"_http_(\d{3})\b")
_WORD_RE = re.compile(r"\w+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_ENTITY_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")

_STOPWORDS = {
    "the", "and", "for", "was", "were", "what", "when", "where", "who", "with",
    "that", "this", "from", "have", "had", "has", "did", "does", "how", "why",
    "about", "into", "its", "our", "your", "their", "they", "then", "than",
    "after", "before", "first", "last", "been", "are", "not", "you", "his", "her",
}

# Frozen harness identities, read from the single source of truth. There are no
# CLI override flags for models/prompts, so the run is always the frozen pair.
READER_MODEL = locomo_reader.DEFAULT_READER_MODEL
JUDGE_MODEL = locomo_reader.DEFAULT_JUDGE_MODEL
BASE_URL = locomo_reader.DEFAULT_BASE_URL
READER_PROMPT = locomo_reader.READER_PROMPT_7STEP
JUDGE_PROMPT = locomo_reader.JUDGE_PROMPT_STRICT
READER_PROVIDER = "kilo"
JUDGE_PROVIDER = "kilo"


# --------------------------------------------------------------------------- #
# Small honest helpers (no key values, no gold values in outputs)
# --------------------------------------------------------------------------- #


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def status_from_error(code: str | None) -> int | None:
    """Return 429/5xx from an adapter error code, else None (no cooldown)."""
    match = _HTTP_STATUS.search(str(code or ""))
    if not match:
        return None
    status = int(match.group(1))
    return status if status == 429 or 500 <= status <= 599 else None


def token_f1(prediction: Any, reference: Any) -> float:
    """Deterministic SQuAD-style token F1; matches harnesses.common.token_f1."""
    pred = _WORD_RE.findall("" if prediction is None else str(prediction).lower())
    ref = _WORD_RE.findall("" if reference is None else str(reference).lower())
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


# --------------------------------------------------------------------------- #
# Pinned dataset preparation (verify before use, outside the repo)
# --------------------------------------------------------------------------- #


def fetch_pinned_dataset(dest: Path) -> tuple[Path, str]:
    """Stream the pinned dataset to ``dest`` and verify SHA-256 before moving."""
    if not _HEX64.fullmatch(str(DATASET_SHA256).strip()):
        raise ValueError("dataset SHA-256 pin is not 64 hex characters")
    if urlparse(DATASET_URL).scheme.lower() not in {"http", "https"}:
        raise ValueError("dataset URL is not http(s)")
    pin = DATASET_SHA256.strip().lower()
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file():
        if dest.stat().st_size == DATASET_BYTES and hmac.compare_digest(sha256_file(dest), pin):
            return dest, "dataset_cache_hit"
        dest.unlink()  # stale/corrupt cache: refetch under the same pin

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
        if not hmac.compare_digest(sha256_file(temp), pin):
            raise RuntimeError(f"SHA-256 mismatch for {dest.name}; refusing unverified bytes")
        if total != DATASET_BYTES:
            raise RuntimeError(f"byte-count mismatch for {dest.name}: expected {DATASET_BYTES}, got {total}")
        os.replace(temp, dest)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return dest, "dataset_verified"


# --------------------------------------------------------------------------- #
# Model-free CLI plumbing (KeyPool is NEVER used here)
# --------------------------------------------------------------------------- #


def resolve_cli() -> tuple[Path | None, Path | None]:
    try:
        bin_path = mn_json.discover_bin()
    except FileNotFoundError:
        bin_path = None
    configured = os.environ.get("MN_CLI_ROOT")
    cli_root = Path(configured).expanduser() if configured else ROOT.parent / "CLI"
    return bin_path, (cli_root if cli_root.is_dir() else None)


def entity_name(value: Any) -> str:
    return _ENTITY_UNSAFE.sub("-", str(value)).strip("-")[:120] or "entity"


def vault_remember(
    bin_path: Path | None, cli_root: Path | None, vault: Path, entity: str, text: str
) -> bool:
    result = mn_json.run_verb(
        "vault",
        ["remember", entity_name(entity), str(text)[:2000], "--vault", str(vault)],
        bin_path,
        timeout=30.0,
        cwd=cli_root,
    )
    return bool(result.get("ok")) and "error" not in result


def vault_recall(
    bin_path: Path | None, cli_root: Path | None, vault: Path, query: str, limit: int
) -> list[str]:
    result = mn_json.run_verb(
        "vault",
        ["recall", str(query)[:2000], "--limit", str(limit), "--vault", str(vault)],
        bin_path,
        timeout=30.0,
        cwd=cli_root,
    )
    if not result.get("ok") or "error" in result:
        return []
    hits = (result.get("data") or {}).get("hits") or []
    names: list[str] = []
    for hit in hits:
        if isinstance(hit, dict):
            name = hit.get("name") or hit.get("canonical_name")
            if name:
                names.append(str(name))
    return names


def keyword_query(text: str) -> str:
    tokens = _TOKEN_RE.findall(str(text or "").lower())
    return " ".join(token for token in tokens if token not in _STOPWORDS)[:200] or str(text)[:200]


def cli_recall_sweep(
    bin_path: Path | None, cli_root: Path | None, vault: Path, question: str, limit: int
) -> list[str]:
    """The proven bench.py recall sweep: full question, then keyword forms."""
    ranked = vault_recall(bin_path, cli_root, vault, question, limit)
    if ranked:
        return ranked
    terms = [token for token in keyword_query(question).split() if token]
    queries: list[str] = []
    if terms:
        queries.append(" ".join(terms))
    for count in (3, 2, 1):
        if len(terms) >= count:
            queries.append(" ".join(terms[:count]))
    for term in dict.fromkeys(terms):
        queries.append(term)
    seen: set[str] = set()
    found: list[str] = []
    for query in queries:
        for name in vault_recall(bin_path, cli_root, vault, query, limit):
            if name not in seen:
                seen.add(name)
                found.append(name)
        if found:
            break
    return found


def vault_asof(
    bin_path: Path | None, cli_root: Path | None, vault: Path, entity: str
) -> list[str]:
    result = mn_json.run_verb(
        "vault",
        ["asof", "--name", entity_name(entity), "--vault", str(vault)],
        bin_path,
        timeout=30.0,
        cwd=cli_root,
    )
    if not result.get("ok") or "error" in result:
        return []
    facts = (result.get("data") or {}).get("facts") or []
    return [str(fact.get("content")) for fact in facts if isinstance(fact, dict)]


# --------------------------------------------------------------------------- #
# Model calls: KeyPool ONLY here; short error codes only
# --------------------------------------------------------------------------- #


def reader_with_pool(pool: KeyPool, question: str, facts: list[str]) -> tuple[str | None, str | None]:
    """Frozen reader over CLI-recalled facts, with next-key retries on failure."""
    last_error = "reader_unavailable"
    for _ in range(MAX_KEY_RETRIES):
        key = pool.get()
        if key is None:
            return None, "no_key_available"
        answer, error = locomo_reader.reader_answer(
            question,
            facts,
            key,
            reader_model=READER_MODEL,
            base_url=BASE_URL,
            system_prompt=READER_PROMPT,
        )
        if error is None and isinstance(answer, str) and answer.strip():
            pool.release(key, True, 200)
            return answer.strip()[:300], None
        pool.release(key, False, status_from_error(error))
        last_error = str(error or "reader_unknown_error")
    return None, last_error


def judge_with_pool(
    pool: KeyPool, question: str, gold: str, pred: str
) -> tuple[str | None, str | None]:
    """Frozen strict judge; raw verdict label only (no smoothing)."""
    last_error = "judge_unavailable"
    for _ in range(MAX_KEY_RETRIES):
        key = pool.get()
        if key is None:
            return None, "no_key_available"
        verdict, error = locomo_reader.judge_answer(
            question,
            gold,
            pred,
            key,
            model=JUDGE_MODEL,
            base_url=BASE_URL,
            system_prompt=JUDGE_PROMPT,
        )
        if verdict is True:
            pool.release(key, True, 200)
            return "correct", None
        if verdict is False:
            pool.release(key, True, 200)
            return "incorrect", None
        # UNSURE / bad verdict are clean responses, not transport failures: no
        # point spending two more keys on the same deterministic outcome.
        if error in ("judge_unsure", "judge_bad_verdict"):
            pool.release(key, True, 200)
            return None, str(error)
        pool.release(key, False, status_from_error(error))
        last_error = str(error or "judge_unknown_error")
    return None, last_error


# --------------------------------------------------------------------------- #
# One sequential case chain (recall -> asof -> reader -> judge -> local F1)
# --------------------------------------------------------------------------- #


def process_case(
    *,
    index: int,
    question: dict[str, Any],
    conversation_index: int,
    bin_path: Path | None,
    cli_root: Path | None,
    vault: Path,
    pool: KeyPool,
) -> dict[str, Any]:
    question_id = f"locomo-{conversation_index}-{index + 1}"
    question_text = str(question.get("question") or "")
    gold = str(question.get("answer") or "")

    def record(
        pred: str | None, verdict: str | None, error: str | None, f1: float = 0.0
    ) -> dict[str, Any]:
        return {
            "question_id": question_id,
            "pred": pred,
            "gold_hash_prefix": sha256_text(gold)[:12] if gold else None,
            "f1": f1,
            "verdict": verdict,
            "reader_model": READER_MODEL,
            "judge_model": JUDGE_MODEL,
            "error": error,
        }

    try:
        hits = locomo_seed.recall_question(
            lambda q, n: cli_recall_sweep(bin_path, cli_root, vault, q, n),
            question_text,
            10,
        )
        if not hits:
            return record(None, None, "no_recall_observation")
        facts: list[str] = []
        for name in hits:
            for item in vault_asof(bin_path, cli_root, vault, name):
                text = str(item).strip()
                if text and text not in facts:
                    facts.append(text)
        if not facts:
            return record(None, None, "no_recalled_content")

        pred, reader_error = reader_with_pool(pool, question_text, facts)
        if pred is None:
            return record(None, None, reader_error or "reader_failed")

        f1 = token_f1(pred, gold)
        verdict, judge_error = judge_with_pool(pool, question_text, gold, pred)
        # A missing verdict is reported, never converted into a "correct".
        error = reader_error or judge_error or ("no_verdict" if verdict is None else None)
        return record(pred, verdict, error, f1)
    except Exception as exc:  # never let a worker exception become a fabricated score
        return record(None, None, f"case_error_{type(exc).__name__}")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


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


def pool_stats_for(keys: list[str]) -> dict[str, Any]:
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


def write_report(path: Path, result: dict[str, Any]) -> None:
    den = result["denominators"]
    models = result["models"]
    failures = result["top_failure_reasons"]
    lines = [
        "# LoCoMo standalone runner — derived diagnostic",
        "",
        f"- schema: `{result['schema']}`  observed_at: `{result['observed_at']}`",
        "- status: **" + str(result["status"]) + "**",
        "- **INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE.** No score is asserted as a product result.",
        "",
        "## Real denominators",
        "",
        f"- attempted: {den['attempted']}",
        f"- scored (prediction + local F1): {den['scored']}",
        f"- judged (raw verdict returned): {den['judged']}",
        f"- failed (an error was recorded): {den['failed']}",
        f"- no_verdict: {den['no_verdict']}",
        f"- selected: {den['selected']}  resumed_skipped: {den['resumed_skipped']}",
        "",
        "## F1 (computed locally against gold; no judge smoothing)",
        "",
        f"- correct_f1_count (exact token match, F1 == 1.0): {result['correct_f1_count']}",
        f"- f1_sum: {result['f1_sum']} over n={den['scored']} scored cases"
        f"  mean F1: {result['f1_mean']}",
        f"- judge_correct_count: {result['judge_correct_count']} over {den['judged']} judged cases",
        f"- reader UNSUPPORTED answers: {result['unsupported_count']} (real model output,"
        " not a transport failure; still scored with local F1)",
        "",
        "## Models (frozen harness; gateway does not attest identity)",
        "",
        f"- reader: `{models['reader']}` ({models['reader_provider']})",
        f"- judge: `{models['judge']}` ({models['judge_provider']})",
        f"- frozen defaults used: {models['frozen_defaults_used']}",
        "",
        "## Run facts",
        "",
        f"- conversation_index: {result['conversation_index']}  sample_id: `{result.get('sample_id')}`",
        f"- dataset: `{result['dataset']['local_name']}` sha256={result['dataset']['sha256']} verified={result['dataset']['verified']}",
        f"- wall_time_s: {result['wall_time_s']}",
        f"- key_pool: {result['key_pool_stats']['keys_total']} keys"
        f" (values withheld); available_after={result['key_pool_stats']['keys_available']}",
        f"- seed: utterances_seeded={result['seed_detail'].get('utterances_seeded')}"
        f" date_facts_seeded={result['seed_detail'].get('date_facts_seeded')}"
        f" sessions_without_date={len(result['seed_detail'].get('sessions_without_date', []))}",
        f"- case workers: {result['workers']}  key retries: {result['max_key_retries']}",
        "",
        "## Top failure reasons",
        "",
    ]
    if failures:
        lines.extend(f"- {item['reason']}: {item['count']}" for item in failures)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in result["limitations"])
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-cases", type=int, default=0,
        help="question cases to run (0 = all questions in the chosen conversation)",
    )
    parser.add_argument(
        "--conversation-index", type=int, default=0,
        help="which of the 10 pinned conversations to run (default 0; ~199 questions)",
    )
    parser.add_argument("--workers", type=int, default=16, help="case workers, capped at 16")
    parser.add_argument("--keys", type=int, default=8, help="use at most N loaded keys (0 = all)")
    parser.add_argument(
        "--keys-file", default=None,
        help="path to the KILO key env file (or set MEMNOTH_KEYS_FILE); a path, never a key value",
    )
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_DIR))
    return parser.parse_args(argv)


def blocked_result(
    *, reason: str, conversation_index: int, selected: int, keys: list[str], sample_id: str | None
) -> dict[str, Any]:
    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "observed_at": utc_now(),
        "official": False,
        "diagnostic": "derived",
        "score_asserted": False,
        "note": NOTE,
        "status": "blocked",
        "reason": reason,
        "conversation_index": conversation_index,
        "sample_id": sample_id,
        "models": {
            "reader": READER_MODEL,
            "judge": JUDGE_MODEL,
            "reader_provider": READER_PROVIDER,
            "judge_provider": JUDGE_PROVIDER,
            "frozen_defaults_used": True,
        },
        "denominators": {
            "attempted": 0, "scored": 0, "judged": 0, "failed": 0, "no_verdict": 0,
            "selected": selected, "resumed_skipped": 0,
        },
        "correct_f1_count": 0,
        "f1_sum": 0.0,
        "f1_mean": 0.0,
        "judge_correct_count": 0,
        "unsupported_count": 0,
        "key_pool_stats": pool_stats_for(keys),
        "seed_detail": {},
        "workers": 0,
        "max_key_retries": MAX_KEY_RETRIES,
        "wall_time_s": 0.0,
        "top_failure_reasons": [],
        "limitations": ["No score is asserted when no real call was made."],
    }
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_cases < 0 or args.keys < 0 or args.workers < 1:
        print("error: --max-cases/--keys must be >= 0 and --workers >= 1")
        return 2

    started = time.monotonic()
    conversation_index = args.conversation_index
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    results_dir = Path(args.results).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"locomo_conv{conversation_index}_checkpoint.jsonl"
    suffix = str(args.max_cases) if args.max_cases else "all"
    result_path = results_dir / f"locomo_conv{conversation_index}_max{suffix}.json"

    # Dataset: pinned URL, SHA-256 verified before use (no gold is seeded).
    try:
        dataset_path, dataset_note = fetch_pinned_dataset(DEFAULT_DATASET_DIR / "locomo10.json")
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"blocker: dataset unavailable ({type(exc).__name__}: {exc})")
        return 2
    if not isinstance(dataset, list) or not dataset:
        print("blocker: pinned dataset is empty or not a list")
        return 2
    if conversation_index < 0 or conversation_index >= len(dataset):
        print(f"blocker: --conversation-index must be 0..{len(dataset) - 1}")
        return 2

    record = dataset[conversation_index]
    if not isinstance(record, dict):
        print("blocker: conversation record is not an object")
        return 2
    sample_id = str(record.get("sample_id") or "")
    conversation = record.get("conversation") if isinstance(record.get("conversation"), dict) else record
    questions = [item for item in (record.get("qa") or []) if isinstance(item, dict)]
    if not questions:
        print("blocker: conversation has no question rows")
        return 2
    if args.max_cases:
        questions = questions[: args.max_cases]

    # CLI binary and model keys.
    bin_path, cli_root = resolve_cli()
    if bin_path is None:
        print("blocker: mn binary not found; set MN_BIN to a public CLI build")
        return 2
    # Pass an explicit --keys-file straight to load_keys(). kilo_pool freezes
    # DEFAULT_KEYS_PATH at import time, so setting the env var here would be too
    # late; the explicit path is the robust route. Counts only; values and the
    # path itself are never printed or written.
    all_keys = load_keys(args.keys_file) if args.keys_file else load_keys()
    keys = all_keys[: args.keys] if args.keys else all_keys
    print(f"keys_loaded: {len(keys)} (values withheld)")
    if not keys:
        result = blocked_result(
            reason="no_keys_configured", conversation_index=conversation_index,
            selected=len(questions), keys=keys, sample_id=sample_id,
        )
        result["dataset"] = {
            "local_name": dataset_path.name, "sha256": DATASET_SHA256, "verified": True,
            "note": dataset_note,
        }
        write_json(result_path, result)
        write_report(REPORT_PATH, result)
        print(f"blocker: no_keys_configured; result: {result_path}")
        return 2

    done = load_done(checkpoint_path)
    target = [
        (index, question)
        for index, question in enumerate(questions)
        if f"locomo-{conversation_index}-{index + 1}" not in done
    ]
    print(
        f"checkpoint_resume: already_done={len(done)} selected={len(questions)} "
        f"target={len(target)}"
    )

    dataset_manifest = {
        "local_name": dataset_path.name,
        "sha256": DATASET_SHA256,
        "bytes": DATASET_BYTES,
        "verified": True,
        "note": dataset_note,
        "url": DATASET_URL,
    }
    models = {
        "reader": READER_MODEL,
        "judge": JUDGE_MODEL,
        "reader_provider": READER_PROVIDER,
        "judge_provider": JUDGE_PROVIDER,
        "frozen_defaults_used": True,
        "note": (
            "Reader receives only the question plus CLI-recalled facts; gold is "
            "used only by the strict judge and local F1. Gateway does not attest "
            "model identity."
        ),
    }

    # Full-resume with everything already checkpointed: preserve the prior real
    # result rather than overwriting it with a zeros "noop" envelope.
    if not target and result_path.is_file():
        print("resume no-op: all selected cases already checkpointed; prior result preserved")
        return 0

    seed_detail: dict[str, Any] = {}
    pool = KeyPool(keys)
    records: list[dict[str, Any]] = []
    seed_note = "not_attempted"
    case_workers = max(1, min(int(args.workers), MAX_WORKERS_CEILING))

    # The temp vault must stay alive for every case (seed once, read many).
    with tempfile.TemporaryDirectory(prefix="mn-bench-locomo-") as temporary:
        vault = Path(temporary) / "vault.db"

        def remember(entity: str, text: str) -> bool:
            return vault_remember(bin_path, cli_root, vault, entity, text)

        all_writes_ok = locomo_seed.seed_conversation(
            remember, conversation, MAX_CLI_WRITE_WORKERS, detail=seed_detail
        )
        seeded = int(seed_detail.get("utterances_seeded", 0)) + int(
            seed_detail.get("date_facts_seeded", 0)
        )
        seed_note = "ok" if all_writes_ok else "partial"
        print(
            f"seed: all_writes_ok={all_writes_ok} utterances={seed_detail.get('utterances_seeded')} "
            f"date_facts={seed_detail.get('date_facts_seeded')} "
            f"timestamps_added={seed_detail.get('timestamps_added')}"
        )
        if seeded == 0:
            result = blocked_result(
                reason="vault_seed_produced_no_facts", conversation_index=conversation_index,
                selected=len(questions), keys=keys, sample_id=sample_id,
            )
            result["dataset"] = dataset_manifest
            result["seed_detail"] = seed_detail
            result["wall_time_s"] = round(time.monotonic() - started, 3)
            write_json(result_path, result)
            write_report(REPORT_PATH, result)
            print("blocker: vault_seed_produced_no_facts")
            return 2

        if target:
            with ThreadPoolExecutor(max_workers=case_workers) as executor:
                futures = {
                    executor.submit(
                        process_case,
                        index=index,
                        question=question,
                        conversation_index=conversation_index,
                        bin_path=bin_path,
                        cli_root=cli_root,
                        vault=vault,
                        pool=pool,
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
    pool_stats = pool.stats()

    attempted = len(records)
    scored = sum(1 for row in records if row["pred"] is not None)
    judged = sum(1 for row in records if row["verdict"] is not None)
    failed = sum(1 for row in records if row["error"] is not None)
    no_verdict = attempted - judged
    correct_f1_count = sum(1 for row in records if row["f1"] == 1.0)
    judge_correct_count = sum(1 for row in records if row["verdict"] == "correct")
    unsupported_count = sum(
        1 for row in records if locomo_reader.is_unsupported(row["pred"])
    )
    f1_sum = round(sum(float(row["f1"]) for row in records), 6)
    f1_mean = round(f1_sum / scored, 6) if scored else 0.0
    failure_counts = Counter(str(row["error"]) for row in records if row["error"])
    top_failures = [
        {"reason": reason, "count": count}
        for reason, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "observed_at": utc_now(),
        "official": False,
        "diagnostic": "derived",
        "score_asserted": False,
        "note": NOTE,
        "status": "completed",
        "lane": "locomo-small",
        "conversation_index": conversation_index,
        "sample_id": sample_id,
        "dataset": dataset_manifest,
        "models": models,
        "denominators": {
            "attempted": attempted,
            "scored": scored,
            "judged": judged,
            "failed": failed,
            "no_verdict": no_verdict,
            "selected": len(questions),
            "resumed_skipped": len(done),
        },
        "correct_f1_count": correct_f1_count,
        "f1_sum": f1_sum,
        "f1_mean": f1_mean,
        "judge_correct_count": judge_correct_count,
        "unsupported_count": unsupported_count,
        "top_failure_reasons": top_failures,
        "key_pool_stats": pool_stats,
        "seed_detail": seed_detail,
        "seed_note": seed_note,
        "workers": case_workers,
        "max_key_retries": MAX_KEY_RETRIES,
        "wall_time_s": wall,
        "checkpoint_path": str(checkpoint_path),
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
        "limitations": [
            "Derived diagnostic on a pinned conversation, not an official LoCoMo split or score.",
            "Token F1 is computed locally against gold; the judge verdict is raw with no smoothing.",
            "A missing verdict is reported as no_verdict, never converted into a correct answer.",
            "Reader/judge identity is recorded from call arguments; the gateway does not attest it.",
            "Gold values and key values are never written; checkpoints carry only a gold hash prefix.",
        ],
    }
    write_json(result_path, result)
    write_report(REPORT_PATH, result)

    print("locomo standalone: official=false diagnostic=derived score_asserted=false")
    print(f"conversation={conversation_index} sample_id={sample_id} max_cases={args.max_cases}")
    print(
        "denominators: "
        f"attempted={attempted} scored={scored} judged={judged} "
        f"failed={failed} no_verdict={no_verdict}"
    )
    print(
        f"f1: correct_f1_count={correct_f1_count} sum={f1_sum}/{scored} "
        f"mean={f1_mean} judge_correct={judge_correct_count} "
        f"unsupported={unsupported_count}"
    )
    print(f"models: reader={READER_MODEL} judge={JUDGE_MODEL}")
    print(f"wall_time_s: {wall}")
    print("top_failure_reasons: " + (json.dumps(top_failures, sort_keys=True) if top_failures else "none"))
    print(f"checkpoint: {checkpoint_path}")
    print(f"result: {result_path}")
    print(f"report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
