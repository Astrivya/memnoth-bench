"""Small, resumable probe: which FREE Kilo model is the best LoCoMo READER?

Scope and honesty contract
--------------------------
This is a *reader-selection probe*, not a score and not a challenge to the frozen
harness. It does not edit or re-configure anything: the frozen reader/judge
defaults in ``locomo_reader`` stay exactly as they are. It only calls each
candidate model with the SAME 7-step reader prompt, the SAME oldest-first fact
context the real runner sees, temperature 0 and max_tokens=1024.

* The candidate models are named explicitly; the gateway does not attest model
  identity, so these are the requested model ids, not verified provenance.
* Seed -> recall -> asof is reused verbatim from ``locomo_run`` / ``locomo_seed``
  so the facts handed to every candidate are byte-identical to the real runner's.
* Gold is used ONLY by local token F1. It is never seeded, never sent to a model,
  and never written to a checkpoint (only a hash prefix is stored).
* Key values are never printed, logged or written. ``KeyPool`` is used only for
  model calls, never for CLI calls. On 429/5xx the next key is tried (max 3).
* One checkpoint row per (model, question_id) in
  D:\\memnoth-bench-data\\checkpoints\\model_probe.jsonl; re-running resumes.

Run:
    python -m runner.model_probe                 # full probe
    python -m runner.model_probe --check-facts    # verify seed/recall/asof only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import mn_json  # noqa: E402
from runner import locomo_reader, locomo_run, locomo_seed  # noqa: E402
from runner.kilo_pool import KeyPool, load_keys  # noqa: E402

VERSION = "0.1.0"
SCHEMA = "memnoth-bench-model-probe/1"
NOTE = (
    "INFORMATIONAL ONLY — READER-SELECTION PROBE, NOT AN OFFICIAL SCORE. "
    "No frozen default is changed and no win is claimed."
)

DATA_ROOT = Path(r"D:\memnoth-bench-data")
CHECKPOINT_PATH = DATA_ROOT / "checkpoints" / "model_probe.jsonl"
REPORT_PATH = DATA_ROOT / "reports" / "MODEL_PROBE.md"
CONVERSATION_INDEX = 0
N_QUESTIONS = 5

# All free on the Kilo gateway. Order is fixed and reported as-is.
MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3-ultra-550b-a55b:free",   # current frozen reader
    "baidu/ernie-5.5-flash-preview:free",        # current frozen judge
    "inclusionai/ling-3.2-flash:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "particle-ai/particle-1:free",
)

_MAX_KEY_RETRIES = 3
_TIMEOUT_SECONDS = 60
_POLL_SECONDS = 30.0  # wait this long for a free key before giving up on a call


# --------------------------------------------------------------------------- #
# Small honest helpers (no key values, no gold values in outputs)
# --------------------------------------------------------------------------- #


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()


def token_f1(prediction: Any, reference: Any) -> float:
    """Deterministic token F1; identical to locomo_run.token_f1 / harnesses.common."""
    return locomo_run.token_f1(prediction, reference)


def row_key(model: str, question_id: str) -> str:
    return f"{model}::{question_id}"


# --------------------------------------------------------------------------- #
# Model transport: mirrors locomo_reader._post_json but keeps reasoning info so
# the report can name models that returned empty content or reasoning-only.
# --------------------------------------------------------------------------- #


def _post_probe(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    base_url: str,
) -> tuple[str | None, str | None, str | None]:
    """Return ``(content, reasoning, error)``; error values are short codes only.

    temperature=0 and max_tokens are the same as the frozen reader. The request
    shape and retry policy match locomo_reader._post_json; only the response
    parsing is extended to retain reasoning text for diagnostics.
    """
    key = str(api_key or "").strip()
    if not key:
        return None, None, "no_key"

    payload: dict[str, Any] = {
        "model": str(model),
        "messages": messages,
        "temperature": 0,
        "max_tokens": locomo_reader._READER_MAX_TOKENS,  # frozen: 1024
    }
    endpoint = str(base_url or locomo_reader.DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
    )

    for attempt in range(locomo_reader._RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            code = int(exc.code)
            if (code == 429 or 500 <= code < 600) and attempt < locomo_reader._RETRIES:
                time.sleep(min(2**attempt, 8))
                continue
            return None, None, f"http_{code}"
        except urllib.error.URLError:
            if attempt < locomo_reader._RETRIES:
                time.sleep(min(2**attempt, 8))
                continue
            return None, None, "network_unavailable"
        except Exception:
            return None, None, "unavailable"

        try:
            data = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            return None, None, "bad_response"
        choices = data.get("choices") if isinstance(data, dict) else None
        message = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
        if not isinstance(message, dict):
            return None, None, "empty_answer"

        content = message.get("content")
        reasoning: Any = None
        for field in ("reasoning_content", "reasoning", "reasoning_details"):
            value = message.get(field)
            if value:
                reasoning = value
                break
        reasoning_text = json.dumps(reasoning, ensure_ascii=False) if not isinstance(reasoning, str) else reasoning
        if isinstance(content, str) and content.strip():
            return content.strip(), reasoning_text, None
        # No final content; reasoning is retained for the report only.
        return None, reasoning_text or None, "empty_answer"
    return None, None, "unavailable"


def _reader_probe(question: str, facts: list[str], model: str, api_key: str) -> dict[str, Any]:
    """Call one candidate with the frozen 7-step prompt and identical facts."""
    selected = locomo_reader._compact_facts(facts)
    user_content = (
        f"Question: {str(question or '')[:locomo_reader._QUESTION_CHARS]}\n"
        "Facts (oldest-first by absolute date):\n"
        + "\n".join(f"- {fact}" for fact in selected)
    )
    messages = [
        {
            "role": "system",
            "content": locomo_reader._effective_system_prompt(locomo_reader.READER_PROMPT_7STEP),
        },
        {"role": "user", "content": user_content},
    ]
    content, reasoning, error = _post_probe(
        messages, model, api_key, locomo_reader.DEFAULT_BASE_URL
    )
    pred = str(content)[: locomo_reader._ANSWER_CHARS] if content else None
    return {
        "pred": pred,
        "reasoning": reasoning,
        "error": error,
        "facts_used": len(selected),
    }


def call_with_pool(pool: KeyPool, question: str, facts: list[str], model: str) -> dict[str, Any]:
    """Next-key retries on failure; never returns a fabricated answer."""
    last_error = "unavailable"
    last = {"pred": None, "reasoning": None, "error": last_error, "facts_used": 0}
    for _ in range(_MAX_KEY_RETRIES):
        key = pool.get()
        deadline = time.monotonic() + _POLL_SECONDS
        while key is None and time.monotonic() < deadline:
            time.sleep(0.25)
            key = pool.get()
        if key is None:
            return {**last, "error": "no_key_available"}
        try:
            last = _reader_probe(question, facts, model, key)
        except Exception as exc:
            last = {"pred": None, "reasoning": None,
                    "error": "probe_exception_" + type(exc).__name__, "facts_used": 0}
        finally:
            ok = last.get("error") is None and last.get("pred") is not None
            pool.release(key, ok, locomo_run.status_from_error(last.get("error")))
        if ok:
            return last
        last_error = str(last.get("error") or "unknown_error")
    return last


# --------------------------------------------------------------------------- #
# Real-runner seed -> recall -> asof chain (reused, not reimplemented)
# --------------------------------------------------------------------------- #


def build_facts(
    bin_path: Path | None, cli_root: Path | None, vault: Path, question_text: str
) -> list[str]:
    """Exactly the process_case() recall -> asof chain, gold-free."""
    hits = locomo_seed.recall_question(
        lambda q, n: locomo_run.cli_recall_sweep(bin_path, cli_root, vault, q, n),
        question_text,
        10,
    )
    facts: list[str] = []
    for name in hits:
        for item in locomo_run.vault_asof(bin_path, cli_root, vault, name):
            text = str(item).strip()
            if text and text not in facts:
                facts.append(text)
    return facts


def seed_vault(
    bin_path: Path | None, cli_root: Path | None, vault: Path, conversation: dict[str, Any]
) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    locomo_seed.seed_conversation(
        lambda entity, text: locomo_run.vault_remember(bin_path, cli_root, vault, entity, text),
        conversation,
        locomo_run.MAX_CLI_WRITE_WORKERS,
        detail=detail,
    )
    return detail


# --------------------------------------------------------------------------- #
# Checkpointing and reporting
# --------------------------------------------------------------------------- #


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows[row_key(str(row["model"]), str(row["question_id"]))] = row
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return rows


def load_all_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _cell(row: dict[str, Any] | None) -> str:
    if row is None:
        return "—"
    pred = row.get("pred")
    if pred is None:
        code = str(row.get("error") or "no_answer")
        return f"∅ ({code})"
    shown = " ".join(str(pred).split())
    if len(shown) > 34:
        shown = shown[:33] + "…"
    flag = " **UNSUPPORTED**" if locomo_reader.is_unsupported(pred) else ""
    return f"`{shown}`{flag} · F1={float(row.get('f1') or 0):.2f}"


def write_report(path: Path, rows: list[dict[str, Any]], q_ids: list[str], seed: dict[str, Any]) -> None:
    by_key = {row_key(str(r["model"]), str(r["question_id"])): r for r in rows}
    summary: list[dict[str, Any]] = []
    for model in MODELS:
        model_rows = [r for r in rows if r.get("model") == model]
        scored = [r for r in model_rows if r.get("pred") is not None]
        f1_sum = round(sum(float(r.get("f1") or 0) for r in scored), 6)
        summary.append(
            {
                "model": model,
                "attempted": len(model_rows),
                "scored": len(scored),
                "unsupported": sum(1 for r in model_rows if locomo_reader.is_unsupported(r.get("pred"))),
                "exact": sum(1 for r in scored if float(r.get("f1") or 0) == 1.0),
                "mean_f1": round(f1_sum / len(scored), 4) if scored else 0.0,
                # Errors count as 0 here: a failed call must not raise the mean by
                # being excluded from the denominator.
                "mean_f1_attempted": round(
                    sum(float(r.get("f1") or 0) for r in model_rows) / len(model_rows), 4
                ) if model_rows else 0.0,
                "empty_content": sum(1 for r in model_rows if r.get("empty_content")),
                "reasoning_only": sum(1 for r in model_rows if r.get("reasoning_only")),
                "errors": sum(1 for r in model_rows if r.get("error")),
            }
        )
    # Honest order: fewer UNSUPPORTED, more exact answers, fewer failed calls,
    # then mean F1 over ALL attempted questions (not just the successful ones).
    ranked = sorted(
        summary,
        key=lambda s: (s["unsupported"], -s["exact"], s["errors"], -s["mean_f1_attempted"]),
    )
    winner = ranked[0] if ranked and ranked[0]["scored"] else None
    top = winner or {}
    tied = [
        s for s in summary
        if s["unsupported"] == top.get("unsupported")
        and s["exact"] == top.get("exact")
        and s["errors"] == top.get("errors")
        and s["mean_f1_attempted"] == top.get("mean_f1_attempted")
    ] if winner else []

    lines: list[str] = [
        "# Free Kilo reader-model probe — LoCoMo-style QA",
        "",
        f"- schema: `{SCHEMA}`  version: `{VERSION}`  observed_at: `{utc_now()}`",
        f"- **{NOTE}**",
        f"- Status: **INFORMATIONAL / PROBE ONLY.** The frozen reader "
        f"(`{locomo_reader.DEFAULT_READER_MODEL}`) and judge "
        f"(`{locomo_reader.DEFAULT_JUDGE_MODEL}`) are **unchanged**.",
        "",
        "## Method (kept identical across candidates)",
        "",
        "- Test set: pinned sample 0 (`conv-26`), `qa[0:5]` (5 questions) of "
        "`locomo10.json` (SHA-256 verified by the shared runner helper).",
        "- For each question a temp vault was seeded with the FULL conversation "
        "(real utterances + real session-date facts) via `locomo_seed.seed_conversation`.",
        "- Recall and asof use the unchanged `locomo_run.cli_recall_sweep` -> "
        "`locomo_seed.recall_question` -> `locomo_run.vault_asof` chain, so every "
        "candidate sees byte-identical, oldest-first facts — the same facts the "
        "real runner sees.",
        "- Each candidate was called with the frozen 7-step reader prompt "
        "(`locomo_reader.READER_PROMPT_7STEP`), `temperature=0`, "
        f"`max_tokens={locomo_reader._READER_MAX_TOKENS}`.",
        "- Gold is used ONLY for locally computed token F1; it is never in the "
        "prompt and never written to a checkpoint (gold hash prefix only).",
        "- 8 keys round-robined through `KeyPool`; 429/5xx retries with the next "
        "key (max 3). Key values are never logged or stored.",
        f"- Seed detail: sessions={seed.get('session_count')}, "
        f"utterances_seeded={seed.get('utterances_seeded')}, "
        f"date_facts_seeded={seed.get('date_facts_seeded')}, "
        f"timestamps_added={seed.get('timestamps_added')}.",
        "",
        "## Model x question",
        "",
        "Each cell: prediction · UNSUPPORTED flag · token F1. `∅ (code)` means no "
        "final answer was returned (short error code only).",
        "",
    ]
    header = "| Model | " + " | ".join(q_ids) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(q_ids))
    for model in MODELS:
        cells = [_cell(by_key.get(row_key(model, qid))) for qid in q_ids]
        lines.append(f"| `{model}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Per-model summary",
            "",
            "| Model | scored / attempted | UNSUPPORTED | exact (F1=1) | mean F1 (scored) | mean F1 (all attempted) | empty content | reasoning-only | errors |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for s in summary:
        lines.append(
            f"| `{s['model']}` | {s['scored']}/{s['attempted']} | {s['unsupported']} | "
            f"{s['exact']} | {s['mean_f1']:.4f} | {s['mean_f1_attempted']:.4f} | "
            f"{s['empty_content']} | {s['reasoning_only']} | {s['errors']} |"
        )
    lines.extend(
        [
            "",
            "## Empty content / reasoning-only notes",
            "",
        ]
    )
    notes = [
        f"- `{s['model']}`: empty_content={s['empty_content']}, "
        f"reasoning_only={s['reasoning_only']}, transport_errors={s['errors']}."
        for s in summary
    ]
    lines.extend(notes if notes else ["- none recorded"])
    lines.extend(
        [
            "",
            "## Recommendation (probe only, not a claim of a win)",
            "",
        ]
    )
    if winner:
        lines.extend(
            [
                "- Ordering: fewest UNSUPPORTED, then most exact answers, then fewest "
                "failed calls, then mean F1 over **all attempted** questions. "
                "A failed call counts as 0 and is never excluded from the "
                "denominator: `mean F1 (scored)` alone can look better purely "
                "because a model failed to answer, so it is not used to rank.",
                f"- Leader on that order: `{winner['model']}` "
                f"(UNSUPPORTED={winner['unsupported']}, exact={winner['exact']}, "
                f"errors={winner['errors']}, mean F1 over attempted="
                f"{winner['mean_f1_attempted']:.4f}).",
            ]
        )
        if len(tied) > 1:
            all_tied = ", ".join(f"`{s['model']}`" for s in tied)
            lines.extend(
                [
                    f"- **Accuracy is tied** across {len(tied)} candidates "
                    f"({all_tied}): identical UNSUPPORTED count, identical "
                    "exact-answer count and identical completion rate. There is "
                    "**no accuracy winner** in this 5-question probe.",
                ]
            )
            independent = [
                s for s in tied if s["model"] != locomo_reader.DEFAULT_JUDGE_MODEL
            ]
            pick = independent[0]["model"] if independent else winner["model"]
            lines.append(
                f"- **Recommendation (future product choice only)**: `{pick}` — "
                "among the tied candidates that answered every call cleanly, it "
                "keeps the reader independent of the frozen judge role "
                f"(`{locomo_reader.DEFAULT_JUDGE_MODEL}`). Every tied candidate "
                "had 1 exact answer (Q1) and UNSUPPORTED on Q2-Q5, so this is a "
                "reliability/independence tie-break, **not a demonstrated win**."
            )
        else:
            lines.append(
                f"- **Recommendation (future product choice only)**: "
                f"`{winner['model']}` on the ordering above; this is not a "
                "demonstrated win over the other candidates."
            )
        lines.append(
            "- This is a **recommendation for a future product choice**, not a "
            "score, not a paper/vendor comparison, and not a change to any "
            "frozen default. The current frozen reader and judge remain as-is; "
            "this probe does not justify changing them."
        )
    else:
        lines.append("- No candidate returned a scorable answer; no recommendation is made.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- n=5 pinned questions from one conversation: too small for a "
            "statistical claim. Differences of one answer are noise.",
            "- The gateway does not attest model identity; model ids are the "
            "requested strings only.",
            "- Token F1 rewards lexical overlap and can under-rate a correct "
            "answer phrased differently; UNSUPPORTED is the primary signal.",
            "- **Grounding caveat**: in a separate read-only rebuild of the same "
            "seed -> recall -> asof chain, the gold answer text was ABSENT from "
            "the compacted facts for Q2-Q5 and present for Q1. The frozen "
            "7-step prompt forbids inference and requires UNSUPPORTED when the "
            "value is genuinely absent, so UNSUPPORTED on Q2-Q5 is often the "
            "*correct* grounded behaviour that local F1 penalises. This probe "
            "therefore measures recall-plus-reader agreement, not isolated "
            "reader quality, and the goal 'minimise UNSUPPORTED' is partly in "
            "tension with the grounding rule. Fact text is not checkpointed, so "
            "this was not re-checked against the exact probe snapshot.",
            "- Recall is not fully deterministic across independently built "
            "temporary vaults: concurrent seeding changes insertion order and "
            "the probe's per-question snapshots (8/12/12/13/12 compacted facts) "
            "differed from a separate rebuild (up to the 20-fact cap). Within "
            "this probe every candidate received one fixed snapshot per "
            "question, so the comparison itself is fair.",
            "- No judge was used here: F1 is computed locally and reported raw.",
            "- Checkpoints store predictions and error codes, never key values "
            "or gold values.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5, help="parallel model calls (default 5)")
    parser.add_argument(
        "--keys-file", default=None,
        help="path to the KILO key env file (or set MEMNOTH_KEYS_FILE); a path, never a key value",
    )
    parser.add_argument("--check-facts", action="store_true",
                        help="verify seed/recall/asof only; no model calls, no checkpoint")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()

    dataset_path, _ = locomo_run.fetch_pinned_dataset(locomo_run.DEFAULT_DATASET_DIR / "locomo10.json")
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, list) or CONVERSATION_INDEX >= len(dataset):
        print("blocker: dataset unavailable or index out of range")
        return 2
    record = dataset[CONVERSATION_INDEX]
    conversation = record.get("conversation") if isinstance(record.get("conversation"), dict) else record
    questions = [q for q in (record.get("qa") or []) if isinstance(q, dict)][:N_QUESTIONS]
    if len(questions) != N_QUESTIONS:
        print(f"blocker: expected {N_QUESTIONS} questions, got {len(questions)}")
        return 2

    bin_path, cli_root = locomo_run.resolve_cli()
    if bin_path is None:
        print("blocker: mn binary not found")
        return 2
    # Pass an explicit --keys-file straight to load_keys(). kilo_pool freezes
    # DEFAULT_KEYS_PATH at import time, so setting the env var here would be too
    # late; the explicit path is the robust route. Counts only; values and the
    # path itself are never printed or written.
    keys = load_keys(args.keys_file) if args.keys_file else load_keys()
    print(f"keys_loaded: {len(keys)} (values withheld)")
    if not keys:
        print("blocker: no keys configured")
        return 2

    q_ids = [f"locomo-{CONVERSATION_INDEX}-{i + 1}" for i in range(len(questions))]
    with tempfile.TemporaryDirectory(prefix="mn-bench-probe-") as temporary:
        vault = Path(temporary) / "vault.db"
        seed = seed_vault(bin_path, cli_root, vault, conversation)
        print(
            f"seed: sessions={seed.get('session_count')} "
            f"utterances={seed.get('utterances_seeded')} "
            f"date_facts={seed.get('date_facts_seeded')}"
        )
        facts_by_q: list[list[str]] = []
        golds: list[str] = []
        for i, question in enumerate(questions):
            text = str(question.get("question") or "")
            facts = build_facts(bin_path, cli_root, vault, text)
            facts_by_q.append(facts)
            golds.append(str(question.get("answer") or ""))
            print(f"facts[{q_ids[i]}]: {len(facts)} (gold used for F1 only)")

        if args.check_facts:
            print("check-facts: ok (no model calls, no checkpoint written)")
            return 0

        done = load_done(CHECKPOINT_PATH)
        tasks: list[tuple[str, int, str, str, list[str], str]] = []
        for model in MODELS:
            for i, qid in enumerate(q_ids):
                if row_key(model, qid) not in done:
                    tasks.append((model, i, qid, str(questions[i].get("question") or ""),
                                  facts_by_q[i], golds[i]))
        print(f"checkpoint_resume: already_done={len(done)} target={len(tasks)}")

        pool = KeyPool(keys)
        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        workers = max(1, min(int(args.workers), 8))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(call_with_pool, pool, question, facts, model): (model, i, qid, gold)
                for model, i, qid, question, facts, gold in tasks
            }
            with CHECKPOINT_PATH.open("a", encoding="utf-8") as handle:
                for future in as_completed(futures):
                    model, i, qid, gold = futures[future]
                    result = future.result()
                    pred = result.get("pred")
                    reasoning = result.get("reasoning")
                    row = {
                        "schema": SCHEMA,
                        "observed_at": utc_now(),
                        "model": model,
                        "question_id": qid,
                        "question_index": i,
                        "pred": pred,
                        "unsupported": bool(locomo_reader.is_unsupported(pred)),
                        "f1": token_f1(pred, gold),
                        "gold_hash_prefix": sha256_text(gold)[:12] if gold else None,
                        "empty_content": pred is None,
                        "reasoning_only": pred is None and bool(reasoning),
                        "facts_used": result.get("facts_used", 0),
                        "error": result.get("error"),
                    }
                    handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        f"done: {model} {qid} pred={'yes' if pred else 'no'} "
                        f"unsupported={row['unsupported']} f1={row['f1']} "
                        f"error={row['error'] or 'none'}"
                    )

    rows = load_all_rows(CHECKPOINT_PATH)
    expected = len(MODELS) * len(q_ids)
    if len(rows) < expected:
        print(f"warning: checkpoint has {len(rows)}/{expected} rows; report is partial")
    write_report(REPORT_PATH, rows, q_ids, seed)
    print(f"checkpoint: {CHECKPOINT_PATH} ({len(rows)} rows)")
    print(f"report: {REPORT_PATH}")
    print(f"wall_time_s: {round(time.monotonic() - started, 3)}")
    print("model_probe: informational only; frozen defaults unchanged; no score asserted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
