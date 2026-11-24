"""Frozen-harness LoCoMo reader and strict judge (derived diagnostic only).

Scope and non-goals
-------------------
The READER receives ONLY the question and CLI-recalled fact texts. It never
receives a dataset gold answer, gold evidence ID, label or speaker hint. The
gold value is passed ONLY to ``judge_answer``, which judges the final answer
alone and never feeds back into the reader. This module fabricates no value: it
forwards the model's short answer or a short error code.

Frozen harness (do not vary to inflate a score):
  reader model = nvidia/nemotron-3-ultra-550b-a55b:free
  judge model  = baidu/ernie-5.5-flash-preview:free   (different provider)
Both are exposed as parameters with those defaults. Reader evidence is ordered
oldest-first by the absolute dates carried in the seeded facts, not by recall
rank. No key, raw HTTP body or prompt text is ever logged; error values are
short codes only.

No score is asserted here; counts travel with denominators in the caller's
result envelope.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any

# Model-call concurrency. The original cap of 2 was for a single key; we now
# rotate across 8 keys via KeyPool, so raise it (env-overridable) and let the
# pool's per-key cooldown handle 429s. CLI/SQLite writes stay capped at 8.
_MODEL_GATE_LIMIT = max(1, min(int(os.environ.get("MEMNOTH_MODEL_GATE", "8")), 16))
_RATE_GATE = threading.Semaphore(_MODEL_GATE_LIMIT)

_MAX_FACTS = 20
_FACT_CHARS = 800
_CONTEXT_CHARS = 6000
_QUESTION_CHARS = 1000
_ANSWER_CHARS = 300
_RETRIES = 3          # retries after the first attempt, on 429 / 5xx / empty answer
_TIMEOUT_SECONDS = 60
# The frozen 550B reader and ernie judge are reasoning models: the final answer
# only appears after ~1k reasoning tokens. BEAM's adapter proved 1024 works on
# this gateway; 256 (reader) and 16 (judge) produced empty content. Never lower.
_READER_MAX_TOKENS = 1024
_JUDGE_MAX_TOKENS = 1024

# ---- FROZEN HARNESS DEFAULTS: pin these; do not vary. -----------------------
DEFAULT_READER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_JUDGE_MODEL = "baidu/ernie-5.5-flash-preview:free"
DEFAULT_BASE_URL = "https://ai-gateway.kilo.ai/api/v1"

READER_PROMPT_7STEP = (
    "You answer a question using ONLY the provided facts.\n"
    "Perform these seven steps internally; output ONLY step 7:\n"
    "1. Identify exactly what the question asks for.\n"
    "2. Find every absolute date stated in the facts. A bracketed prefix such "
    "as [8 May 2023] is the absolute date of that fact.\n"
    "3. Resolve each relative reference (yesterday, last week, two days ago, "
    "the day before) by simple calendar arithmetic on an absolute date stated "
    "in the facts. Never use your own knowledge and never guess a date.\n"
    "4. Read the evidence in the order given: it is oldest-first by absolute "
    "date. Do not reorder it.\n"
    "5. Select only the fact(s) that directly answer the question; ignore "
    "unrelated facts.\n"
    "6. Derive the exact short value from those fact(s). Never invent, infer "
    "from plausibility, or add detail that is not stated in the facts.\n"
    "7. Output ONLY the final value on one line, with no explanation, no "
    "reasoning and no labels. If the value is genuinely absent from the "
    "facts, output UNSUPPORTED.\n"
    "The verifier reads ONLY the final line, not your internal steps."
)

JUDGE_PROMPT_STRICT = (
    "You are a STRICT final-answer verifier. You are given a question, a "
    "reference answer and a candidate answer.\n"
    "- Judge ONLY the final answer value. Ignore reasoning, evidence, "
    "formatting flourishes and confidence.\n"
    "- Decide CORRECT only if the candidate states the SAME single value as "
    "the reference.\n"
    "- Allow only trivial normalization: case, whitespace, punctuation and "
    "unambiguous equivalent date formats (7 May 2023 = May 7, 2023 = "
    "2023-05-07).\n"
    "- Do NOT award partial credit. Do NOT accept a relative expression where "
    "an absolute value is required. Do NOT accept a different but plausible "
    "value.\n"
    "- If either answer is missing or unreadable, answer UNSURE.\n"
    "Respond with exactly one line: CORRECT, INCORRECT, or UNSURE."
)

_GROUNDING_CLAUSE = (
    "Resolve relative references only from absolute dates present in the "
    "facts, read evidence oldest-first, output only the exact value, and say "
    "UNSUPPORTED only when the value is genuinely absent."
)

_UNSUPPORTED_SENTINELS = {
    "unsupported", "un-supported", "un supported", "not supported",
    "no support", "not in facts", "not found in facts",
    "insufficient information", "no information", "unknown", "n/a", "na", "none",
}

_MONTHS = {
    name.casefold(): number
    for number, name in enumerate(
        ("January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"),
        start=1,
    )
}
_BRACKET_RE = re.compile(r"^\s*\[([^\]]+)\]")
_DAY_MONTH_YEAR_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\b")
_MONTH_DAY_YEAR_RE = re.compile(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\s*(am|pm)\b", re.IGNORECASE)


def _parse_bracket_timestamp(text: str) -> tuple[int, int, int, int, int] | None:
    """Parse ONLY a real bracketed date prefix; return None when not parseable.

    No date is ever invented: an unparseable or missing prefix simply opts the
    fact out of date ordering instead of receiving a guessed timestamp.
    """
    match = _BRACKET_RE.match(str(text or ""))
    if not match:
        return None
    value = match.group(1)
    year = month = day = None
    iso = _ISO_DATE_RE.search(value)
    if iso:
        year, month, day = (int(part) for part in iso.groups())
    else:
        dmy = _DAY_MONTH_YEAR_RE.search(value)
        if dmy:
            day, month_name, year = dmy.groups()
            month = _MONTHS.get(str(month_name).casefold())
        else:
            mdy = _MONTH_DAY_YEAR_RE.search(value)
            if mdy:
                month_name, day, year = mdy.groups()
                month = _MONTHS.get(str(month_name).casefold())
    if not year or not month or not day:
        return None
    hour, minute = 0, 0
    clock = _TIME_RE.search(value)
    if clock:
        hour = int(clock.group(1)) % 12
        minute = int(clock.group(2))
        if str(clock.group(3)).casefold() == "pm":
            hour += 12
    return int(year), int(month), int(day), hour, minute


def order_facts_oldest_first(facts: list[str]) -> list[str]:
    """De-duplicate and order evidence by its real absolute date, oldest first.

    Recall rank is deliberately discarded. Facts with no parseable date keep
    their relative input order and are placed after all dated facts; they are
    not assigned an invented date.
    """
    seen: set[str] = set()
    decorated: list[tuple[bool, tuple[int, int, int, int, int], int, str]] = []
    for position, fact in enumerate(facts or []):
        text = str(fact).strip()
        if not text:
            continue
        key = " ".join(text.casefold().split())
        if key in seen:
            continue
        seen.add(key)
        stamp = _parse_bracket_timestamp(text)
        missing = stamp is None
        decorated.append((missing, stamp or (9999, 12, 31, 23, 59), position, text))
    decorated.sort(key=lambda item: item[:3])
    return [item[3] for item in decorated]


def _effective_system_prompt(system_prompt: Any) -> str:
    base = str(system_prompt or "").strip()
    if not base:
        return READER_PROMPT_7STEP
    folded = base.casefold()
    if "oldest-first" in folded and "unsupported" in folded:
        return base
    return f"{base}\n\n{_GROUNDING_CLAUSE}"


def _post_json(
    messages: list[dict[str, str]],
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
) -> tuple[str | None, str | None]:
    """Shared transport: rate gate, bounded retries, short error codes only."""
    key = str(api_key or "").strip()
    if not key:
        return None, "no_key"

    payload: dict[str, Any] = {
        "model": str(model),
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    endpoint = str(base_url or DEFAULT_BASE_URL).rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
        },
    )

    with _RATE_GATE:
        for attempt in range(_RETRIES + 1):
            try:
                with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                    raw = response.read()
            except urllib.error.HTTPError as exc:
                code = int(exc.code)
                if (code == 429 or 500 <= code < 600) and attempt < _RETRIES:
                    time.sleep(min(2**attempt, 8))
                    continue
                return None, f"http_{code}"
            except urllib.error.URLError:
                if attempt < _RETRIES:
                    time.sleep(min(2**attempt, 8))
                    continue
                return None, "network_unavailable"
            except Exception:
                return None, "unavailable"

            try:
                data = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                return None, "bad_response"
            choices: list[Any] = []
            if isinstance(data, dict) and isinstance(data.get("choices"), list):
                choices = data["choices"]
            message = None
            if choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                # A blank/None content is a transient gateway failure (e.g. a
                # reasoning model emitting zero visible tokens), not a real
                # answer. Retry it like 429/5xx within the same bounded budget;
                # no raw body or key is logged, only the short code on exhaustion.
                if attempt < _RETRIES:
                    time.sleep(min(2**attempt, 8))
                    continue
                return None, "empty_answer"
            return content.strip(), None
    return None, "unavailable"


def _compact_facts(facts: list[str]) -> list[str]:
    """Oldest-first, de-duplicated, count-capped and char-budgeted context."""
    selected: list[str] = []
    used = 0
    for fact in order_facts_oldest_first(facts):
        clipped = str(fact)[:_FACT_CHARS]
        if len(selected) >= _MAX_FACTS:
            break
        if selected and used + len(clipped) > _CONTEXT_CHARS:
            break
        selected.append(clipped)
        used += len(clipped)
    return selected


def reader_answer(
    question: str,
    facts: list[str],
    api_key: str,
    *,
    reader_model: str = DEFAULT_READER_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    system_prompt: str = READER_PROMPT_7STEP,
    model: str | None = None,
) -> tuple[str | None, str | None]:
    """Answer from the question and recalled facts alone, using the frozen reader.

    ``model`` is a legacy alias for ``reader_model``; the frozen default is used
    unless the caller explicitly overrides it. Evidence is laid out oldest-first
    before it is sent. Returns ``(answer, None)`` or ``(None, short_error_code)``.
    Gold data is never available to this function.
    """
    chosen_model = str(reader_model if model is None else model)
    selected = _compact_facts(facts)
    user_content = (
        f"Question: {str(question or '')[:_QUESTION_CHARS]}\n"
        "Facts (oldest-first by absolute date):\n"
        + "\n".join(f"- {fact}" for fact in selected)
    )
    content, error = _post_json(
        [
            {"role": "system", "content": _effective_system_prompt(system_prompt)},
            {"role": "user", "content": user_content},
        ],
        chosen_model,
        api_key,
        base_url,
        _READER_MAX_TOKENS,
    )
    if error is not None:
        return None, f"reader_{error}" if error != "no_key" else "reader_no_key"
    return str(content)[:_ANSWER_CHARS], None


def is_unsupported(answer: str | None) -> bool:
    """Robust sentinel check, independent of the reader's casing/punctuation."""
    if answer is None:
        return True
    text = " ".join(str(answer).split()).strip().casefold().strip(" .!?:;_-")
    if not text:
        return True
    if text in _UNSUPPORTED_SENTINELS:
        return True
    return any(
        text == prefix or text.startswith(prefix + " ")
        for prefix in ("unsupported", "un-supported", "un supported", "not supported")
    )


def judge_answer(
    question: str,
    gold: str,
    pred: str,
    api_key: str,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    system_prompt: str = JUDGE_PROMPT_STRICT,
) -> tuple[bool | None, str | None]:
    """Strict final-answer verdict using the frozen, different-provider judge.

    Gold is used here and ONLY here; it is never passed to ``reader_answer``.
    Returns ``(True/False, None)`` for a clean CORRECT/INCORRECT verdict,
    ``(None, 'judge_unsure')`` for UNSURE, or ``(None, short_error_code)`` on a
    transport/parse failure. The verdict is returned raw -- no score smoothing,
    no aggregation, no lenient re-interpretation.
    """
    user_content = (
        f"Question: {str(question or '')[:_QUESTION_CHARS]}\n"
        f"Reference answer: {str(gold or '')[:_ANSWER_CHARS]}\n"
        f"Candidate answer: {str(pred or '')[:_ANSWER_CHARS]}"
    )
    content, error = _post_json(
        [
            {"role": "system", "content": str(system_prompt or JUDGE_PROMPT_STRICT)},
            {"role": "user", "content": user_content},
        ],
        model,
        api_key,
        base_url,
        _JUDGE_MAX_TOKENS,
    )
    if error is not None:
        if error == "no_key":
            return None, "judge_no_key"
        return None, f"judge_{error}"
    first = " ".join(str(content).split()).split(" ")[0].strip(" .!:").upper() if content else ""
    if first == "CORRECT":
        return True, None
    if first == "INCORRECT":
        return False, None
    if first in ("UNSURE", "UNKNOWN"):
        return None, "judge_unsure"
    return None, "judge_bad_verdict"


if __name__ == "__main__":
    import json

    # Offline smoke only: synthetic facts with a relative-date case. No
    # network, no keys, no dataset access, no score. Empty keys make both
    # calls return before any request can be built.
    synthetic_facts = [
        "[9 June 2023] Melanie: We met on Tuesday.",
        "Melanie: I joined the LGBTQ support group yesterday.",
        "[8 May 2023] Melanie: I joined the LGBTQ support group yesterday.",
        "[8 May 2023] Melanie: I joined the LGBTQ support group yesterday.",
        "[1 January 2022] Caroline: I painted a sunrise last year.",
    ]
    question = "When did Melanie join the LGBTQ support group?"
    answer, answer_error = reader_answer(question, synthetic_facts, "")
    verdict, verdict_error = judge_answer(question, "7 May 2023", "8 May 2023", "")
    print(
        json.dumps(
            {
                "note": "INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE",
                "diagnostic": "derived",
                "frozen_reader_model": DEFAULT_READER_MODEL,
                "frozen_judge_model": DEFAULT_JUDGE_MODEL,
                "oldest_first_facts": order_facts_oldest_first(synthetic_facts),
                "offline_answer": answer,
                "offline_answer_error": answer_error,
                "offline_verdict": verdict,
                "offline_verdict_error": verdict_error,
                "reader_prompt_steps": len(re.findall(r"^\d\.", READER_PROMPT_7STEP, re.MULTILINE)),
                "reader_prompt_mentions_oldest_first": "oldest-first" in READER_PROMPT_7STEP.casefold(),
                "judge_prompt_is_strict": "Do NOT award partial credit" in JUDGE_PROMPT_STRICT,
                "sentinel_checks": {
                    "UNSUPPORTED": is_unsupported("UNSUPPORTED"),
                    "unsupported.": is_unsupported("unsupported."),
                    "7 May 2023": is_unsupported("7 May 2023"),
                    "None (no answer)": is_unsupported(None),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
