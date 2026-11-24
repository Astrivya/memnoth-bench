"""Pinned BEAM adapter -- 1M and 10M tiers (ICLR 2026, arXiv 2510.27246).

WHAT THIS IS
------------
A structural, derived diagnostic harness. It is NOT an official BEAM run and
asserts no score. BEAM publishes conversations and probing questions, but the
paper does not fix one reader/judge pair: vendors pick their own, and the same
answers rescore differently under different scorers. We therefore FREEZE one
harness (READER_MODEL / JUDGE_MODEL below), keep the reader and judge on
different providers, publish raw verdicts, and never shop for a lenient judge.

NO GOLD LEAKAGE
---------------
The reader receives only the question text and conversation/CLI-recalled facts.
Ideal answers, rubric nuggets and annotations are used only by ``evaluate`` and
by the judge, never by the reader. No gold answer is written into a reader
prompt, and no gold-derived hint is injected into the context formatter.

DERIVED METRIC
--------------
The primary metric is a deterministic SubEM/F1 proxy computed from the dataset's
own ideal-answer strings. It is NOT the official BEAM nugget score (0.0/0.5/1.0
per rubric item with Kendall tau-b for event ordering). An optional judge call
is reported separately with its RAW verdict preserved; it never replaces the
proxy. Every result is stamped official=False / diagnostic="derived" / Tier C.
"""

from __future__ import annotations

import ast
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

try:  # normal package import
    from ..common import (
        apply_judge,
        denominator,
        fetch_pinned,
        fixture_section,
        limiter,
        ratio,
        ratio_text,
        stable_rows,
        stamps,
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
        ratio,
        ratio_text,
        stable_rows,
        stamps,
        token_f1,
    )

LANE = "beam"
SOURCE_URL = "https://github.com/mohammadtavakoli78/BEAM"
PAPER_URL = "https://arxiv.org/abs/2510.27246"

# --------------------------------------------------------------------------- #
# PINS -- verified 2026-11-24. Never guess a hash; compute it from exact bytes.
# --------------------------------------------------------------------------- #
# Resolved upstream HEAD from https://api.github.com/repos/mohammadtavakoli78/BEAM/commits/main
UPSTREAM_COMMIT = "b2da22eac88bb0874c64665f13457eb99835774a"

# HF dataset commits (resolved, immutable). Provenance:
#   https://huggingface.co/api/datasets/Mohammadta/BEAM/commits/main
#   https://huggingface.co/api/datasets/Mohammadta/BEAM-10M/commits/main
HF_COMMIT_1M = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
HF_COMMIT_10M = "9b2096193fe74e2837e4713e483351e19817773c"

# Each SHA-256 below was verified TWO ways: the HF tree API reports it as the
# Git-LFS `oid` at the pinned commit, and it was recomputed with hashlib over
# the full downloaded artifact (exact byte counts recorded per file). No hash
# was invented. Probing questions are embedded in these same parquet files;
# there is no separate probe artifact.
_DATA_1M = {
    "path": "data/1M-00000-of-00001.parquet",
    "url": (
        "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/"
        f"{HF_COMMIT_1M}/data/1M-00000-of-00001.parquet"
    ),
    "sha256": "41b5acbbb55a586b1305514ef9d9fb03365d9b3331b598a1c2dd7603d93ef533",
    "bytes": 66156374,
}
_DATA_10M_0 = {
    "path": "data/10M-00000-of-00002.parquet",
    "url": (
        "https://huggingface.co/datasets/Mohammadta/BEAM-10M/resolve/"
        f"{HF_COMMIT_10M}/data/10M-00000-of-00002.parquet"
    ),
    "sha256": "31d96fd47ec56221d202e68792f26c00e49467dd4b36ee105c36ebd19ef78ad5",
    "bytes": 153844664,
}
_DATA_10M_1 = {
    "path": "data/10M-00001-of-00002.parquet",
    "url": (
        "https://huggingface.co/datasets/Mohammadta/BEAM-10M/resolve/"
        f"{HF_COMMIT_10M}/data/10M-00001-of-00002.parquet"
    ),
    "sha256": "a4f13fe25af51d57405ae41008689c31d1421377f3efde56a024b441deb2ee65",
    "bytes": 189980875,
}

TIERS: dict[str, dict[str, Any]] = {
    "beam-1m": {
        "tier": "1M",
        "hf_repo": "Mohammadta/BEAM",
        "hf_commit": HF_COMMIT_1M,
        "split": "1M",
        "conversations": 35,
        "questions_per_conversation": 20,
        "files": (_DATA_1M,),
    },
    "beam-10m": {
        "tier": "10M",
        "hf_repo": "Mohammadta/BEAM-10M",
        "hf_commit": HF_COMMIT_10M,
        "split": "10M",
        "conversations": 10,
        "questions_per_conversation": 20,
        "files": (_DATA_10M_0, _DATA_10M_1),
    },
}
DEFAULT_TIER = "beam-1m"

# Interface-compatible default pins (the 1M tier). Multi-shard tiers use TIERS.
DATASET_URL = _DATA_1M["url"]
DATASET_SHA256 = _DATA_1M["sha256"]

ABILITIES: tuple[str, ...] = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)

# The dataset uses different ideal-answer field names per ability.
_GOLD_FIELDS: dict[str, tuple[str, ...]] = {
    "abstention": ("ideal_response", "answer", "ideal_answer"),
    "contradiction_resolution": ("ideal_answer", "answer"),
    "event_ordering": ("answer",),
    "information_extraction": ("answer",),
    "instruction_following": ("expected_compliance",),
    "knowledge_update": ("answer",),
    "multi_session_reasoning": ("answer",),
    "preference_following": ("expected_compliance",),
    "summarization": ("ideal_summary", "answer"),
    "temporal_reasoning": ("answer",),
}

# --------------------------------------------------------------------------- #
# FROZEN HARNESS -- defaults. Do not vary per run; that is judge shopping.
# --------------------------------------------------------------------------- #
READER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
JUDGE_MODEL = "baidu/ernie-5.5-flash-preview:free"
READER_PROVIDER = "kilo"
JUDGE_PROVIDER = "kilo"
READER_BASE_URL = "https://ai-gateway.kilo.ai/api/v1"
JUDGE_BASE_URL = "https://ai-gateway.kilo.ai/api/v1"
ANSWER_SOURCE = "reader over CLI-recalled facts"

# The runner formats the whole pinned conversation (millions of characters) and
# the reader has a 1M-token window. A hard 12k-character slice discarded >99% of
# the evidence and caused blanket UNSUPPORTED. Reader-side evidence selection is
# bounded instead, and every character it sees still comes from conversation
# facts (never from gold, rubric or annotations).
READER_CONTEXT_CHARS = 90_000

READER: dict[str, Any] = {
    "model": READER_MODEL,
    "provider": READER_PROVIDER,
    "context_window": 1_000_000,
    "pinned": True,
    "enforced_by_adapter": False,
    "note": (
        "Frozen default. The caller must pass this model to reader_call and "
        "record it; the adapter cannot verify a remote model's identity."
    ),
}
JUDGE: dict[str, Any] = {
    "model": JUDGE_MODEL,
    "provider": JUDGE_PROVIDER,
    "temperature": 0,
    "pinned": True,
    "enforced_by_adapter": False,
    "note": (
        "Frozen default, on a different provider from the reader. Raw verdicts "
        "are published; the paper fixes no judge, so this is not an official "
        "BEAM scorer. Third-party BEAM reports cite gpt-5 judges and the pinned "
        "upstream evaluation script uses gpt-4.1-mini; neither is reproduced here."
    ),
}

# Retrieval profile used by the surrounding system; encoded here so a figure is
# attributable to one frozen configuration. This adapter only formats context.
RETRIEVAL_PROFILE: dict[str, Any] = {
    "absolute_date_grounding": True,
    "chronological_evidence_layout": True,
    "atomic_facts_with_speaker_and_timestamp": True,
    "hybrid_recall_rrf_rerank": True,
    # None = render every supplied fact. The runner calls format_context without
    # a question, so reader_call performs the question-conditioned selection.
    "top_k": None,
    "multi_query_decomposition": True,
    "sufficiency_check": True,
    "temporal_invalidation_versioned_facts": True,
    "note": (
        "Retrieval itself lives in the CLI harness. format_context renders the "
        "facts it is given (top_k=None means all of them); reader_call then "
        "selects question-relevant fact lines under READER_CONTEXT_CHARS. The "
        "runner passes no question to format_context, so selection has to happen "
        "at the reader boundary. No gold or rubric text is ever in scope."
    ),
}

_NOTE = (
    "Derived BEAM 1M/10M diagnostic. official=False, Tier C: no paired "
    "same-pin comparison exists, and the BEAM paper does not fix a reader or "
    "judge, so this figure is not comparable to paper (LIGHT/RAG, "
    "Llama-4-Maverick, paper scorer) or vendor leaderboard rows. SubEM/F1 is a "
    "deterministic proxy, not the official BEAM nugget score. Raw judge verdicts "
    "are retained and never selected for leniency."
)


def _resolve_tier(tier: str | None) -> tuple[str, dict[str, Any]]:
    name = tier or DEFAULT_TIER
    if name not in TIERS:
        raise ValueError(f"unknown BEAM tier: {name!r}; choose from {sorted(TIERS)}")
    return name, TIERS[name]


def prepare(local_dir: str | Path = "data/beam", tier: str | None = None) -> Path:
    """Download and SHA-256-verify every shard of ``tier`` under ``data/``.

    Returns the first verified shard path. The 10M tier needs both shards; all
    are downloaded and verified before this returns. This is the only network
    boundary and is never invoked at import time or by ``--smoke``.
    """
    _name, spec = _resolve_tier(tier)
    root = Path(local_dir)
    paths: list[Path] = []
    for item in spec["files"]:
        dest = root / f"{spec['tier']}-{Path(item['path']).name}"
        paths.append(fetch_pinned(item["url"], item["sha256"], dest))
    return paths[0]


def select(max_cases: int | None = None, tier: str | None = None) -> dict[str, Any]:
    """Deterministic pinned selection description; no network, no padding."""
    name, spec = _resolve_tier(tier)
    planned = int(spec["conversations"]) * int(spec["questions_per_conversation"])
    ordinals = [position + 1 for position in limiter(range(planned), max_cases)]
    section, found = fixture_section(LANE)
    ids = [str(value) for value in limiter(section.get("ids", []), max_cases)] if found else []
    return {
        "lane": LANE,
        "tier": name,
        "split": spec["split"],
        "source": SOURCE_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "hf_repo": spec["hf_repo"],
        "hf_commit": spec["hf_commit"],
        "dataset_url": spec["files"][0]["url"],
        "dataset_sha256": spec["files"][0]["sha256"],
        "files": [
            {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]}
            for item in spec["files"]
        ],
        "fixture_found": found,
        "planned_denominator": planned,
        "effective_denominator": denominator(ordinals),
        "ordinals": ordinals,
        "ids": ids,
        "ids_pinned": bool(ids),
        "id_resolution": "sort conversations by conversation_id, then ability order, then question index",
        "official_split": bool(section.get("official_split", False)),
        "note": _NOTE,
    }


def _parse_questions(raw: Any) -> Any:
    if isinstance(raw, str):
        return ast.literal_eval(raw)
    return raw


def _gold_for(item: dict[str, Any], ability: str) -> str | None:
    candidates = list(_GOLD_FIELDS.get(ability, ("answer",)))
    candidates.extend(("answer", "ideal_answer", "ideal_response", "ideal_summary", "expected_compliance"))
    for field in dict.fromkeys(candidates):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)):
            text = " ".join(str(part) for part in value).strip()
            if text:
                return text
    return None


def validate(record: Any) -> list[str]:
    """Validate one prepared BEAM conversation row; [] means usable."""
    problems: list[str] = []
    if not isinstance(record, dict):
        return ["row was not a dict"]
    if not str(record.get("conversation_id") or "").strip():
        problems.append("missing conversation_id")
    chat = record.get("chat")
    if not isinstance(chat, list) or not chat:
        problems.append("missing chat sessions")
    try:
        parsed = _parse_questions(record.get("probing_questions"))
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return problems + ["probing_questions is not a literal dict"]
    if not isinstance(parsed, dict):
        return problems + ["probing_questions did not parse to a dict"]
    found = 0
    for ability in ABILITIES:
        items = parsed.get(ability)
        if not isinstance(items, list) or not items:
            problems.append(f"{ability}: no probing questions")
            continue
        found += 1
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                problems.append(f"{ability}[{index}]: item was not a dict")
                continue
            if not str(item.get("question") or "").strip():
                problems.append(f"{ability}[{index}]: empty question")
            if not _gold_for(item, ability):
                problems.append(f"{ability}[{index}]: no ideal-answer field")
            rubric = item.get("rubric")
            if not isinstance(rubric, list) or not rubric:
                problems.append(f"{ability}[{index}]: no rubric nuggets")
    if not found:
        problems.append("no recognised BEAM memory ability")
    return problems


def rrf_fuse(ranked_lists: Sequence[Sequence[Any]], *, k: int = 60) -> list[Any]:
    """Reciprocal-rank fusion over independent recall lists (deterministic)."""
    scores: dict[str, float] = {}
    items: dict[str, Any] = {}
    for ranked in ranked_lists:
        for position, item in enumerate(ranked):
            key = json.dumps(item, sort_keys=True, default=str)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + position + 1)
            items.setdefault(key, item)
    return [items[key] for key in sorted(scores, key=lambda value: (-scores[value], value))]


_DATE_FORMATS = ("%B-%d-%Y", "%b-%d-%Y", "%Y-%m-%d", "%B %d, %Y")
_READER_STOPWORDS = frozenset(
    "a an and are as at be by can could did do does for from had has have how i if in "
    "into is it its me my of on or our say so that the their them then there these they "
    "this to was were what when where which who why will with you your about tell more "
    "like only mention items".split()
)


def _time_sort_key(fact: dict[str, Any]) -> tuple[int, float, str]:
    """Sort parsed absolute dates oldest-first; undated facts go last.

    BEAM turns carry a date on the first turn of each session only; an empty
    anchor must not sort ahead of real dates (it used to, because "" < "March").
    """
    fact = fact if isinstance(fact, dict) else {}
    raw = str(fact.get("time_anchor") or fact.get("valid_from") or "").strip()
    if not raw:
        return (1, 0.0, raw)
    for fmt in _DATE_FORMATS:
        try:
            return (0, datetime.strptime(raw, fmt).timestamp(), raw)
        except ValueError:
            continue
    return (0, float("inf"), raw)


def _terms(text: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").casefold())
        if len(token) > 2 and token not in _READER_STOPWORDS
    }


def _select_evidence(question: str, context: str, max_chars: int = READER_CONTEXT_CHARS) -> str:
    """Pick question-relevant conversation-fact lines, then restore their order.

    This is lexical evidence selection over the *already formatted conversation
    facts*. It never sees gold, rubric or annotations, and it never adds text of
    its own: every returned line is a verbatim line from ``context``. When no
    line has any question-term overlap, it returns a neutral prefix instead of
    inventing a hit -- a genuinely unsupported question still gets UNSUPPORTED
    from the unchanged reader abstention rule.
    """
    lines = [line for line in str(context or "").splitlines() if line.strip()]
    if not lines:
        return ""
    question_terms = _terms(question)
    if not question_terms:
        return "\n".join(lines)[:max_chars]
    scored = [(len(question_terms & _terms(line)), position, line) for position, line in enumerate(lines)]
    if not any(score > 0 for score, _position, _line in scored):
        return "\n".join(lines)[:max_chars]
    scored.sort(key=lambda item: (-item[0], item[1]))
    picked: list[tuple[int, str]] = []
    total = 0
    for score, position, line in scored:
        if not picked and score == 0:
            break
        if picked and (score == 0 or total + len(line) + 1 > max_chars):
            break
        picked.append((position, line))
        total += len(line) + 1
    picked.sort(key=lambda item: item[0])
    return "\n".join(line for _position, line in picked)


def format_context(
    facts: Sequence[dict[str, Any]],
    *,
    top_k: int | None = RETRIEVAL_PROFILE["top_k"],
    chronological: bool = True,
) -> str:
    """Render recalled facts as atomic speaker+absolute-date lines, no gold.

    Facts are caller-supplied observations (never probing-question annotations).
    Versioned facts may carry ``valid_from``/``valid_to``; superseded intervals
    are marked so the reader can apply temporal invalidation. Ordering is by
    absolute time anchor when requested, otherwise by the caller's rank.
    ``top_k=None`` renders every supplied fact (the runner passes all fused
    conversation turns); reader_call narrows them to question-relevant evidence.
    """
    selected = list(facts) if not top_k else list(facts)[: int(top_k)]
    if chronological:
        selected.sort(key=_time_sort_key)
    lines: list[str] = []
    for fact in selected:
        if not isinstance(fact, dict):
            continue
        speaker = str(fact.get("speaker") or fact.get("role") or "unknown").strip()
        when = str(fact.get("time_anchor") or fact.get("valid_from") or "undated").strip()
        content = " ".join(str(fact.get("content") or "").split())
        if not content:
            continue
        suffix = ""
        if fact.get("valid_to"):
            suffix = f" [superseded {fact['valid_to']}]"
        lines.append(f"- [{when}] {speaker}: {content}{suffix}")
    return "\n".join(lines)


def reader_call(
    question: str,
    context: str,
    *,
    api_key: str,
    model: str = READER_MODEL,
    base_url: str = READER_BASE_URL,
    timeout: float = 60.0,
    opener: Any | None = None,
) -> tuple[str | None, str | None]:
    """OpenAI-compatible reader call. Returns (answer, error_code), never logs.

    Pass a key obtained from ``KeyPool.get()``; this function stores none. The
    prompt contains only the question and question-relevant conversation-fact
    excerpts -- no ideal answer, rubric, or annotation. The bare-UNSUPPORTED
    abstention rule in the system prompt is unchanged.
    """
    call = opener or urllib.request.urlopen
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer using ONLY the provided conversation facts. Resolve relative "
                    "references (yesterday, last week) using the absolute dates in the "
                    "facts. Read the evidence oldest-first. Be concise and preserve "
                    "absolute dates in the answer. If the answer is not supported by "
                    "the facts, say UNSUPPORTED."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {str(question)[:4000]}\n"
                    f"Facts (question-relevant excerpts from the conversation):\n"
                    f"{_select_evidence(question, context)}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 1024,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    for attempt in (0, 1):
        try:
            with call(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8", "replace"))
            choices = data.get("choices") or []
            message = choices[0].get("message") if choices else None
            answer = message.get("content") if isinstance(message, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                return None, "reader_empty_answer"
            return answer.strip()[:4000], None
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt == 0:
                time.sleep(1.0)
                continue
            return None, f"reader_http_{exc.code}"
        except Exception as exc:
            return None, f"reader_error_{type(exc).__name__}"
    return None, "reader_unavailable"


def _loose_json(text: str) -> Any:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", cleaned, re.DOTALL)
        cleaned = match.group(1).strip() if match else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}|\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# An abstention is not an answer. The reader prompt tells the model to emit
# UNSUPPORTED when the facts do not support a response; tolerate case and
# surrounding punctuation/markdown, but do not infer abstention from prose.
_ABSTENTION_RE = re.compile(
    r"^[^A-Za-z0-9]*unsupported[^A-Za-z0-9]*$", re.IGNORECASE | re.DOTALL
)


def is_abstention(answer: Any) -> bool:
    """True only when a reader answer is the bare UNSUPPORTED sentinel."""
    if not isinstance(answer, str):
        return False
    return bool(_ABSTENTION_RE.match(answer.strip()))


_ABSTENTION_JUDGE_RESULT = {
    "verdict": "incorrect",
    "score": 0.0,
    "reason": "reader_abstained",
    "raw": None,
    "error": None,
    "judge_called": False,
    "abstention_guard": True,
}


def judge_call(
    question: str,
    answer: str,
    gold_answer: str,
    rubric: Sequence[Any],
    *,
    api_key: str,
    model: str = JUDGE_MODEL,
    base_url: str = JUDGE_BASE_URL,
    timeout: float = 60.0,
    opener: Any | None = None,
) -> dict[str, Any]:
    """Frozen judge call. Returns RAW verdict + parse status; no key is logged.

    The judge is given the gold answer and rubric by design; the reader never
    is. Raw verdicts are returned verbatim so a figure can be re-scored without
    switching judges.

    Pre-judge guard: a bare UNSUPPORTED sentinel is an abstention, not an
    answer. It is recorded as INCORRECT / ``reader_abstained`` without any
    network call, so no key is spent and no scorer can call it correct.
    """
    if is_abstention(answer):
        return dict(_ABSTENTION_JUDGE_RESULT)
    call = opener or urllib.request.urlopen
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a BEAM rubric judge. Score the answer against the "
                    "gold answer and each rubric item. Reply with JSON only: "
                    '{"verdict": "correct|partial|incorrect", "score": 0.0-1.0, "reason": "..."}.'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {str(question)[:4000]}\nGold answer: {str(gold_answer)[:4000]}\n"
                    f"Rubric: {json.dumps(list(rubric), ensure_ascii=False)[:6000]}\n"
                    f"Answer to judge: {str(answer)[:4000]}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 1024,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
    )
    for attempt in (0, 1):
        try:
            with call(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8", "replace"))
            choices = data.get("choices") or []
            message = choices[0].get("message") if choices else None
            raw = message.get("content") if isinstance(message, dict) else None
            if not isinstance(raw, str) or not raw.strip():
                return {"verdict": None, "score": None, "raw": raw, "error": "judge_empty_answer"}
            parsed = _loose_json(raw)
            verdict = parsed.get("verdict") if isinstance(parsed, dict) else None
            score = parsed.get("score") if isinstance(parsed, dict) else None
            return {
                "verdict": str(verdict).strip().lower() if verdict is not None else None,
                "score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
                "reason": parsed.get("reason") if isinstance(parsed, dict) else None,
                "raw": raw.strip()[:4000],
                "error": None,
            }
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt == 0:
                time.sleep(1.0)
                continue
            return {"verdict": None, "score": None, "raw": None, "error": f"judge_http_{exc.code}"}
        except Exception as exc:
            return {"verdict": None, "score": None, "raw": None, "error": f"judge_error_{type(exc).__name__}"}
    return {"verdict": None, "score": None, "raw": None, "error": "judge_unavailable"}


def build_rows(
    dataset: Any,
    *,
    reader_fn: Callable[[str, Any], str | None] | None = None,
    max_cases: int | None = None,
    abilities: Sequence[str] | None = None,
    tier: str | None = None,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Build result rows from a prepared dataset without leaking gold.

    ``reader_fn(question, chat)`` receives ONLY the question and conversation
    sessions. Return ``None`` from it to record a missing prediction honestly;
    no answer is fabricated. ``reader_call`` is the intended implementation,
    but any caller-supplied reader must keep the same two-argument shape.
    """
    name, spec = _resolve_tier(tier)
    wanted = tuple(abilities) if abilities else ABILITIES
    unknown = [ability for ability in wanted if ability not in ABILITIES]
    if unknown:
        raise ValueError(f"unknown BEAM abilities: {unknown}")

    records = [item for item in (dataset if isinstance(dataset, list) else []) if isinstance(item, dict)]
    records.sort(key=lambda item: str(item.get("conversation_id") or ""))
    records = limiter(records, max_cases)
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    attempted = 0

    for record in records:
        problems = validate(record)
        if problems:
            failures.append(f"{record.get('conversation_id')}: {'; '.join(problems[:3])}")
            continue
        try:
            parsed = _parse_questions(record.get("probing_questions"))
        except (ValueError, SyntaxError, TypeError):
            failures.append(f"{record.get('conversation_id')}: probing_questions parse failed")
            continue
        conversation_id = str(record["conversation_id"])
        chat = record.get("chat")
        for ability in wanted:
            for index, item in enumerate(parsed.get(ability) or []):
                attempted += 1
                question = str(item.get("question") or "").strip()
                gold = _gold_for(item, ability)
                case_id = f"{conversation_id}::{ability}::{index + 1}"
                prediction: str | None = None
                error: str | None = "no_reader_fn"
                if reader_fn is not None:
                    try:
                        prediction = reader_fn(question, chat)
                        error = None if isinstance(prediction, str) and prediction.strip() else "reader_empty_answer"
                    except Exception as exc:
                        prediction = None
                        error = f"reader_error_{type(exc).__name__}"
                rows.append(
                    {
                        "case_id": case_id,
                        "conversation_id": conversation_id,
                        "tier": name,
                        "ability": ability,
                        "question": question,
                        "pred_answer": prediction,
                        "gold_answer": gold,  # evaluate/judge only; never sent to reader
                        "rubric_count": len(item.get("rubric") or []),
                        "answer_source": ANSWER_SOURCE,
                        "reader_model": READER_MODEL,
                        "judge_model": JUDGE_MODEL,
                        "gold_withheld_from_reader": True,
                        "error": error,
                    }
                )
    return rows, attempted, failures


def _norm(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def evaluate(
    results: list[Any],
    judge_fn: Callable[[Any], Any] | None = None,
    *,
    tier: str | None = None,
) -> dict[str, Any]:
    """Evaluate supplied result rows with a deterministic SubEM/F1 proxy.

    Each row needs ``case_id``, ``ability``, ``gold_answer`` and ``pred_answer``.
    ``judge_fn(record)`` is optional; its RAW output is retained per row and is
    never used to choose or replace the primary metric. Denominators count only
    rows with both a gold label and a prediction.

    A bare UNSUPPORTED answer is an abstention, never a correct answer. Such a
    row is fixed to INCORRECT / ``reader_abstained`` before any judge is called,
    and it earns no SubEM/F1 credit. The guard is recorded per row.
    """
    name, spec = _resolve_tier(tier)
    source_rows = stable_rows(results or [])
    rows: list[dict[str, Any]] = []
    abstention_positions: set[int] = set()
    scored = 0
    subem_hits = 0
    f1_sum = 0.0
    label_missing = 0
    prediction_missing = 0
    per_ability: dict[str, dict[str, Any]] = {
        ability: {"hits": 0, "denominator": 0, "f1_sum": 0.0} for ability in ABILITIES
    }

    for index, record in enumerate(source_rows):
        if not isinstance(record, dict):
            rows.append({"index": index, "valid": False, "note": "row was not a dict; excluded"})
            continue
        ability = str(record.get("ability") or "")
        gold = record.get("gold_answer") if isinstance(record.get("gold_answer"), str) else None
        pred = record.get("pred_answer") if isinstance(record.get("pred_answer"), str) else None
        abstention = is_abstention(pred)
        if abstention:
            abstention_positions.add(index)
        row: dict[str, Any] = {
            "index": index,
            "case_id": record.get("case_id") or record.get("id"),
            "ability": ability,
            "answer_source": record.get("answer_source") or ANSWER_SOURCE,
            "reader_model": record.get("reader_model") or READER_MODEL,
            "judge_model": record.get("judge_model") or JUDGE_MODEL,
            "valid_label": bool(gold and gold.strip()),
            "valid_prediction": bool(pred and pred.strip()),
            "judge_raw": record.get("judge_raw"),
            "abstention_guard": abstention,
        }
        if abstention:
            # Pre-judge guard: no judge output exists (or is trusted) here.
            row["judge_verdict"] = "incorrect"
            row["judge_reason"] = "reader_abstained"
            row["judge_response"] = None
            row["judge_called"] = False
        if not gold or not gold.strip():
            label_missing += 1
            row["note"] = "no gold answer; excluded from proxy denominators"
        elif not pred or not pred.strip():
            prediction_missing += 1
            row["note"] = "no prediction; excluded from proxy denominators"
        else:
            gold_norm = _norm(gold)
            pred_norm = _norm(pred)
            # An abstention earns no answer credit, even if gold is itself the
            # UNSUPPORTED sentinel. Nothing is fabricated: the score is zero.
            subem = 0 if abstention else int(bool(gold_norm) and gold_norm in pred_norm)
            f1 = 0.0 if abstention else token_f1(pred, gold)
            if abstention:
                row["note"] = "reader abstained (UNSUPPORTED); fixed incorrect, no proxy credit"
            scored += 1
            subem_hits += subem
            f1_sum += f1
            row["subem"] = subem
            row["f1"] = f1
            bucket = per_ability.setdefault(ability, {"hits": 0, "denominator": 0, "f1_sum": 0.0})
            bucket["hits"] += subem
            bucket["denominator"] += 1
            bucket["f1_sum"] += f1
        rows.append(row)

    def _guarded_judge(record: Any) -> Any:
        """Pre-judge guard: never invoke the caller's judge on an abstention."""
        pred_value = record.get("pred_answer") if isinstance(record, dict) else None
        if is_abstention(pred_value):
            return {
                "correct": False,
                "is_correct": False,
                "ok": False,
                "verdict": "incorrect",
                "reason": "reader_abstained",
                "abstention_guard": True,
            }
        return judge_fn(record) if judge_fn is not None else None

    judged = apply_judge(source_rows, _guarded_judge if judge_fn is not None else None, key="case_id")
    for judge_row in judged["rows"]:
        position = judge_row["index"]
        if position in abstention_positions:
            # Defensive: even a judge that answered must not override the guard.
            rows[position]["judge_verdict"] = "incorrect"
            rows[position]["judge_reason"] = "reader_abstained"
            rows[position]["judge_response"] = None
            rows[position]["judge_called"] = False
            continue
        if position < len(rows):
            rows[position]["judge_verdict"] = judge_row["verdict"]
            rows[position]["judge_reason"] = judge_row["reason"]
            rows[position]["judge_response"] = judge_row["raw"]

    metric_note = " (derived SubEM proxy; not an official BEAM score)"
    mean_f1 = ratio(round(f1_sum, 6), scored)
    per_ability_out: dict[str, Any] = {}
    for ability in ABILITIES:
        bucket = per_ability.get(ability) or {"hits": 0, "denominator": 0, "f1_sum": 0.0}
        denom = bucket["denominator"]
        mean = ratio(round(bucket["f1_sum"], 6), denom)
        per_ability_out[ability] = {
            "subem_hits": bucket["hits"],
            "denominator": denom,
            "mean_f1": mean,
            "display": ratio_text(f"{ability} subem", bucket["hits"], denom, " (per-ability; diagnostic)"),
        }

    summary_parts = [
        ratio_text("rows supplied", len(source_rows), len(source_rows), ""),
        ratio_text("subem_proxy", subem_hits, scored, metric_note),
        ratio_text("mean_f1", round(f1_sum, 6), scored, metric_note),
        f"label missing: {label_missing}",
        f"prediction missing: {prediction_missing}",
        f"judge raw verdicts retained: {judged['ran']}",
        f"abstention pre-judge guards: {len(abstention_positions)}",
        f"tier: {spec['tier']}",
    ]
    return {
        "lane": LANE,
        "tier": name,
        "split": spec["split"],
        "denominator": denominator(source_rows),
        "correct": subem_hits,
        "primary_metric": "subem_proxy",
        "primary_denominator": scored,
        "rows": rows,
        "metrics": {
            "primary": {
                "name": "subem_proxy",
                "correct": subem_hits,
                "denominator": scored,
                "display": ratio_text("subem_proxy", subem_hits, scored, metric_note),
                "judge_free": True,
            },
            "f1": {"sum": round(f1_sum, 6), "denominator": scored, "mean_f1": mean_f1},
            "per_ability": per_ability_out,
            "label_missing": label_missing,
            "prediction_missing": prediction_missing,
            "abstention_guarded": len(abstention_positions),
            "abstention_reason": "reader_abstained",
            "judge": {
                "ran": judged["ran"],
                "correct": judged["correct"],
                "denominator": judged["denominator"],
                "raw_verdicts_published": True,
                "abstention_guarded": len(abstention_positions),
                "note": judged["note"],
            },
        },
        "answer_source": ANSWER_SOURCE,
        "comparability_tier": "C",
        "official_baselines_not_comparable": True,
        "summary": "; ".join(summary_parts),
        "source_url": SOURCE_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "hf_commit": spec["hf_commit"],
        **stamps(
            _NOTE,
            judge=JUDGE,
            extra={
                "reader": READER,
                "reader_model": READER_MODEL,
                "judge_model": JUDGE_MODEL,
                "retrieval_profile": RETRIEVAL_PROFILE,
                "dataset_files": [
                    {"path": item["path"], "sha256": item["sha256"], "bytes": item["bytes"]}
                    for item in spec["files"]
                ],
            },
        ),
    }


def self_test() -> dict[str, Any]:
    """Offline smoke with synthetic rows. No network, no keys, no spend."""
    rows = [
        {
            "case_id": "c1::information_extraction::1", "ability": "information_extraction",
            "gold_answer": "Ada Lovelace", "pred_answer": "Ada Lovelace",
        },
        {
            "case_id": "c1::temporal_reasoning::1", "ability": "temporal_reasoning",
            "gold_answer": "in March 2024", "pred_answer": "March 2024",
        },
        {
            "case_id": "c1::abstention::1", "ability": "abstention",
            "gold_answer": "UNSUPPORTED", "pred_answer": None,
        },
    ]
    result = evaluate(rows)
    assert result["official"] is False
    assert result["diagnostic"] == "derived"
    assert result["comparability_tier"] == "C"
    assert result["denominator"] == 3
    assert result["primary_denominator"] == 2
    assert result["correct"] == 1
    assert result["reader_model"] == READER_MODEL
    assert result["judge_model"] == JUDGE_MODEL
    assert result["answer_source"] == ANSWER_SOURCE
    assert result["metrics"]["f1"]["mean_f1"] is not None

    fused = rrf_fuse([["a", "b"], ["b", "c"]])
    assert fused[0] == "b"
    context = format_context(
        [
            {"content": "met in Paris", "speaker": "Ada", "time_anchor": "2024-03-01"},
            {"content": "old value", "speaker": "Bob", "time_anchor": "2023-01-01", "valid_to": "2024-01-01"},
        ]
    )
    assert "[2024-03-01] Ada:" in context and "[superseded 2024-01-01]" in context

    # Full-conversation rendering + question-conditioned evidence selection.
    all_facts = [
        {"content": "unrelated cooking notes", "speaker": "Bob", "time_anchor": "2024-06-01"},
        {"content": "studied psychology at Cambridge", "speaker": "Ada", "time_anchor": "2024-03-01"},
        {"content": "another unrelated item", "speaker": "Bob", "time_anchor": "2024-07-01"},
    ]
    full = format_context(all_facts, top_k=None, chronological=True)
    assert len(full.splitlines()) == 3
    assert full.index("studied psychology") < full.index("unrelated cooking")
    evidence = _select_evidence("Where did Ada study psychology?", full, max_chars=500)
    assert "studied psychology" in evidence and "unrelated cooking" not in evidence
    neutral = _select_evidence("quantum physics", full, max_chars=500)
    assert neutral.startswith("- [2024-03-01] Ada:")  # neutral prefix, no invented hit

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "a real answer"}}]}).encode("utf-8")

    captured: dict[str, Any] = {}

    def _reader_opener(request: Any, timeout: float | None = None) -> _Response:
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response()

    answer, reader_error = reader_call(
        "Where did Ada study psychology?", full, api_key="unused", opener=_reader_opener
    )
    assert (answer, reader_error) == ("a real answer", None)
    assert "studied psychology" in captured["payload"]["messages"][1]["content"]
    assert "unrelated cooking" not in captured["payload"]["messages"][1]["content"]

    synthetic_record = {
        "conversation_id": "c1",
        "chat": [[{"role": "user", "content": "hi", "time_anchor": "2024-03-01"}]],
        "probing_questions": {
            ability: [
                {"question": "q", "answer": "a", "rubric": ["r"]},
                {"question": "q2", "answer": "a2", "rubric": ["r"]},
            ]
            for ability in ABILITIES
        },
    }
    assert validate(synthetic_record) == []
    assert validate({"conversation_id": "c1"}) != []
    built, attempted, failures = build_rows([synthetic_record])
    assert attempted == 20 and len(built) == 20 and failures == []
    assert all(row["gold_withheld_from_reader"] is True for row in built)
    assert select(max_cases=7)["effective_denominator"] == 7

    # Abstention guard: no judge call, no lenient scoring, no proxy credit.
    def _no_network(_request: Any, timeout: float | None = None) -> Any:
        raise AssertionError("judge must not be called for an abstention")

    guarded = judge_call("q", "UNSUPPORTED", "gold", ["r"], api_key="unused", opener=_no_network)
    assert guarded["verdict"] == "incorrect"
    assert guarded["reason"] == "reader_abstained"
    assert guarded["judge_called"] is False
    assert is_abstention(" **Unsupported.** ") and not is_abstention("not unsupported")

    def _lenient_judge(_record: Any) -> str:
        return "correct"

    guarded_eval = evaluate(
        [
            {
                "case_id": "c1::abstention::3",
                "ability": "abstention",
                "gold_answer": "UNSUPPORTED",
                "pred_answer": "UNSUPPORTED",
            }
        ],
        judge_fn=_lenient_judge,
    )
    assert guarded_eval["rows"][0]["judge_verdict"] == "incorrect"
    assert guarded_eval["rows"][0]["judge_reason"] == "reader_abstained"
    assert guarded_eval["rows"][0]["subem"] == 0
    assert guarded_eval["metrics"]["judge"]["correct"] == 0
    assert guarded_eval["metrics"]["abstention_guarded"] == 1
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
