"""Gold-free LoCoMo seeding and hybrid recall for the derived diagnostic runner.

Scope and non-goals
-------------------
This module is *derived diagnostic plumbing only*. It never reads a dataset
gold answer, a gold evidence ID, a dataset label or a speaker hint in order to
build a query or a fact. Every seeded string is either:

* a real utterance from the pinned conversation, prefixed with that session's
  real absolute date metadata as ``[timestamp] speaker: text``, or
* an absolute date fact carrying the verbatim ``session_N_date_time`` string.

No fabrication: if a session has no real date metadata, its utterances are
seeded without a timestamp and the omission is recorded in the optional
``detail`` out-dict. Hybrid recall uses question decomposition, the existing
keyword/n-gram sweep, neighbour expansion (+/-1 before, +2 after) and a
gold-free sufficiency stopping rule. No score is asserted here; callers remain
responsible for the seed -> recall -> asof -> reader chain and for reporting
counts with denominators.
"""

from __future__ import annotations

import concurrent.futures
import re
from typing import Any, Callable

# Keep the same hard ceiling as the runner so this module cannot open more CLI
# fan-out than the lane already allows.
_MAX_WORKERS = 8

# Real LoCoMo conversations store dates as ``session_<n>_date_time`` next to the
# ``session_<n>`` turn lists. A generic suffix match also tolerates neighbouring
# datasets that use ``_date`` / ``_datetime`` for the same real metadata field.
_DATE_KEY_RE = re.compile(
    r"^(?P<sid>session_[A-Za-z0-9_-]+?)_(?:date_time|date|datetime)$",
    re.IGNORECASE,
)
_DATE_KEY_PATTERN = r"^(session_[A-Za-z0-9_-]+?)_(date_time|date|datetime)$"

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_ENTITY_RE = re.compile(r"^(?P<sid>.+?)-u(?P<idx>\d+)-(?P<slug>.+)$")

_STOPWORDS = {
    "the", "and", "for", "was", "were", "what", "when", "where", "who", "with",
    "that", "this", "from", "have", "had", "has", "did", "does", "how", "why",
    "about", "into", "its", "our", "your", "their", "they", "then", "than",
    "after", "before", "first", "last", "been", "are", "not", "you", "his", "her",
    "would", "could", "should", "likely", "pursue",
}

# Split only on discourse boundaries, so fragments stay real substrings of the
# question rather than invented paraphrases.
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:;|\bbut\b|\band\b|,)\s*", re.IGNORECASE)


def _slug(text: str, max_words: int = 6) -> str:
    words = _WORD_RE.findall(str(text or ""))[:max_words]
    return "-".join(word.lower() for word in words)[:60] or "utterance"


def _keyword_terms(question: str) -> list[str]:
    """Content words from the question only; no dataset field is consulted."""
    return [
        token
        for token in (match.group(0).casefold() for match in _TOKEN_RE.finditer(str(question or "")))
        if token not in _STOPWORDS
    ]


def _question_clauses(question: str) -> list[str]:
    """Genuine substrings of the question, split on discourse boundaries."""
    clauses: list[str] = []
    for raw in _CLAUSE_SPLIT_RE.split(str(question or "")):
        clause = str(raw).strip()
        terms = _keyword_terms(clause)
        if len(terms) >= 2 and len(clause) <= 200 and clause not in clauses:
            clauses.append(clause)
    return clauses


def _discover_session_dates(
    sessions: dict[str, Any], session_dates: dict[str, Any] | None
) -> dict[str, str]:
    """Collect only real, non-empty string date metadata; never synthesize one."""
    found: dict[str, str] = {}

    # Explicit mapping supplied by the parent from the pinned conversation's
    # own date fields. This is the preferred integration path.
    for session_id, value in (session_dates or {}).items():
        if isinstance(value, str) and value.strip():
            found[str(session_id)] = value.strip()

    # Also accept the whole conversation dict as ``sessions``: the real dataset
    # mixes ``session_1`` turn lists with ``session_1_date_time`` strings.
    for key, value in sessions.items():
        match = _DATE_KEY_RE.match(str(key))
        if match and isinstance(value, str) and value.strip():
            found[match.group("sid")] = value.strip()
        elif str(key).casefold() == "session_dates" and isinstance(value, dict):
            for session_id, raw in value.items():
                if isinstance(raw, str) and raw.strip():
                    found[str(session_id)] = raw.strip()
    return found


def _sufficient(found: list[str]) -> bool:
    """Gold-free stopping heuristic for the recall sweep, not a score.

    Breadth alone (six distinct observed entities) or a smaller set that also
    contains at least one real absolute-date entity is treated as enough
    evidence to stop spending recall calls. No answer or evidence ID is used.
    """
    if len(found) >= 6:
        return True
    return len(found) >= 3 and any(name.endswith("-date") for name in found)


def _neighbour_queries(found: list[str]) -> list[str]:
    """Expand observed turn entities to one before and two after each hit."""
    queries: list[str] = []
    seen: set[str] = set()
    for name in found[:6]:
        match = _ENTITY_RE.match(name)
        if not match:
            continue
        session_id = match.group("sid")
        index = int(match.group("idx"))
        for neighbour in (index - 1, index + 1, index + 2):
            if neighbour < 1:
                continue
            for query in (f"{session_id}-u{neighbour}", f"{session_id} u{neighbour}"):
                if query not in seen:
                    seen.add(query)
                    queries.append(query)
    return queries


def seed_conversation(
    remember: Callable[[str, str], bool],
    sessions: dict[str, list],
    workers: int,
    *,
    session_dates: dict[str, Any] | None = None,
    detail: dict[str, Any] | None = None,
) -> bool:
    """Seed timestamped atomic turn facts and real session-date facts.

    ``remember(entity, text) -> bool`` is supplied by the caller; this module
    never speaks to the CLI or the network itself. Each utterance becomes its
    own entity ``{session_id}-u{n}-{slug}`` (1-based ``n``). Its text is
    ``[absolute date] speaker: text`` when the session has real date metadata,
    otherwise ``speaker: text`` with no invented timestamp. One extra fact per
    dated session is seeded under ``{session_id}-date`` carrying the verbatim
    metadata date.

    Returns True only when every attempted write succeeded. Sessions without
    real dates are skipped for date seeding and listed in
    ``detail['sessions_without_date']``.
    """
    report: dict[str, Any] = {
        "utterances_seeded": 0,
        "utterances_failed": 0,
        "date_facts_seeded": 0,
        "date_facts_failed": 0,
        "session_count": 0,
        "sessions_with_date": [],
        "sessions_without_date": [],
        "date_entities": [],
        "date_fact_texts": [],
        "date_entity_by_session": {},
        "timestamps_added": 0,
        "timestamps_omitted": 0,
    }
    if isinstance(detail, dict):
        detail.clear()
        detail.update(report)
    target = detail if isinstance(detail, dict) else report

    sessions = sessions if isinstance(sessions, dict) else {}
    discovered = _discover_session_dates(sessions, session_dates)
    session_ids = [
        str(key)
        for key, value in sessions.items()
        if isinstance(value, list) and not _DATE_KEY_RE.match(str(key))
    ]
    target["session_count"] = len(session_ids)
    target["sessions_with_date"] = [sid for sid in session_ids if sid in discovered]
    target["sessions_without_date"] = [sid for sid in session_ids if sid not in discovered]

    # (kind, entity, text) -- kind is only used for accounting, never seeded.
    tasks: list[tuple[str, str, str]] = []
    for session_id in session_ids:
        raw_date = discovered.get(session_id)
        turns = sessions.get(session_id) or []
        for index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                continue
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            speaker = str(turn.get("speaker") or "").strip()
            body = f"{speaker}: {text}" if speaker else text
            fact = f"[{raw_date}] {body}" if raw_date else body
            if raw_date:
                target["timestamps_added"] = int(target["timestamps_added"]) + 1
            else:
                target["timestamps_omitted"] = int(target["timestamps_omitted"]) + 1
            entity = f"{session_id}-u{index}-{_slug(text)}"
            tasks.append(("utterance", entity, fact))

    for session_id, raw_date in discovered.items():
        entity = f"{session_id}-date"
        text = f"[{raw_date}] {session_id} absolute date: {raw_date}"
        tasks.append(("date", entity, text))
        target["date_entity_by_session"][session_id] = entity

    def attempt(entity_and_text: tuple[str, str]) -> bool:
        entity, text = entity_and_text
        try:
            return bool(remember(entity, text))
        except Exception:
            return False

    pairs = [(entity, text) for _, entity, text in tasks]
    count = max(1, min(int(workers or 1), _MAX_WORKERS))
    if count == 1:
        results = [attempt(pair) for pair in pairs]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            results = list(pool.map(attempt, pairs))

    seeded_key = {"utterance": "utterances_seeded", "date": "date_facts_seeded"}
    failed_key = {"utterance": "utterances_failed", "date": "date_facts_failed"}
    for (kind, entity, text), ok in zip(tasks, results):
        key = seeded_key[kind] if ok else failed_key[kind]
        target[key] = int(target[key]) + 1
        if ok and kind == "date":
            target["date_entities"].append(entity)
            target["date_fact_texts"].append(text)
    return all(results)


def recall_question(
    recall: Callable[[str, int], list[str]], question: str, limit: int
) -> list[str]:
    """Hybrid, gold-free recall over the question only.

    Forms, in stable order: the full question, real clause substrings from a
    discourse split, the content-word sweep, short 3/2/1-gram prefixes and each
    distinct content word. Each form is a real ``recall(query, limit)`` call.
    After the sweep, observed turn entities are expanded to one neighbour before
    and two after. A gold-free sufficiency rule may stop early. Results are
    de-duplicated by entity name in first-seen order. No gold answer, gold
    evidence ID or dataset label is ever read or injected.
    """
    text = str(question or "").strip()
    forms: list[str] = []
    if text:
        forms.append(text[:2000])
    forms.extend(_question_clauses(text))
    terms = _keyword_terms(text)
    if terms:
        forms.append(" ".join(terms)[:200])
    for size in (3, 2, 1):
        if len(terms) >= size:
            forms.append(" ".join(terms[:size]))
    forms.extend(dict.fromkeys(terms))

    seen_forms: set[str] = set()
    ordered: list[str] = []
    for form in forms:
        if form and form not in seen_forms:
            seen_forms.add(form)
            ordered.append(form)

    found: list[str] = []
    seen: set[str] = set()

    def absorb(hits: Any) -> None:
        for hit in hits or []:
            name = str(hit).strip()
            if name and name not in seen:
                seen.add(name)
                found.append(name)

    # Do NOT stop the form sweep on sufficiency: the first 6 hits are often all
    # from one session (e.g. generic "melanie"/"caroline" turns) while the
    # answer-bearing turn needs a decisive single term ("sunrise", "painted").
    # Run every gold-free form, then neighbour expansion, so those turns surface.
    for form in ordered:
        try:
            absorb(recall(form, limit))
        except Exception:
            continue

    for query in _neighbour_queries(found):
        try:
            absorb(recall(query, limit))
        except Exception:
            continue
    return found


def inspect_dataset_shape(dataset: Any) -> dict[str, Any]:
    """Report which real keys hold session dates, for parent integration.

    Accepts the whole dataset list, one record, or one conversation dict. Never
    inspects question/answer/evidence/label fields: it only scans key names and
    string values for the date metadata pattern observed in the pinned file.
    """
    record: Any = dataset
    if isinstance(record, list):
        record = next((item for item in record if isinstance(item, dict)), None)
    if not isinstance(record, dict):
        return {
            "found": False,
            "reason": "unsupported dataset container",
            "date_key_pattern": _DATE_KEY_PATTERN,
            "date_keys": [],
            "session_keys": [],
            "sessions_with_date": [],
            "sessions_without_date": [],
            "date_count": 0,
            "sample": None,
        }

    source = "record"
    conversation = record.get("conversation")
    scope: dict[str, Any] = conversation if isinstance(conversation, dict) else record
    if conversation is not None:
        source = "conversation"

    date_entries: dict[str, dict[str, str]] = {}
    date_keys: list[str] = []
    for key, value in scope.items():
        match = _DATE_KEY_RE.match(str(key))
        if match and isinstance(value, str) and value.strip():
            date_keys.append(str(key))
            date_entries[match.group("sid")] = {"key": str(key), "value": value.strip()}

    session_keys = [
        str(key)
        for key, value in scope.items()
        if isinstance(value, list) and not _DATE_KEY_RE.match(str(key))
    ]
    sample = None
    if date_keys:
        first = date_entries[next(iter(date_entries))]
        sample = {"key": first["key"], "value": first["value"]}

    return {
        "found": bool(date_keys),
        "source": source,
        "date_key_pattern": _DATE_KEY_PATTERN,
        "date_keys": date_keys,
        "session_keys": session_keys,
        "sessions_with_date": [sid for sid in session_keys if sid in date_entries],
        "sessions_without_date": [sid for sid in session_keys if sid not in date_entries],
        "date_count": len(date_entries),
        "date_entries": date_entries,
        "sample": sample,
    }


if __name__ == "__main__":
    import json

    # Offline smoke only: synthetic turns, synthetic metadata, in-memory
    # callback. No network, no keys, no dataset access, no score.
    synthetic_sessions: dict[str, list] = {
        "session_1": [
            {"speaker": "Melanie", "text": "I joined the LGBTQ support group yesterday."},
            {"speaker": "Caroline", "text": "I painted a sunrise last year."},
        ],
        "session_2": [{"speaker": "Melanie", "text": "We met on Tuesday."}],
        "session_1_date_time": "1:56 pm on 8 May, 2023",  # real-field shape, synthetic value
    }
    writes: list[tuple[str, str]] = []

    def fake_remember(entity: str, text: str) -> bool:
        writes.append((entity, text))
        return True

    detail: dict[str, Any] = {}
    ok = seed_conversation(
        fake_remember,
        synthetic_sessions,
        workers=2,
        detail=detail,
    )
    entities = [entity for entity, _ in writes]
    recall_queries: list[str] = []

    def fake_recall(query: str, limit: int) -> list[str]:
        recall_queries.append(query)
        if "support" in query.casefold():
            return [
                "session_1-u1-i-joined-the-lgbtq-support-group",
                "session_1-date",
            ]
        if "-u2" in query or " u2" in query:
            return ["session_1-u2-i-painted-a-sunrise-last-year"]
        return []

    ranked = recall_question(fake_recall, "When did Melanie join the support group?", 5)
    shape = inspect_dataset_shape({"conversation": synthetic_sessions})

    print(
        json.dumps(
            {
                "note": "INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE",
                "diagnostic": "derived",
                "all_writes_ok": ok,
                "write_count": len(writes),
                "seeded_facts": [text for _, text in writes],
                "timestamps_added": detail["timestamps_added"],
                "timestamps_omitted": detail["timestamps_omitted"],
                "sessions_without_date": detail["sessions_without_date"],
                "recall_forms": recall_queries,
                "recall_ranked": ranked,
                "date_key_found": shape["found"],
                "date_keys": shape["date_keys"],
            },
            indent=2,
            sort_keys=True,
        )
    )
