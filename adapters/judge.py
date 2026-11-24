"""Cached, key-by-name judge adapter.

The API key is read from ``os.environ`` by NAME only, is never logged, and is
used solely as an HTTP header value in memory. Without a key the result is a
structured ``skipped`` envelope — never a verdict. ``dry_run=True`` produces a
deterministic heuristic explicitly labelled ``heuristic-not-judge``; such a
result must not be aggregated with real judge results.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "memnoth-bench-judge/1"
VERSION = "1"
NOTE = "INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE"
DEFAULT_MODEL = "gpt-4o-2024-08-06"
DEFAULT_KEY_NAME = "OPENAI_API_KEY"
DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0

_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|"
    r"AUTHORIZATION|AUTH|COOKIE)[A-Z0-9_]*)\s*[=:]\s*(\"[^\"]*\"|'[^']*'|\S+)"
)
_MAX_DIAGNOSTIC_CHARS = 400

_SYSTEM_PROMPT = (
    "You grade one answer against a rubric. Reply with a single JSON object only, "
    "with keys verdict (one of correct, partial, incorrect), score (a number from "
    "0 to 1), and rationale (one short sentence). Do not use percentages."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(text: Any) -> str:
    raw = "" if text is None else str(text)
    return _ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", raw)[
        :_MAX_DIAGNOSTIC_CHARS
    ]


def canonical_json(value: Any) -> str:
    """Stable serialization used for every hash in this module."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_for(question: str, rubric: str, model: str) -> str:
    """The exact prompt sent to the judge; also the input to the prompt hash."""
    return (
        f"model={model}\n"
        f"question={question}\n"
        f"rubric={rubric}\n"
        "Return JSON with verdict, score and rationale."
    )


def prompt_hash(question: str, rubric: str, model: str) -> str:
    return sha256_text(canonical_json({"prompt": prompt_for(question, rubric, model)}))


def answer_hash(answer: str) -> str:
    return sha256_text(canonical_json({"answer": answer}))


def cache_key_for(
    question_id: str,
    question: str,
    answer: str,
    rubric: str,
    model: str,
    version: str,
) -> str:
    """sha256 of (question_id, prompt hash, judge version, answer hash)."""
    return sha256_text(
        canonical_json(
            [
                str(question_id),
                prompt_hash(question, rubric, model),
                str(version),
                answer_hash(answer),
            ]
        )
    )


def cache_root(root: Path | None = None) -> Path:
    base = root if root is not None else Path(__file__).resolve().parents[1]
    return Path(base) / "cache" / "judge"


def _base_envelope(*, status: str, source: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "observed_at": _utc_now(),
        "diagnostic": "derived",
        "official": False,
        "note": NOTE,
        "status": status,
        "source": source,
        "verdict": None,
        "score": None,
        "rationale": None,
        "denominators": {"responses_expected": 1, "responses_received": 0},
    }


def heuristic_verdict(question: str, answer: str, rubric: str) -> dict[str, Any]:
    """Deterministic token-overlap heuristic. Not a judge, not a correctness label."""
    tokens = re.compile(r"[a-z0-9]+")
    question_tokens = set(tokens.findall(f"{question} {rubric}".lower()))
    answer_tokens = set(tokens.findall(answer.lower()))
    if not answer_tokens:
        return {
            "verdict": "heuristic-empty",
            "score": 0.0,
            "rationale": "answer contains no word or number tokens",
        }
    if not question_tokens:
        return {
            "verdict": "heuristic-indeterminate",
            "score": None,
            "rationale": "question and rubric contain no word or number tokens",
        }
    overlap = len(answer_tokens & question_tokens) / len(answer_tokens)
    if overlap >= 0.5:
        label = "heuristic-supported"
    elif overlap >= 0.2:
        label = "heuristic-partial"
    else:
        label = "heuristic-unsupported"
    return {
        "verdict": label,
        "score": round(overlap, 4),
        "rationale": "deterministic token overlap against question and rubric; not a judged verdict",
    }


def _dry_run_result(
    *,
    question_id: str,
    question: str,
    answer: str,
    rubric: str,
    model: str,
    version: str,
    api_key_name: str,
    key: str | None,
) -> dict[str, Any]:
    result = _base_envelope(status="dry_run", source="heuristic-not-judge")
    result.update(heuristic_verdict(question, answer, rubric))
    result["judge"] = {
        "model": model,
        "version": version,
        "temperature": 0,
        "key_name": api_key_name,
        "key_present": key is not None,
        "called": False,
    }
    result["cache"] = {"used": False, "hash": None}
    result["question_id"] = str(question_id)
    result["limit"] = (
        "heuristic-not-judge; never aggregate this result with real judge verdicts"
    )
    return result


def _envelope_from_result(envelope: dict[str, Any], status: str, source: str) -> dict[str, Any]:
    base = _base_envelope(status=status, source=source)
    base.update(
        {
            "verdict": envelope.get("verdict"),
            "score": envelope.get("score"),
            "rationale": envelope.get("rationale"),
            "judge": envelope.get("judge"),
            "cache": envelope.get("cache"),
            "question_id": envelope.get("question_id"),
            "judged_at": envelope.get("judged_at"),
            "denominators": envelope.get(
                "denominators", {"responses_expected": 1, "responses_received": 1}
            ),
        }
    )
    return base


def _load_cache(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        return None
    if value.get("status") != "judged" or value.get("source") != "judge":
        return None
    return value


def _save_cache(path: Path, envelope: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="judge-", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, sort_keys=True, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _http_judge(
    *,
    api_base: str,
    model: str,
    version: str,
    prompt: str,
    key: str,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (verdict_fields, diagnostics). The key is never returned or logged."""
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return {}, {
            "status": "error",
            "error": {"kind": "http_error", "message": f"judge HTTP {exc.code}"},
        }
    except (urllib.error.URLError, OSError) as exc:
        return {}, {
            "status": "error",
            "error": {"kind": "network_error", "message": redact(type(exc).__name__)},
        }
    try:
        body = json.loads(raw)
        content = body["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return {}, {
            "status": "error",
            "error": {"kind": "invalid_response", "message": "judge response had no content"},
        }
    parsed = _extract_json_object(str(content))
    if parsed is None:
        return {
            "verdict": "unparsed",
            "score": None,
            "rationale": redact(content)[:_MAX_DIAGNOSTIC_CHARS],
        }, {"status": "judged", "parse": "unparsed"}
    verdict = parsed.get("verdict")
    score = parsed.get("score")
    return {
        "verdict": verdict if isinstance(verdict, str) else "unparsed",
        "score": score if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
        "rationale": redact(parsed.get("rationale")),
    }, {"status": "judged", "parse": "json"}


def judge(
    question_id: str,
    question: str,
    answer: str,
    rubric: str,
    model: str,
    version: str,
    api_key_name: str,
    *,
    dry_run: bool = False,
    api_base: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    cache_root_path: Path | None = None,
) -> dict[str, Any]:
    """Judge one answer, or skip/block without inventing a verdict.

    ``api_key_name`` is an environment variable NAME (default convention:
    ``OPENAI_API_KEY``). Its value is read only for the HTTP header. A missing
    key yields ``status='skipped'`` with null verdict and score.
    """
    name = api_key_name or DEFAULT_KEY_NAME
    key = os.environ.get(name)
    prompt = prompt_for(question, rubric, model)
    digest = cache_key_for(question_id, question, answer, rubric, model, version)

    if dry_run:
        return _dry_run_result(
            question_id=question_id,
            question=question,
            answer=answer,
            rubric=rubric,
            model=model,
            version=version,
            api_key_name=name,
            key=key,
        )

    if not key:
        result = _base_envelope(status="skipped", source="none")
        result["question_id"] = str(question_id)
        result["reason"] = "api_key_not_set"
        result["judge"] = {
            "model": model,
            "version": version,
            "temperature": 0,
            "key_name": name,
            "key_present": False,
            "called": False,
        }
        result["cache"] = {"used": False, "hash": digest}
        result["denominators"] = {"responses_expected": 1, "responses_received": 0}
        return result

    path = cache_root(cache_root_path) / f"{digest}.json"
    cached = _load_cache(path)
    if cached is not None:
        result = _envelope_from_result(cached, "judged", "judge")
        result["cache"] = {
            "used": True,
            "hash": digest,
            "path": str(Path("cache") / "judge" / f"{digest}.json"),
        }
        return result

    verdict_fields, diagnostics = _http_judge(
        api_base=api_base or os.environ.get("OPENAI_BASE_URL") or DEFAULT_API_BASE,
        model=model,
        version=version,
        prompt=prompt,
        key=key,
        timeout=timeout,
    )
    if diagnostics["status"] == "error":
        result = _base_envelope(status="error", source="judge")
        result["question_id"] = str(question_id)
        result["error"] = diagnostics["error"]
        result["judge"] = {
            "model": model,
            "version": version,
            "temperature": 0,
            "key_name": name,
            "key_present": True,
            "called": True,
        }
        result["cache"] = {"used": False, "hash": digest}
        result["denominators"] = {"responses_expected": 1, "responses_received": 0}
        return result

    envelope = _base_envelope(status="judged", source="judge")
    envelope.update(verdict_fields)
    envelope["question_id"] = str(question_id)
    envelope["judged_at"] = envelope["observed_at"]
    envelope["judge"] = {
        "model": model,
        "version": version,
        "temperature": 0,
        "key_name": name,
        "key_present": True,
        "called": True,
        "parse": diagnostics.get("parse"),
    }
    envelope["cache"] = {
        "used": False,
        "hash": digest,
        "path": str(Path("cache") / "judge" / f"{digest}.json"),
    }
    envelope["denominators"] = {"responses_expected": 1, "responses_received": 1}
    _save_cache(path, envelope)
    return envelope


if __name__ == "__main__":
    print(f"{SCHEMA}: self-check")
    sample_key = cache_key_for("q1", "What serial?", "A-1", "exact serial", DEFAULT_MODEL, VERSION)
    print(f"cache_key_sha256={sample_key}")
    print(json.dumps(heuristic_verdict("What serial?", "A-1", "exact serial"), sort_keys=True))
    print(redact("OPENAI_API_KEY=example-not-a-real-key"))
