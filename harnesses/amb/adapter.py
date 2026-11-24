"""Pinned AMB / PrecisionMemBench belief-ID retrieval adapter.

This is a **judge-free derived diagnostic**, not an official AMB or
PrecisionMemBench result. The harness runs AMB's ``precisionmembench`` dataset in
``--mode retrieval``: no answer LLM, no judge, no reranker. The only measurement
is whether the correct belief ID comes back from the memory system in the top-k.

Why an adapter is needed at all
-------------------------------
AMB's ``MemoryProvider`` contract is Python ``ingest(documents)`` /
``retrieve(query, k, user_id)`` with document IDs that round-trip to gold belief
IDs. Our public CLI/MCP has no such Add/Search contract: ``mem_ingest`` takes a
directory, ``mem_add_fact`` has no ``beliefId`` field, and ``mem_search`` has no
per-call ``user_id``. This adapter supplies the missing pieces deterministically:

* a **belief-ID <-> vault-entity map** (identity, so it is trivially auditable),
* **per-user vault isolation** (one temp vault per case, seeded only with the
  beliefs belonging to that case's ``user_id``), and
* a **fresh-vault reset** (a new temp vault file per case; the bi-temporal vault
  invalidates rather than deletes, so a fresh file is the honest reset).

Gold belief IDs are used **only** to score. The recall query is the case's
``query`` text verbatim; no gold ID, ID map or expectation is ever sent to it.

Pins
----
Every hash below was computed locally from bytes downloaded at the exact public
URL/revision; resolve/<commit>/ URLs are used, never a mutable branch. The
retired MemBench dataset is explicitly UNVERIFIED and is refused. No key value,
no private path, and no score is present here.

Stamps for every number produced from this adapter: ``official=False``,
``diagnostic="derived"``, ``comparability_tier="C"``, ``score_asserted=False``,
and the standing "NOT an official score" note.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import re
import shutil
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

LANE = "amb-pmb"
SOURCE_URL = "https://github.com/vectorize-io/agent-memory-benchmark"
SOURCE_PAPER_URL = "https://arxiv.org/abs/2605.11325"
PMB_SOURCE_URL = "https://github.com/tenurehq/precisionmembench"

# Immutable pins. AMB HEAD and the upstream PrecisionMemBench commit are both
# recorded; the vendored fixtures are byte-verified at AMB HEAD.
AMB_HEAD = "01bd83f3fa5245d6be3986cfab81fdd828f26acd"
UPSTREAM_COMMIT = "b95d6abb471c0d591c440172283dd74e7af000df"

_AMB_RAW = (
    "https://raw.githubusercontent.com/vectorize-io/agent-memory-benchmark/"
    + AMB_HEAD
    + "/data/precisionmembench/"
)
_PMB_RAW = (
    "https://raw.githubusercontent.com/tenurehq/precisionmembench/"
    + UPSTREAM_COMMIT
    + "/fixtures/"
)

# The five vendored PMB .json.gz files: relative path -> (sha256, exact bytes).
VENDORED_GZ: dict[str, tuple[str, int]] = {
    "info.json.gz": (
        "e2b02701e3accece8a98ce2eb1842f19dd623ea28628752c48213c02dcdb87ad",
        182,
    ),
    "single-turn/categories.json.gz": (
        "d5ff4a1df28138d90bb430d4713b36ec56d6913d063093ac67909acae9f83c9a",
        1351,
    ),
    "single-turn/documents.json.gz": (
        "a7bb7ed587f1fee494559211313a75a23133170863070c1ef9bb13e95df11b46",
        3604,
    ),
    "single-turn/queries.json.gz": (
        "fdbd66f0bb907215cc21282f7879bdae88fb499b874ede615d3c0acb3c9fbfa3",
        7261,
    ),
    "single-turn/stats.json.gz": (
        "c5637f8204805421a7a0f495b5ff0c0520ea287624ee44a50a584742bb96b575",
        388,
    ),
}

# Upstream fixtures. The vendored `documents.json.gz` projects each belief down
# to {id, content, user_id, timestamp}; the full fixture supplies the metadata
# fields (pinned/scope/type/resolved_at/superseded_by/participants) that the
# local, provider-independent scoring tiers need. Both files are verified pins.
UPSTREAM_FIXTURES: dict[str, tuple[str, int]] = {
    "beliefs.seed.json": (
        "35659b3b9a581b644fc6746e1b4e902c3e7b56c0973b546bf7b0816a69e3de98",
        33303,
    ),
    "retrieval.cases.json": (
        "09e99f0852280538abe34b03570f65583fdb10df3c554b7150f0f665d5db709c",
        35940,
    ),
}

# Retired MemBench (import-myself/Membench) is NOT this dataset. Its four test
# files ship through mutable Drive/Baidu links with no published digest and its
# AMB adapter was removed on 2026-08-03. No hash is invented and the data is
# refused by refuse_membench_data().
RETIRED_MEMBENCH: dict[str, Any] = {
    "status": "UNVERIFIED — NOT COMPUTED",
    "reason": (
        "No published SHA-256, no immutable revision; mutable Drive/Baidu links "
        "only. The AMB adapter was deleted 2026-08-03. This harness does not "
        "and will not load MemBench data."
    ),
    "source": "https://github.com/import-myself/Membench",
}

# Verified dataset shape and the AMB self-check's pass-type split.
N_CASES = 77
N_BELIEFS = 35
N_ACTIVE = 43
N_STRUCTURAL = 25
N_EMPTY = 9
PASS_TYPES: tuple[str, ...] = ("active", "structural", "trivially-empty")

DEFAULT_USER_ID = "test-user"
DEFAULT_MAX_BELIEFS = 20
DEFAULT_MAX_QUESTIONS = 15
DEFAULT_TOP_K = 10

# Verbatim from the pinned AMB adapter: the fixed persona prelude, which is a
# local harness tier, never a provider result and never an answer model.
EVAL_PERSONA_UNIVERSAL = (
    "You prefer direct answers without preamble. You push back when plans have "
    "problems rather than defaulting to agreement. You edit AI output; you do not "
    "let AI edit your prose."
)

# Approved external data root. Dataset bytes never enter the repository.
BENCH_DATA_ROOT = Path(r"D:\memnoth-bench-data")
DEFAULT_DATA_DIR = BENCH_DATA_ROOT / "datasets" / "amb"
CAP_BYTES = 80_000_000_000  # hard 80 GB cap for the whole bench-data root

_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_ENTITY_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")

_NOTE = (
    "Derived diagnostic on the pinned 77-case PrecisionMemBench single-turn "
    "retrieval lane (AMB --mode retrieval): belief-ID top-k accuracy with no "
    "answer model and no judge. official=false, diagnostic=derived, "
    "comparability_tier=C, score_asserted=false. NOT an official score. No "
    "ranking, win or leaderboard comparison is claimed or implied."
)

__all__ = [
    "LANE", "SOURCE_URL", "SOURCE_PAPER_URL", "PMB_SOURCE_URL", "AMB_HEAD",
    "UPSTREAM_COMMIT", "VENDORED_GZ", "UPSTREAM_FIXTURES", "RETIRED_MEMBENCH",
    "N_CASES", "N_BELIEFS", "N_ACTIVE", "N_STRUCTURAL", "N_EMPTY", "PASS_TYPES",
    "DEFAULT_USER_ID", "DEFAULT_MAX_BELIEFS", "DEFAULT_MAX_QUESTIONS",
    "DEFAULT_TOP_K", "EVAL_PERSONA_UNIVERSAL", "BENCH_DATA_ROOT",
    "DEFAULT_DATA_DIR", "CAP_BYTES", "sha256_text", "sha256_file",
    "gold_hash_prefix", "refuse_membench_data", "prepare", "load_payload",
    "compute_pass_types", "belief_entity", "build_id_maps", "validate",
    "validate_result", "expected_relevant_ids", "pass_type", "score_case",
    "build_rows", "public_row", "evaluate", "select", "self_test",
]


# --------------------------------------------------------------------------- #
# Hashing / pin hygiene
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
    """Only a hash prefix of gold belief IDs is ever published, never the IDs."""
    if gold is None:
        return None
    if isinstance(gold, (list, tuple, set)):
        text = "\n".join(sorted(str(item) for item in gold))
    else:
        text = str(gold)
    if not text.strip():
        return None
    return sha256_text(text)[:length]


def refuse_membench_data(local_dir: str | Path | None = None) -> None:
    """Raise if retired, unpinned MemBench data is present or requested.

    The only accepted dataset here is the pinned PrecisionMemBench snapshot.
    This guard makes the UNVERIFIED status an active refusal, not a comment.
    """
    if local_dir is not None:
        root = Path(local_dir).expanduser()
        for name in ("MemData", "membench", "membench-data"):
            if (root / name).exists():
                raise RuntimeError(
                    f"refusing retired MemBench data at {root / name}: "
                    + RETIRED_MEMBENCH["reason"]
                )
    raise RuntimeError(
        "MemBench is retired and UNVERIFIED — NOT COMPUTED; "
        + RETIRED_MEMBENCH["reason"]
    )


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


def _fetch_one(url: str, sha256: str, expected_bytes: int, dest: Path) -> str:
    """Stream, verify, then atomically place one pinned artifact.

    Returns ``dataset_cache_hit`` or ``dataset_verified``. This is the only
    network boundary; it is never called at import time or by ``--dry-run``.
    """
    if not _HEX64.fullmatch(str(sha256).strip()):
        raise ValueError("dataset SHA-256 pin must be exactly 64 hex characters")
    if urlparse(str(url)).scheme.lower() not in {"http", "https"}:
        raise ValueError("dataset URL must be http(s)")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int) or expected_bytes <= 0:
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
            return "dataset_cache_hit"
        target.unlink()  # stale/corrupt cache: refetch under the same pin

    temp = target.with_name(f"{target.name}.part-{os.getpid()}")
    try:
        total = 0
        with urllib.request.urlopen(str(url), timeout=120.0) as response, temp.open("wb") as out:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                total += len(block)
        observed = sha256_file(temp)
        if not hmac.compare_digest(observed, pin):
            raise RuntimeError(
                f"SHA-256 mismatch for {target.name}: expected {pin}, got {observed}"
            )
        if total != expected_bytes:
            raise RuntimeError(
                f"byte-count mismatch for {target.name}: "
                f"expected {expected_bytes}, got {total}"
            )
        os.replace(temp, target)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    return "dataset_verified"


def prepare(local_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    """Download and SHA-verify all pinned PMB artifacts into ``local_dir``.

    The destination must stay under the approved data root, outside the repo.
    Seven files land there: five vendored .json.gz snapshots at AMB HEAD plus
    the two upstream fixture JSONs. No network at import time.
    """
    root = Path(local_dir).expanduser().resolve()
    try:
        root.relative_to(BENCH_DATA_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"refusing to write outside approved data root {BENCH_DATA_ROOT}: {root}"
        ) from exc
    root.mkdir(parents=True, exist_ok=True)
    notes: dict[str, str] = {}
    for relative, (sha, count) in VENDORED_GZ.items():
        notes[relative] = _fetch_one(_AMB_RAW + relative, sha, count, root / relative)
    for filename, (sha, count) in UPSTREAM_FIXTURES.items():
        notes[filename] = _fetch_one(_PMB_RAW + filename, sha, count, root / filename)
    return root


# --------------------------------------------------------------------------- #
# Offline loading and shape validation
# --------------------------------------------------------------------------- #


def _read_gz(path: Path, sha256: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"pinned PMB artifact not prepared: {path}")
    if not hmac.compare_digest(sha256_file(path), sha256):
        raise RuntimeError(f"SHA-256 mismatch for {path.name}; refusing to use bytes")
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _read_json(path: Path, sha256: str) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"pinned PMB artifact not prepared: {path}")
    if not hmac.compare_digest(sha256_file(path), sha256):
        raise RuntimeError(f"SHA-256 mismatch for {path.name}; refusing to use bytes")
    return json.loads(path.read_text(encoding="utf-8"))


def load_payload(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(beliefs, cases, documents)`` from locally verified bytes.

    ``beliefs`` are the full upstream fixtures with a ``document_text`` field
    copied from the vendored document projection; ``cases`` come from the
    vendored ``queries.json.gz``. Only field shapes are asserted here. Gold IDs
    are present in the returned cases but are never used to build a recall query.
    """
    root = Path(data_dir).expanduser()
    info = _read_gz(root / "info.json.gz", VENDORED_GZ["info.json.gz"][0])
    categories = _read_gz(
        root / "single-turn/categories.json.gz", VENDORED_GZ["single-turn/categories.json.gz"][0]
    )
    documents = _read_gz(
        root / "single-turn/documents.json.gz", VENDORED_GZ["single-turn/documents.json.gz"][0]
    )
    cases = _read_gz(
        root / "single-turn/queries.json.gz", VENDORED_GZ["single-turn/queries.json.gz"][0]
    )
    stats = _read_gz(
        root / "single-turn/stats.json.gz", VENDORED_GZ["single-turn/stats.json.gz"][0]
    )
    beliefs = _read_json(root / "beliefs.seed.json", UPSTREAM_FIXTURES["beliefs.seed.json"][0])
    # retrieval.cases.json is a cross-check of the vendored query ID set, not a
    # second scoring source. Both are pinned and verified.
    upstream_cases = _read_json(
        root / "retrieval.cases.json", UPSTREAM_FIXTURES["retrieval.cases.json"][0]
    )

    if not isinstance(documents, list) or not isinstance(cases, list):
        raise ValueError("documents/queries payload must be a list")
    if not isinstance(beliefs, list) or not isinstance(upstream_cases, list):
        raise ValueError("belief fixture payload must be a list")
    if not isinstance(info, dict) or not isinstance(categories, dict) or not isinstance(stats, dict):
        raise ValueError("info/categories/stats payload must be an object")
    if len(documents) != N_BELIEFS or len(beliefs) != N_BELIEFS or len(cases) != N_CASES:
        raise ValueError(
            f"unexpected PMB shape: documents={len(documents)} beliefs={len(beliefs)} "
            f"cases={len(cases)} (expected {N_BELIEFS}/{N_BELIEFS}/{N_CASES})"
        )
    if stats.get("queries") != N_CASES or stats.get("docs") != N_BELIEFS:
        raise ValueError(f"stats.json.gz disagrees with the pinned shape: {stats}")

    vendored_ids = {str(case.get("id")) for case in cases}
    upstream_ids = {str(case.get("caseId")) for case in upstream_cases}
    if vendored_ids != upstream_ids:
        raise ValueError(
            "vendored query IDs differ from the pinned upstream case IDs; "
            f"vendored_only={sorted(vendored_ids - upstream_ids)[:4]} "
            f"upstream_only={sorted(upstream_ids - vendored_ids)[:4]}"
        )

    text_by_id = {str(doc.get("id")): str(doc.get("content") or "") for doc in documents}
    enriched: list[dict[str, Any]] = []
    for belief in beliefs:
        if not isinstance(belief, dict):
            raise ValueError("belief fixture row is not an object")
        bid = str(belief.get("_id") or "")
        if bid not in text_by_id:
            raise ValueError(f"belief {bid!r} has no vendored document text")
        row = dict(belief)
        row["document_text"] = text_by_id[bid]
        enriched.append(row)

    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("meta"), dict):
            raise ValueError("query row is missing its meta object")
        if "query" not in case:
            raise ValueError(f"query row {case.get('id')!r} has no query field")

    split = compute_pass_types(enriched, cases)
    if (split["active"], split["structural"], split["trivially-empty"]) != (
        N_ACTIVE,
        N_STRUCTURAL,
        N_EMPTY,
    ):
        raise ValueError(
            "PMB pass-type split does not match the pinned self-check: "
            f"active={split['active']} structural={split['structural']} "
            f"empty={split['trivially-empty']} (expected "
            f"{N_ACTIVE}/{N_STRUCTURAL}/{N_EMPTY})"
        )
    return enriched, cases, documents


def compute_pass_types(
    beliefs: Iterable[dict[str, Any]], cases: Iterable[dict[str, Any]]
) -> dict[str, int]:
    """Compute AMB's three-way case split without any retrieval or model call."""
    rows = list(beliefs)
    pinned_in_seed = {
        str(b.get("_id"))
        for b in rows
        if b.get("pinned") is True and b.get("user_id") == DEFAULT_USER_ID
    }
    counts: Counter[str] = Counter()
    for case in cases or []:
        expect = ((case.get("meta") or {}).get("expect") or {}) if isinstance(case, dict) else {}
        expected = expected_relevant_ids(expect, pinned_in_seed)
        counts[pass_type(expect, expected)] += 1
    return {key: int(counts.get(key, 0)) for key in PASS_TYPES}


# --------------------------------------------------------------------------- #
# Deterministic belief-ID <-> vault-entity mapping
# --------------------------------------------------------------------------- #


def belief_entity(belief_id: Any) -> str:
    """Deterministic vault entity name for a belief ID.

    The map is deliberately identity: PMB belief IDs are already lowercase,
    unique, whitespace-free and FTS-friendly (``b-db-decision``). An identity
    map is the most auditable form and avoids adding noise tokens that would
    damage short-alias queries (k8s, ts, gha, kube).
    """
    return _ENTITY_UNSAFE.sub("-", str(belief_id)).strip("-") or "belief"


def build_id_maps(
    beliefs: Iterable[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(belief_to_entity, entity_to_belief)``; refuse a collision."""
    forward: dict[str, str] = {}
    inverse: dict[str, str] = {}
    for belief in beliefs or []:
        bid = str(belief.get("_id") or "")
        if not bid:
            raise ValueError("belief has no _id; cannot build a deterministic ID map")
        entity = belief_entity(bid)
        if entity in inverse and inverse[entity] != bid:
            raise ValueError(f"belief-ID entity collision at {entity!r}")
        forward[bid] = entity
        inverse[entity] = bid
    if len(forward) != len(inverse):
        raise ValueError("belief-ID map is not bijective")
    return forward, inverse


# --------------------------------------------------------------------------- #
# Case validation and AMB scoring (ported from the pinned adapter)
# --------------------------------------------------------------------------- #


def validate(record: dict[str, Any]) -> list[str]:
    """Return validation errors for one prepared case; any error excludes it."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["case row is not a dict"]
    if not str(record.get("id") or "").strip():
        errors.append("case id is missing")
    if str(record.get("query") or "") == "" and "query" not in record:
        errors.append("case query field is missing")
    meta = record.get("meta")
    if not isinstance(meta, dict):
        errors.append("case meta is not an object")
        return errors
    expect = meta.get("expect")
    if not isinstance(expect, dict):
        errors.append("case meta.expect is not an object")
    scope = meta.get("scope")
    if scope is not None and not isinstance(scope, list):
        errors.append("case meta.scope is not a list")
    return errors


def validate_result(record: dict[str, Any]) -> list[str]:
    """Validate one *result* row (a built case), distinct from a prepared case."""
    if not isinstance(record, dict):
        return ["result row is not a dict"]
    errors: list[str] = []
    if not str(record.get("case_id") or "").strip():
        errors.append("result case_id is missing")
    if record.get("pass_type") not in PASS_TYPES:
        errors.append("result pass_type is not a known type")
    if not isinstance(record.get("hit"), (bool, int)):
        errors.append("result hit is missing")
    return errors


def expected_relevant_ids(expect: dict[str, Any], pinned_in_seed: set[str]) -> list[str]:
    """AMB's expected relevant tier: shouldOnlyInclude, else non-pinned mustInclude."""
    rb = expect.get("relevantBeliefs") or {}
    only = rb.get("shouldOnlyInclude")
    if only is not None:
        ids = only if isinstance(only, list) else []
    else:
        must = rb.get("mustInclude") or []
        ids = [bid for bid in must if bid not in pinned_in_seed]
    return list(dict.fromkeys(str(item) for item in ids))


def pass_type(expect: dict[str, Any], expected_relevant: Iterable[str]) -> str:
    """AMB's ``_pass_type``: active / structural / trivially-empty."""
    if list(expected_relevant):
        return "active"
    rb = expect.get("relevantBeliefs") or {}
    pf = expect.get("pinnedFacts") or {}
    other = (
        bool(rb.get("mustExclude"))
        or bool(expect.get("openQuestions"))
        or bool(pf.get("mustInclude"))
        or bool(pf.get("mustExclude"))
        or bool(expect.get("personaPrelude"))
    )
    return "structural" if other else "trivially-empty"


def _in_scope(belief: dict[str, Any], scope: list[str]) -> bool:
    if not scope:
        return True
    return any(value in scope for value in (belief.get("scope") or []))


def _expand_relation_participants(
    by_id: dict[str, dict[str, Any]],
    relation_ids: list[str],
    user_id: str,
    scope: list[str],
    exclude: set[str],
) -> list[str]:
    """Port of AMB's one-hop relation expansion, with the open-question guard."""
    out: list[str] = []
    for rid in relation_ids:
        relation = by_id.get(rid)
        if not relation or relation.get("type") != "relation":
            continue
        for pid in relation.get("participants") or []:
            if pid in exclude or pid not in by_id:
                continue
            participant = by_id[pid]
            if participant.get("user_id") != user_id:
                continue
            if participant.get("type") == "open_question":
                continue  # routed to the openQuestions tier by AMB
            if scope and not _in_scope(participant, scope):
                continue
            out.append(pid)
    return out


def score_case(
    case: dict[str, Any],
    beliefs: Iterable[dict[str, Any]],
    retrieved_ids: Iterable[str],
    k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Score one case exactly like the pinned AMB ``score_retrieval``.

    Tiers ``pinnedFacts`` and ``openQuestions`` are derived locally from the
    pinned seed corpus; ``relevantBeliefs`` is the only provider-dependent tier
    and is fed solely by ``retrieved_ids``. No answer model and no judge.
    """
    rows = list(beliefs)
    by_id = {str(b.get("_id")): b for b in rows}
    meta = case.get("meta") or {}
    expect = meta.get("expect") or {}
    scope = list(meta.get("scope") or [])
    user_id = str(case.get("user_id") or DEFAULT_USER_ID)
    retrieval_limit = int(meta.get("retrieval_limit") or DEFAULT_MAX_BELIEFS)
    budget = meta.get("budget") or {}
    max_questions = int(budget.get("maxQuestions") or DEFAULT_MAX_QUESTIONS)
    cap = min(int(k) if k and k > 0 else retrieval_limit, retrieval_limit)

    pinned_in_seed = {
        str(b.get("_id"))
        for b in rows
        if b.get("pinned") is True and b.get("user_id") == DEFAULT_USER_ID
    }

    def is_pinned(b: dict[str, Any]) -> bool:
        return (
            b.get("user_id") == user_id
            and b.get("pinned") is True
            and b.get("type") != "open_question"
            and not b.get("superseded_by")
            and not b.get("resolved_at")
            and _in_scope(b, scope)
        )

    def is_question(b: dict[str, Any]) -> bool:
        return (
            b.get("user_id") == user_id
            and b.get("type") == "open_question"
            and b.get("pinned") is True
            and not b.get("resolved_at")
            and _in_scope(b, scope)
        )

    pinned = [b for b in rows if is_pinned(b)]
    questions = [b for b in rows if is_question(b)]
    pinned_ids_all = {str(b.get("_id")) for b in pinned}

    raw = [
        bid for bid in (retrieved_ids or [])
        if bid in by_id and bid not in pinned_ids_all
    ][:cap]
    expansions = (
        _expand_relation_participants(by_id, raw, user_id, scope, pinned_ids_all | set(raw))
        if raw
        else []
    )
    capped_pinned = pinned[:cap]
    capped_relevant = (raw + expansions)[: max(0, cap - len(capped_pinned))]
    pinned_ids = {str(b.get("_id")) for b in capped_pinned}
    relevant_ids = list(dict.fromkeys(capped_relevant))
    relevant_set = set(relevant_ids)
    question_ids = {str(b.get("_id")) for b in questions[:max_questions]}
    union_ids = pinned_ids | relevant_set
    prelude = EVAL_PERSONA_UNIVERSAL if user_id == DEFAULT_USER_ID else ""

    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    rb = expect.get("relevantBeliefs") or {}
    for bid in rb.get("mustInclude") or []:
        check(bid in union_ids, f"missing expected belief: {bid}")
    for bid in rb.get("mustExclude") or []:
        check(bid not in union_ids, f"forbidden belief surfaced: {bid}")
    for bid in rb.get("shouldInclude") or []:
        check(bid in union_ids, f"expected belief missing (shouldInclude): {bid}")

    only = rb.get("shouldOnlyInclude")
    if only is not None:
        expected_set = set(only)
        for bid in relevant_ids:
            check(bid in expected_set, f"unexpected belief in relevantBeliefs: {bid}")
        for bid in expected_set:
            check(bid in relevant_set, f"missing expected belief: {bid}")

    if rb.get("maxCount") is not None:
        check(len(relevant_set) <= rb["maxCount"], f"relevantBeliefs count > maxCount {rb['maxCount']}")
    if rb.get("minCount") is not None:
        check(len(relevant_set) >= rb["minCount"], f"relevantBeliefs count < minCount {rb['minCount']}")
    for first, second in rb.get("orderedBefore") or []:
        i = relevant_ids.index(first) if first in relevant_ids else -1
        j = relevant_ids.index(second) if second in relevant_ids else -1
        check(i != -1, f"orderedBefore: {first} not in relevantBeliefs")
        check(j != -1, f"orderedBefore: {second} not in relevantBeliefs")
        if i != -1 and j != -1:
            check(i < j, f"ranking: {first} should precede {second}")

    pf = expect.get("pinnedFacts") or {}
    for bid in pf.get("mustInclude") or []:
        check(bid in pinned_ids, f"missing pinned belief: {bid}")
    for bid in pf.get("mustExclude") or []:
        check(bid not in pinned_ids, f"forbidden belief in pinnedFacts: {bid}")

    oq = expect.get("openQuestions") or {}
    for bid in oq.get("mustInclude") or []:
        check(bid in question_ids, f"missing expected question: {bid}")
    for bid in oq.get("mustExclude") or []:
        check(bid not in question_ids, f"forbidden question surfaced: {bid}")

    pp = expect.get("personaPrelude") or {}
    if pp.get("nonEmpty"):
        check(len(prelude) > 0, "personaPrelude empty")
    if pp.get("isNull"):
        check(prelude == "", "personaPrelude not empty")
    for text in pp.get("contains") or []:
        check(text in prelude, f'personaPrelude missing "{text}"')
    for text in pp.get("mustNotContain") or []:
        check(text not in prelude, f'personaPrelude contains "{text}"')

    expected = expected_relevant_ids(expect, pinned_in_seed)
    expected_set = set(expected)
    hits = len(expected_set & relevant_set)
    if not relevant_set and not expected_set:
        precision: float | None = None
    elif not relevant_set:
        precision = 0.0
    else:
        precision = hits / len(relevant_set)
    recall = hits / len(expected_set) if expected_set else None
    expected_pinned = set(pf.get("mustInclude") or [])
    pinned_coverage = (
        len(expected_pinned & pinned_ids) / len(expected_pinned) if expected_pinned else None
    )
    passed = not failures
    pt = pass_type(expect, expected)
    ranked = list(retrieved_ids or [])
    return {
        "passed": passed,
        "reason": "pass" if passed else "; ".join(failures[:4]),
        "pass_type": pt,
        "expected_relevant": expected,
        "topk_hit": bool(expected_set & set(ranked[: int(k) if k else DEFAULT_TOP_K])),
        "relevant_ids": relevant_ids,
        "pinned_ids": sorted(pinned_ids),
        "question_ids": sorted(question_ids),
        "precision": None if precision is None else round(precision, 6),
        "recall": None if recall is None else round(recall, 6),
        "pinned_coverage": pinned_coverage,
        "failures": failures,
        "retrieval_limit": retrieval_limit,
        "k": k,
        "user_id": user_id,
    }


# --------------------------------------------------------------------------- #
# Seed -> recall -> score, with zero gold leakage into the recall query
# --------------------------------------------------------------------------- #


def build_rows(
    cases: Iterable[dict[str, Any]],
    beliefs: Iterable[dict[str, Any]],
    *,
    seed_fn: Callable[[str, str], bool],
    recall_fn: Callable[[str, int], list[str]],
    k: int = DEFAULT_TOP_K,
    detail: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one fresh-vault chain per case and score belief-ID retrieval.

    Contracts:
      ``seed_fn(entity, text) -> bool``    -- one vault note per seeded belief
      ``recall_fn(query, k) -> [entity]`   -- receives the case query TEXT ONLY

    Only beliefs whose ``user_id`` equals the case's user are seeded, giving
    per-user isolation via one temp vault per case. The gold IDs stay in the
    returned row for local scoring and are stripped by :func:`public_row`.
    """
    forward, inverse = build_id_maps(beliefs)
    rows = list(beliefs)
    by_user: dict[str, list[dict[str, Any]]] = {}
    for belief in rows:
        by_user.setdefault(str(belief.get("user_id") or DEFAULT_USER_ID), []).append(belief)

    counters = {
        "cases": 0,
        "beliefs_seeded": 0,
        "recall_empty": 0,
        "gold_queries_seen": 0,
    }
    built: list[dict[str, Any]] = []
    for case in cases or []:
        uid = str(case.get("user_id") or DEFAULT_USER_ID)
        user_beliefs = by_user.get(uid, [])
        seeded = 0
        for belief in user_beliefs:
            bid = str(belief.get("_id"))
            if seed_fn(forward[bid], str(belief.get("document_text") or "")):
                seeded += 1

        query = str(case.get("query") or "")
        gold_ids = _case_gold_ids(case)
        # Hard guard: the recall query is the case query verbatim. Never gold.
        if any(str(gid) in query for gid in gold_ids):
            counters["gold_queries_seen"] += 1
        retrieved_names = [str(name) for name in (recall_fn(query, k) or [])]
        retrieved_ids = [inverse[name] for name in retrieved_names if name in inverse]
        scored = score_case(case, rows, retrieved_ids, k)

        built.append(
            {
                "case_id": str(case.get("id") or ""),
                "user_id": uid,
                "category": str((case.get("meta") or {}).get("category") or ""),
                "query_hash_prefix": sha256_text(query)[:12],
                "retrieved_ids": retrieved_ids,
                "retrieved_names": retrieved_names,
                "gold_hash_prefix": gold_hash_prefix(gold_ids),
                "hit": scored["passed"],
                "topk_hit": scored["topk_hit"],
                "pass_type": scored["pass_type"],
                "reason": scored["reason"],
                "failures": scored["failures"],
                "precision": scored["precision"],
                "recall": scored["recall"],
                "expected_relevant_count": len(scored["expected_relevant"]),
                "seeded_count": seeded,
                "k": k,
                "_gold_ids": gold_ids,  # in-memory only; public_row strips it
            }
        )
        counters["cases"] += 1
        counters["beliefs_seeded"] += seeded
        counters["recall_empty"] += int(not retrieved_ids)
    if detail is not None:
        detail.update(counters)
    return built


def _case_gold_ids(case: dict[str, Any]) -> list[str]:
    """AMB's Query.gold_ids: shouldOnlyInclude + mustInclude + pinned mustInclude."""
    expect = ((case.get("meta") or {}).get("expect") or {})
    rb = expect.get("relevantBeliefs") or {}
    pf = expect.get("pinnedFacts") or {}
    ids = [
        *(rb.get("shouldOnlyInclude") or []),
        *(rb.get("mustInclude") or []),
        *(pf.get("mustInclude") or []),
    ]
    return list(dict.fromkeys(str(item) for item in ids))


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    """Exactly the four checkpoint fields: no gold, no key, no raw query."""
    return {
        "case_id": row.get("case_id"),
        "retrieved_ids": list(row.get("retrieved_ids") or []),
        "gold_hash_prefix": row.get("gold_hash_prefix"),
        "hit": int(bool(row.get("hit"))),
    }


# --------------------------------------------------------------------------- #
# Evaluation and reporting
# --------------------------------------------------------------------------- #


def evaluate(results: list[Any]) -> dict[str, Any]:
    """Belief-ID accuracy with real denominators and all honesty stamps.

    The headline is ``active_passes/active_total`` (AMB's comparable number);
    total passes and the structural/empty splits are reported beside it with
    their own denominators. Invalid rows are excluded explicitly, never padded.
    """
    source_rows = stable_rows(results or [])
    rows: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    invalid = 0
    for index, record in enumerate(source_rows):
        errors = validate_result(record)
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
        valid.append(record)
        rows.append(
            {
                "index": index,
                "case_id": record.get("case_id"),
                "user_id": record.get("user_id"),
                "category": record.get("category"),
                "pass_type": record.get("pass_type"),
                "hit": bool(record.get("hit")),
                "topk_hit": bool(record.get("topk_hit")),
                "retrieved_count": len(record.get("retrieved_ids") or []),
                "gold_hash_prefix": gold_hash_prefix(record.get("_gold_ids")),
                "precision": record.get("precision"),
                "recall": record.get("recall"),
                "reason": record.get("reason"),
                "valid": True,
            }
        )

    by_type: dict[str, dict[str, Any]] = {}
    for pt in PASS_TYPES:
        group = [row for row in valid if row.get("pass_type") == pt]
        correct = sum(1 for row in group if bool(row.get("hit")))
        by_type[pt] = {
            "correct": correct,
            "denominator": len(group),
            "display": ratio_text(f"{pt} passes", correct, len(group), " (derived diagnostic)"),
        }
    total = len(valid)
    correct = sum(1 for row in valid if bool(row.get("hit")))
    active = by_type["active"]
    precisions = [row["precision"] for row in valid if row.get("precision") is not None]
    recalls = [row["recall"] for row in valid if row.get("recall") is not None]

    summary_parts = [
        ratio_text("rows supplied", len(source_rows), len(source_rows), ""),
        f"invalid rows excluded: {invalid}",
        ratio_text("belief-ID case pass", correct, total, " (derived; not an official score)"),
        active["display"],
        by_type["structural"]["display"],
        by_type["trivially-empty"]["display"],
        f"judge ran: false",
    ]
    extra: dict[str, Any] = {
        "comparability_tier": "C",
        "score_asserted": False,
        "not_an_official_score": True,
        "judge": None,
        "model_calls": 0,
        "answer_llm": None,
        "primary_metric": "belief_id_case_pass",
        "primary_comparable_metric": "active_passes",
        "active_passes": active["correct"],
        "active_total": active["denominator"],
        "mean_precision": round(sum(precisions) / len(precisions), 6) if precisions else None,
        "mean_recall": round(sum(recalls) / len(recalls), 6) if recalls else None,
        "precision_denominator": len(precisions),
        "recall_denominator": len(recalls),
        "by_pass_type": by_type,
        "official_split_available": False,
        "gold_withheld_from_recall_query": True,
        "amb_head": AMB_HEAD,
        "upstream_commit": UPSTREAM_COMMIT,
        "vendored_gz_sha256": {key: value[0] for key, value in VENDORED_GZ.items()},
        "upstream_fixture_sha256": {key: value[0] for key, value in UPSTREAM_FIXTURES.items()},
        "membench_data_status": RETIRED_MEMBENCH["status"],
    }
    return {
        "lane": LANE,
        "denominator": denominator(valid),
        "correct": correct,
        "received_rows": len(source_rows),
        "invalid_rows": invalid,
        "rows": rows,
        "metrics": {
            "primary": {
                "name": "belief_id_case_pass",
                "correct": correct,
                "denominator": total,
                "display": ratio_text("belief-ID case pass", correct, total, " (derived)"),
                "judge_free": True,
            },
            "active_passes": {
                "correct": active["correct"],
                "denominator": active["denominator"],
                "display": active["display"],
            },
            "by_pass_type": by_type,
            "judge": {"ran": False, "note": "retrieval mode is judge-free; judge=null"},
        },
        "summary": "; ".join(summary_parts),
        "source_url": SOURCE_URL,
        "pmb_source_url": PMB_SOURCE_URL,
        **stamps(_NOTE, judge=None, extra=extra),
    }


def select() -> dict[str, Any]:
    """Pinned dataset description without network or file I/O."""
    split: dict[str, int | None] = {
        "active": N_ACTIVE,
        "structural": N_STRUCTURAL,
        "trivially-empty": N_EMPTY,
    }
    local_presence: dict[str, bool] = {}
    root = DEFAULT_DATA_DIR
    local_presence["prepared"] = root.is_dir()
    if root.is_dir():
        local_presence["documents"] = (
            root / "single-turn/documents.json.gz"
        ).is_file()
        local_presence["queries"] = (root / "single-turn/queries.json.gz").is_file()
        local_presence["beliefs"] = (root / "beliefs.seed.json").is_file()
    return {
        "lane": LANE,
        "source": SOURCE_URL,
        "paper": SOURCE_PAPER_URL,
        "pmb_source": PMB_SOURCE_URL,
        "amb_head": AMB_HEAD,
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset": "precisionmembench/single-turn",
        "mode": "retrieval",
        "judge": None,
        "answer_llm": None,
        "cases": N_CASES,
        "beliefs": N_BELIEFS,
        "pinned_pass_types": split,
        "pass_types_resolved_offline": False,
        "default_top_k": DEFAULT_TOP_K,
        "official_split": False,
        "vendored_gz_sha256": {key: value[0] for key, value in VENDORED_GZ.items()},
        "upstream_fixture_sha256": {key: value[0] for key, value in UPSTREAM_FIXTURES.items()},
        "membench_data_status": RETIRED_MEMBENCH["status"],
        "local_presence": local_presence,
        "note": _NOTE,
    }


# --------------------------------------------------------------------------- #
# Offline synthetic self-test: no network, no keys, no model, no judge
# --------------------------------------------------------------------------- #


def self_test() -> dict[str, Any]:
    """Offline checks of pins, ID mapping, gold isolation, scoring and stamps."""
    # Pin hygiene: every recorded hash is 64 hex; UNVERIFIED data is refused.
    for mapping in (VENDORED_GZ, UPSTREAM_FIXTURES):
        for name, (sha, size) in mapping.items():
            assert _HEX64.fullmatch(sha), f"bad SHA pin for {name}"
            assert isinstance(size, int) and size > 0, f"bad byte pin for {name}"
    assert len(VENDORED_GZ) == 5
    assert RETIRED_MEMBENCH["status"].startswith("UNVERIFIED")
    try:
        refuse_membench_data()
    except RuntimeError:
        pass
    else:
        raise AssertionError("refuse_membench_data() did not refuse")

    # Synthetic belief corpus: test-user beliefs plus one other-user belief.
    def belief(bid: str, **extra: Any) -> dict[str, Any]:
        row: dict[str, Any] = {
            "_id": bid,
            "user_id": "test-user",
            "type": "preference",
            "canonical_name": bid.replace("b-", ""),
            "aliases": [],
            "scope": ["domain:code"],
            "pinned": False,
            "content": f"Synthetic content for {bid}.",
            "document_text": f"Synthetic content for {bid}.",
        }
        row.update(extra)
        return row

    beliefs = [
        belief("b-syn-active"),
        belief("b-syn-pinned", pinned=True, type="decision"),
        belief(
            "b-syn-superseded", type="decision", superseded_by="b-syn-pinned"
        ),
        belief("b-syn-question", pinned=True, type="open_question"),
        belief("b-syn-other", user_id="other-user", content="Other user likes Haskell.",
               document_text="Other user likes Haskell."),
    ]
    forward, inverse = build_id_maps(beliefs)
    assert len(forward) == 5 and len(inverse) == 5
    assert all(inverse[forward[bid]] == bid for bid in forward)
    assert belief_entity("b-db-decision") == "b-db-decision"

    def case(cid: str, query: str, expect: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": cid,
            "query": query,
            "user_id": "test-user",
            "meta": {
                "category": "synthetic",
                "scope": ["domain:code"],
                "retrieval_limit": DEFAULT_MAX_BELIEFS,
                "budget": {},
                "expect": expect,
            },
        }

    cases = [
        case("syn-active-hit", "blue notebooks?", {
            "relevantBeliefs": {"shouldOnlyInclude": ["b-syn-active"]},
        }),
        case("syn-active-miss", "wrong notebooks?", {
            "relevantBeliefs": {"shouldOnlyInclude": ["b-syn-active"]},
        }),
        case("syn-structural-hit", "auth?", {
            "pinnedFacts": {"mustInclude": ["b-syn-pinned"]},
            "relevantBeliefs": {"shouldOnlyInclude": []},
        }),
        case("syn-structural-miss", "old db?", {
            "pinnedFacts": {"mustInclude": ["b-syn-pinned"]},
            "relevantBeliefs": {
                "mustExclude": ["b-syn-superseded"],
                "shouldOnlyInclude": [],
            },
        }),
        case("syn-empty-hit", "nothing?", {
            "relevantBeliefs": {"shouldOnlyInclude": []},
        }),
    ]
    assert compute_pass_types(beliefs, cases) == {
        "active": 2, "structural": 2, "trivially-empty": 1
    }

    # Deterministic recall stub keyed only by query text; proves no gold leak.
    answers = {
        "blue notebooks?": ["b-syn-active"],
        "wrong notebooks?": ["b-syn-superseded"],
        "auth?": [],
        "old db?": ["b-syn-superseded"],
        "nothing?": [],
    }
    gold_by_case = {
        "syn-active-hit": ["b-syn-active"],
        "syn-active-miss": ["b-syn-active"],
        "syn-structural-hit": [],
        "syn-structural-miss": [],
        "syn-empty-hit": [],
    }
    calls: dict[str, Any] = {"seeded": [], "queries": []}

    def seed_fn(entity: str, text: str) -> bool:
        calls["seeded"].append((entity, text))
        return True

    def recall_fn(query: str, k: int) -> list[str]:
        assert k == DEFAULT_TOP_K
        calls["queries"].append(query)
        for gids in gold_by_case.values():
            assert not any(gid in query for gid in gids), "gold leaked into recall query"
        return [forward[bid] for bid in answers.get(query, []) if bid in forward]

    detail: dict[str, Any] = {}
    built = build_rows(
        cases, beliefs, seed_fn=seed_fn, recall_fn=recall_fn,
        k=DEFAULT_TOP_K, detail=detail,
    )
    assert detail["cases"] == 5
    assert detail["beliefs_seeded"] == 5 * 4  # four test-user beliefs per fresh vault
    assert detail["gold_queries_seen"] == 0
    assert all(entity != forward["b-syn-other"] for entity, _ in calls["seeded"])
    assert [row["pass_type"] for row in built] == [
        "active", "active", "structural", "structural", "trivially-empty"
    ]
    assert [bool(row["hit"]) for row in built] == [True, False, True, False, True]
    assert built[0]["retrieved_ids"] == ["b-syn-active"]
    assert built[1]["retrieved_ids"] == ["b-syn-superseded"]

    checkpoint = [public_row(row) for row in built]
    assert all(
        set(row) == {"case_id", "retrieved_ids", "gold_hash_prefix", "hit"}
        for row in checkpoint
    )
    assert all("gold" not in row and "key" not in row and "query" not in row for row in checkpoint)

    result = evaluate(built)
    assert result["official"] is False
    assert result["diagnostic"] == "derived"
    assert result["comparability_tier"] == "C"
    assert result["score_asserted"] is False
    assert result["not_an_official_score"] is True
    assert result["judge"] is None
    assert result["model_calls"] == 0
    assert result["denominator"] == 5
    assert result["correct"] == 3
    assert result["invalid_rows"] == 0
    assert result["active_passes"] == 1 and result["active_total"] == 2
    assert result["metrics"]["by_pass_type"]["structural"]["denominator"] == 2
    assert result["metrics"]["by_pass_type"]["trivially-empty"]["denominator"] == 1
    assert len(result["rows"]) == 5

    # Invalid rows are excluded explicitly, never padded into a denominator.
    invalid_result = evaluate(built + [{"case_id": "bad", "meta": "not-an-object"}])
    assert invalid_result["invalid_rows"] == 1
    assert invalid_result["denominator"] == 5

    return {
        "lane": LANE,
        "checks": {
            "pinned_gz_files": len(VENDORED_GZ),
            "pinned_upstream_fixtures": len(UPSTREAM_FIXTURES),
            "membench_refused": True,
            "id_map_bijective": len(forward) == len(inverse) == 5,
            "synthetic_cases": len(cases),
            "vault_seeds": len(calls["seeded"]),
            "recall_calls": len(calls["queries"]),
            "gold_leak_into_recall_query": detail["gold_queries_seen"],
            "other_user_seeded": 0,
            "pass_type_split": compute_pass_types(beliefs, cases),
            "checkpoint_fields": sorted(checkpoint[0]),
        },
        "evaluation_summary": result["summary"],
        "denominator": result["denominator"],
        "correct": result["correct"],
        "active_passes": result["active_passes"],
        "active_total": result["active_total"],
        "official": result["official"],
        "diagnostic": result["diagnostic"],
        "comparability_tier": result["comparability_tier"],
        "score_asserted": result["score_asserted"],
        "judge": result["judge"],
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
