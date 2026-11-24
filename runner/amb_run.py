"""Standalone, key-free, checkpointed AMB / PrecisionMemBench retrieval runner.

This runner deliberately does NOT go through ``runner/bench.py`` and does NOT
import any key pool, answer model or judge. The pinned ``precisionmembench``
lane in AMB ``--mode retrieval`` is fully deterministic: per case, a fresh temp
vault is seeded from the pinned belief corpus via the public ``mn vault
remember`` surface (8-way concurrent CLI writes), retrieval is a single
``mn vault recall`` on the case's query text, and scoring is local belief-ID
case pass/fail. No key value, model call or judge verdict exists in this lane.

HONESTY CONTRACT
----------------
* Every artifact is stamped official=False, diagnostic="derived",
  comparability_tier="C", score_asserted=False, judge=null, keys_loaded=0,
  model_calls=0 and "NOT an official score". No rank, win or leaderboard
  comparison is made.
* Gold belief IDs are used only by the adapter's local scorer. The recall query
  is the case query verbatim; checkpoints carry only a gold hash prefix.
* Dataset bytes come only from the pinned resolve/<commit>/ URLs in
  ``harnesses.amb.adapter`` and are SHA-256 + byte-count verified before use,
  landing under D:\\memnoth-bench-data, never inside the repository.
* ``--dry-run`` performs no network, no CLI, no model and no key load.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import mn_json  # noqa: E402  (isolated env; reads no key values)
from harnesses.amb import adapter  # noqa: E402

VERSION = "0.1.0"
SCHEMA = "memnoth-bench-amb-run/1"
NOTE = (
    "INFORMATIONAL ONLY — NOT AN OFFICIAL SCORE. Derived diagnostic on the "
    "pinned 77-case PrecisionMemBench single-turn retrieval lane (AMB --mode "
    "retrieval): belief-ID top-k accuracy with no answer model and no judge. "
    "official=false, diagnostic=derived, comparability_tier=C, "
    "score_asserted=false. No rank, win or comparison to published rows is "
    "claimed or implied."
)

DATA_ROOT = Path(r"D:\memnoth-bench-data")
DEFAULT_DATASET_DIR = DATA_ROOT / "datasets" / "amb"
DEFAULT_CHECKPOINT_DIR = DATA_ROOT / "checkpoints" / "amb"
DEFAULT_RESULTS_DIR = DATA_ROOT / "results"

SEED_WORKERS = 8  # 8-way CLI seeding via vault remember, per case
MAX_WORKERS_CEILING = 16
MAX_CLI_TEXT = 2000
_ENTITY_UNSAFE = __import__("re").compile(r"[^A-Za-z0-9_.-]+")

CHECKPOINT_FIELDS = ("case_id", "retrieved_ids", "gold_hash_prefix", "hit")


# --------------------------------------------------------------------------- #
# Honest helpers (no key values, no gold values in published artifacts)
# --------------------------------------------------------------------------- #


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_text(value: Any) -> str:
    return adapter.sha256_text(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
    """One model-free ``mn vault remember`` call against this case's fresh vault."""
    result = mn_json.run_verb(
        "vault",
        ["remember", entity_name(entity), str(text)[:MAX_CLI_TEXT], "--vault", str(vault)],
        bin_path,
        timeout=30.0,
        cwd=cli_root,
    )
    return bool(result.get("ok")) and "error" not in result


def vault_recall_names(
    bin_path: Path | None, cli_root: Path | None, vault: Path, query: str, limit: int
) -> list[str]:
    """One model-free ``mn vault recall`` call; returns ranked entity names."""
    result = mn_json.run_verb(
        "vault",
        ["recall", str(query)[:MAX_CLI_TEXT], "--limit", str(limit), "--vault", str(vault)],
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


def seed_beliefs(
    bin_path: Path | None,
    cli_root: Path | None,
    vault: Path,
    beliefs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Seed one case's user-scoped beliefs with 8-way CLI fan-out.

    Returns an ``entity -> ok`` map plus count-only diagnostics. A failed write
    is recorded as False, never silently retried into a fabricated success.
    """
    detail: dict[str, Any] = {"attempted": 0, "ok": 0, "failed": 0, "error_types": []}
    results: dict[str, bool] = {}

    def one(belief: dict[str, Any]) -> tuple[str, bool]:
        bid = str(belief.get("_id") or "")
        entity = adapter.belief_entity(bid)
        text = str(belief.get("document_text") or "")
        return entity, vault_remember(bin_path, cli_root, vault, entity, text)

    with ThreadPoolExecutor(max_workers=SEED_WORKERS) as executor:
        futures = [executor.submit(one, belief) for belief in beliefs]
        for future in as_completed(futures):
            detail["attempted"] += 1
            try:
                entity, ok = future.result()
            except Exception as exc:  # a worker crash is a failed seed, not a pass
                entity, ok = None, False
                detail["error_types"].append(type(exc).__name__)
            if entity is not None:
                results[entity] = ok
            if ok:
                detail["ok"] += 1
            else:
                detail["failed"] += 1
    detail["error_types"] = sorted(set(detail["error_types"]))
    return {"by_entity": results, "detail": detail}


def checkpoint_row(record: dict[str, Any]) -> dict[str, Any]:
    """Exactly the four declared checkpoint fields, in a fixed order."""
    return {field: record.get(field) for field in CHECKPOINT_FIELDS}


def load_done(checkpoint_path: Path) -> set[str]:
    done: set[str] = set()
    if checkpoint_path.is_file():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["case_id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


def base_envelope(status: str, *, selected: int = 0) -> dict[str, Any]:
    """Base result with mandatory honesty stamps and zeroed metrics."""
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "observed_at": utc_now(),
        "official": False,
        "diagnostic": "derived",
        "comparability_tier": "C",
        "score_asserted": False,
        "not_an_official_score": True,
        "judge": None,
        "keys_loaded": 0,
        "model_calls": 0,
        "answer_llm": None,
        "note": NOTE,
        "status": status,
        "lane": adapter.LANE,
        "denominators": {
            "selected": selected,
            "attempted": 0,
            "scored": 0,
            "resumed_skipped": 0,
            "active": 0,
            "structural": 0,
            "trivially-empty": 0,
        },
        "belief_id_accuracy": {"correct": 0, "denominator": 0, "display": "0/0 (no denominator)"},
        "active_passes": 0,
        "active_total": 0,
        "cases": [],
        "limitations": [
            "Derived diagnostic on the pinned 77-case PMB single-turn retrieval "
            "lane, not an official AMB or PrecisionMemBench submission.",
            "The local scorer ports the pinned AMB scoring semantics but runs "
            "outside the AMB harness; comparability_tier=C, not Tier A.",
            "Default top-k is 10 while AMB's pinned maxBeliefs budget is 20; the "
            "effective budget is min(top-k, 20) and is recorded per case.",
            "Each case gets a fresh temp vault seeded only with that case's "
            "user_id beliefs; this is the honest equivalent of AMB's per-call "
            "user_id filter and of /reset.",
            "Gold belief IDs are never sent to recall; checkpoints carry only a "
            "gold hash prefix. No key, answer model or judge is used.",
        ],
    }


# --------------------------------------------------------------------------- #
# One sequential case chain: fresh vault -> 8-way seed -> query-only recall -> score
# --------------------------------------------------------------------------- #


def process_case(
    *,
    index: int,
    case: dict[str, Any],
    beliefs: list[dict[str, Any]],
    bin_path: Path | None,
    cli_root: Path | None,
    top_k: int,
) -> dict[str, Any]:
    """Run one case and return its full in-memory row (gold stripped publicly)."""
    uid = str(case.get("user_id") or adapter.DEFAULT_USER_ID)
    user_beliefs = [
        belief
        for belief in beliefs
        if str(belief.get("user_id") or adapter.DEFAULT_USER_ID) == uid
    ]
    case_id = str(case.get("id") or f"case-{index}")
    case_detail: dict[str, Any] = {}

    try:
        with tempfile.TemporaryDirectory(prefix="mn-bench-amb-") as temporary:
            vault = Path(temporary) / "vault.db"
            seeded = seed_beliefs(bin_path, cli_root, vault, user_beliefs)

            def seed_fn(entity: str, text: str) -> bool:
                # The 8-way fan-out already ran; report its real outcome.
                return bool(seeded["by_entity"].get(entity, False))

            def recall_fn(query: str, k: int) -> list[str]:
                limit = int((case.get("meta") or {}).get("retrieval_limit") or 20)
                # AMB: no search for a blank query or a zero belief budget.
                if not str(query or "").strip() or min(k, limit) <= 0:
                    return []
                return vault_recall_names(bin_path, cli_root, vault, query, k)

            rows = adapter.build_rows(
                [case], beliefs, seed_fn=seed_fn, recall_fn=recall_fn,
                k=top_k, detail=case_detail,
            )
            row = dict(rows[0])
    except Exception as exc:  # never let a worker exception fabricate a score
        row = {
            "case_id": case_id,
            "user_id": uid,
            "category": str((case.get("meta") or {}).get("category") or ""),
            "query_hash_prefix": None,
            "retrieved_ids": [],
            "retrieved_names": [],
            "gold_hash_prefix": adapter.gold_hash_prefix(adapter._case_gold_ids(case)),
            "hit": False,
            "topk_hit": False,
            "pass_type": "active",
            "reason": f"case_error_{type(exc).__name__}",
            "failures": [f"case_error_{type(exc).__name__}"],
            "precision": None,
            "recall": None,
            "expected_relevant_count": 0,
            "seeded_count": 0,
            "k": top_k,
            "_gold_ids": adapter._case_gold_ids(case),
        }
        case_detail = {"cases": 1, "beliefs_seeded": 0, "recall_empty": 1, "gold_queries_seen": 0}
        seeded = {"detail": {"attempted": 0, "ok": 0, "failed": 0, "error_types": []}}

    row["_seed_detail"] = seeded["detail"]
    row["_case_detail"] = case_detail
    return row


# --------------------------------------------------------------------------- #
# CLI / argparse
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-cases", type=int, default=adapter.N_CASES,
        help=f"cases to run (default {adapter.N_CASES}; 0 = all)",
    )
    parser.add_argument(
        "--top-k", type=int, default=adapter.DEFAULT_TOP_K,
        help=f"belief IDs requested per recall (default {adapter.DEFAULT_TOP_K})",
    )
    parser.add_argument("--workers", type=int, default=8, help="case workers, capped at 16")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="verify pins/selection offline; no network, no CLI, no model, no keys",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Dry run: offline only, proves zero keys / zero model calls / judge=null
# --------------------------------------------------------------------------- #


def run_dry(
    *, args: argparse.Namespace, result_path: Path, started: float
) -> int:
    selected = adapter.N_CASES if args.max_cases == 0 else min(args.max_cases, adapter.N_CASES)
    split = {
        "active": adapter.N_ACTIVE,
        "structural": adapter.N_STRUCTURAL,
        "trivially-empty": adapter.N_EMPTY,
    }
    split_source = "pinned AMB self-check split (no local dataset bytes inspected)"
    resolved_offline = False
    dataset: dict[str, Any] = {
        "dir": str(adapter.DEFAULT_DATA_DIR),
        "present": False,
        "verified": False,
        "note": "no network access; only a locally prepared, hash-verified dataset was inspected",
    }
    root = adapter.DEFAULT_DATA_DIR
    expected = list(adapter.VENDORED_GZ) + list(adapter.UPSTREAM_FIXTURES)
    all_present = all((root / name).is_file() for name in expected)
    if all_present:
        try:
            beliefs, cases, _ = adapter.load_payload(root)
            split = adapter.compute_pass_types(beliefs, cases)
            split_source = "resolved offline from verified local PMB bytes"
            resolved_offline = True
            dataset.update({"present": True, "verified": True, "cases": len(cases), "beliefs": len(beliefs)})
        except Exception as exc:
            dataset["note"] = f"local dataset failed verification: {type(exc).__name__}: {exc}"

    result = base_envelope("dry_run", selected=selected)
    result.update(
        {
            "dataset": dataset,
            "selection": {
                "cases": adapter.N_CASES,
                "beliefs": adapter.N_BELIEFS,
                "max_cases": args.max_cases,
                "selected": selected,
                "pass_types": split,
                "pass_type_source": split_source,
                "pass_types_resolved_offline": resolved_offline,
                "top_k": args.top_k,
                "effective_budget": min(args.top_k, adapter.DEFAULT_MAX_BELIEFS),
            },
            "pins": {
                "amb_head": adapter.AMB_HEAD,
                "upstream_commit": adapter.UPSTREAM_COMMIT,
                "vendored_gz_sha256": {key: value[0] for key, value in adapter.VENDORED_GZ.items()},
                "upstream_fixture_sha256": {
                    key: value[0] for key, value in adapter.UPSTREAM_FIXTURES.items()
                },
                "membench_data_status": adapter.RETIRED_MEMBENCH["status"],
            },
            "retrieval": {
                "mode": "retrieval",
                "recall_query": "case query text only; gold never sent",
                "vault_scope": "one fresh temp vault per case, seeded with that user's beliefs",
                "seed_workers": SEED_WORKERS,
                "case_workers": max(1, min(args.workers, MAX_WORKERS_CEILING)),
            },
            "calls_made": {
                "network": False,
                "cli": False,
                "model": False,
                "judge": False,
                "keys_loaded": False,
            },
            "keys_loaded": 0,
            "model_calls": 0,
            "judge": None,
        }
    )
    result["wall_time_s"] = round(time.monotonic() - started, 3)
    result["limitations"].append(
        "Dry run made no calls: it asserts no score, ran no recall and loaded no keys."
    )
    write_json(result_path, result)

    print("dry_run: network=false cli=false model=false judge=false")
    print("dry_run: keys_loaded=0 model_calls=0 judge=null")
    print(
        f"pins: amb_head={adapter.AMB_HEAD} upstream_commit={adapter.UPSTREAM_COMMIT} "
        f"vendored_gz={len(adapter.VENDORED_GZ)} upstream_fixtures={len(adapter.UPSTREAM_FIXTURES)}"
    )
    print(
        f"selection: cases={adapter.N_CASES} selected={selected} beliefs={adapter.N_BELIEFS} "
        f"active={split['active']} structural={split['structural']} "
        f"empty={split['trivially-empty']} source={split_source}"
    )
    print(f"retrieval: top_k={args.top_k} effective_budget={min(args.top_k, 20)} seed_workers={SEED_WORKERS}")
    print("official=false diagnostic=derived comparability_tier=C score_asserted=false judge=null")
    print(f"result: {result_path}")
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_cases < 0 or args.top_k < 1 or args.workers < 1:
        print("error: --max-cases must be >= 0, --top-k >= 1 and --workers >= 1")
        return 2

    started = time.monotonic()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    results_dir = Path(args.results_dir).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(args.max_cases) if args.max_cases else "all"
    checkpoint_path = checkpoint_dir / f"amb_max{suffix}_top{args.top_k}_checkpoint.jsonl"
    result_path = results_dir / (
        "amb_dry_run.json" if args.dry_run else f"amb_max{suffix}_top{args.top_k}.json"
    )

    if args.dry_run:
        return run_dry(args=args, result_path=result_path, started=started)

    # Network boundary: prepared only in a real run, verified before any use.
    try:
        data_dir = adapter.prepare(DEFAULT_DATASET_DIR)
    except Exception as exc:
        print(f"blocker: dataset unavailable ({type(exc).__name__}: {exc})")
        return 2
    try:
        beliefs, cases, documents = adapter.load_payload(data_dir)
    except Exception as exc:
        # Exact structure mismatch is reported, never silently dropped.
        print(f"blocker: PMB structure/sha mismatch ({type(exc).__name__}: {exc})")
        return 2

    cases = cases[: args.max_cases] if args.max_cases else cases
    if not cases:
        print("blocker: pinned dataset produced no selected cases")
        return 2

    bin_path, cli_root = resolve_cli()
    if bin_path is None:
        print("blocker: mn binary not found; set MN_BIN to a public CLI build")
        return 2

    done = load_done(checkpoint_path)
    target = [
        (index, case)
        for index, case in enumerate(cases)
        if str(case.get("id")) not in done
    ]
    print(
        f"checkpoint_resume: already_done={len(done)} selected={len(cases)} "
        f"target={len(target)}"
    )
    if not target and result_path.is_file():
        print("resume no-op: all selected cases already checkpointed; prior result preserved")
        return 0

    case_workers = max(1, min(args.workers, MAX_WORKERS_CEILING))
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=case_workers) as executor:
        futures = {
            executor.submit(
                process_case,
                index=index,
                case=case,
                beliefs=beliefs,
                bin_path=bin_path,
                cli_root=cli_root,
                top_k=args.top_k,
            ): index
            for index, case in target
        }
        # The main thread owns the append so concurrent cases cannot interleave bytes.
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint_file:
            for future in as_completed(futures):
                record = future.result()
                checkpoint_file.write(
                    json.dumps(checkpoint_row(record), sort_keys=True, ensure_ascii=False) + "\n"
                )
                checkpoint_file.flush()
                records.append(record)

    evaluation = adapter.evaluate(records)
    by_type = evaluation["metrics"]["by_pass_type"]
    attempted = len(records)
    seeds_ok = sum(int(r.get("_seed_detail", {}).get("ok", 0)) for r in records)
    seeds_failed = sum(int(r.get("_seed_detail", {}).get("failed", 0)) for r in records)
    topk_hits = sum(1 for r in records if bool(r.get("topk_hit")))

    result = base_envelope("completed", selected=len(cases))
    result.update(
        {
            "dataset": {
                "dir": str(data_dir),
                "documents": len(documents),
                "beliefs": len(beliefs),
                "cases_available": len(cases),
                "verified": True,
                "url": adapter.SOURCE_URL,
                "amb_head": adapter.AMB_HEAD,
                "upstream_commit": adapter.UPSTREAM_COMMIT,
                "vendored_gz_sha256": {
                    key: value[0] for key, value in adapter.VENDORED_GZ.items()
                },
                "upstream_fixture_sha256": {
                    key: value[0] for key, value in adapter.UPSTREAM_FIXTURES.items()
                },
            },
            "selection": {
                "cases": adapter.N_CASES,
                "selected": len(cases),
                "top_k": args.top_k,
                "effective_budget": min(args.top_k, adapter.DEFAULT_MAX_BELIEFS),
                "known_pass_types": {
                    "active": adapter.N_ACTIVE,
                    "structural": adapter.N_STRUCTURAL,
                    "trivially-empty": adapter.N_EMPTY,
                },
            },
            "retrieval": {
                "mode": "retrieval",
                "recall_query": "case query text only; gold never sent",
                "vault_scope": "one fresh temp vault per case, seeded with that user's beliefs",
                "seed_workers": SEED_WORKERS,
                "case_workers": case_workers,
                "seeds_ok": seeds_ok,
                "seeds_failed": seeds_failed,
                "topk_hits": topk_hits,
            },
            "denominators": {
                "selected": len(cases),
                "attempted": attempted,
                "scored": evaluation["denominator"],
                "resumed_skipped": len(done),
                "active": by_type["active"]["denominator"],
                "structural": by_type["structural"]["denominator"],
                "trivially-empty": by_type["trivially-empty"]["denominator"],
            },
            "belief_id_accuracy": {
                "correct": evaluation["correct"],
                "denominator": evaluation["denominator"],
                "display": evaluation["metrics"]["primary"]["display"],
            },
            "active_passes": evaluation["active_passes"],
            "active_total": evaluation["active_total"],
            "mean_precision": evaluation["mean_precision"],
            "mean_recall": evaluation["mean_recall"],
            "precision_denominator": evaluation["precision_denominator"],
            "recall_denominator": evaluation["recall_denominator"],
            "evaluation_summary": evaluation["summary"],
            "wall_time_s": round(time.monotonic() - started, 3),
            "checkpoint_path": str(checkpoint_path),
            "cases": [
                {
                    "case_id": record.get("case_id"),
                    "retrieved_ids": list(record.get("retrieved_ids") or []),
                    "gold_hash_prefix": record.get("gold_hash_prefix"),
                    "hit": int(bool(record.get("hit"))),
                    "pass_type": record.get("pass_type") or "unknown",
                    "reason": record.get("reason"),
                    "seeded": int(record.get("seeded_count") or 0),
                    "retrieved_count": len(record.get("retrieved_ids") or []),
                }
                for record in records
            ],
            "judge": None,
            "keys_loaded": 0,
            "model_calls": 0,
        }
    )
    # Hard guard: no gold-ID field and no key field may reach a published artifact.
    forbidden = {"gold", "gold_ids", "key", "api_key", "secret", "token"}
    assert all(
        not (set(case) & forbidden) for case in result["cases"]
    ), "gold/key field leaked into published cases"

    write_json(result_path, result)
    print("amb: official=false diagnostic=derived comparability_tier=C score_asserted=false judge=null")
    print(f"selection: n={len(cases)} amb_head={adapter.AMB_HEAD} upstream={adapter.UPSTREAM_COMMIT}")
    print(
        f"denominators: attempted={attempted} scored={evaluation['denominator']} "
        f"active={by_type['active']['denominator']} structural={by_type['structural']['denominator']} "
        f"empty={by_type['trivially-empty']['denominator']}"
    )
    print(
        f"local_scorer: case_pass={evaluation['correct']}/{evaluation['denominator']} "
        f"active_passes={evaluation['active_passes']}/{evaluation['active_total']} "
        f"(derived diagnostic; NOT an official score)"
    )
    print(
        f"retrieval: top_k={args.top_k} workers={case_workers} "
        f"seeds_ok={seeds_ok} seeds_failed={seeds_failed} topk_hits={topk_hits}"
    )
    print("keys_loaded=0 model_calls=0 judge=null")
    print(f"checkpoint: {checkpoint_path}")
    print(f"result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
