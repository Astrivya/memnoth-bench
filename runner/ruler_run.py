"""Judge-free RULER retrieval diagnostic runner (derived, no keys, no model calls).

The haystack from each pinned sample is seeded into a temporary vault with
``mn vault remember`` (one deterministic entity per chunk), then the RULER
question is sent to ``mn vault recall`` and the top recalled entity content is
read with ``mn vault asof``. Scoring is RULER's case-insensitive substring
containment, computed locally by ``harnesses.ruler.adapter``. No reader model,
no judge, no API key and no network call at scoring time.

HONESTY CONTRACT
----------------
* official=False, diagnostic="derived", comparability_tier="C",
  score_asserted=False. No win, rank or comparison is claimed.
* Gold is never placed in a recall query, never written to a checkpoint and
  never written to a result: only a gold hash prefix is stored.
* CLI calls use ``adapters.mn_json`` with an isolated home and no inherited keys.
  This runner deliberately does NOT import the key pool or any judge module.
* The dataset is written only under D:\\memnoth-bench-data, never in the repo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import mn_json  # noqa: E402

# Namespace subpackage (no __init__.py needed) or direct file fallback.
try:
    from harnesses.ruler import adapter  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - only on exotic import setups
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ruler_adapter", ROOT / "harnesses" / "ruler" / "adapter.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load harnesses/ruler/adapter.py")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules["ruler_adapter"] = adapter
    spec.loader.exec_module(adapter)

VERSION = "0.1.0"
SCHEMA = "memnoth-bench-ruler-run/1"
NOTE = (
    "INFORMATIONAL ONLY — NOT AN OFFICIAL RULER SCORE. Derived, judge-free "
    "retrieval diagnostic: the vault's recall replaces RULER's LLM reader. "
    "No score is asserted as a product result and no ranking is implied."
)
DATA_ROOT = Path(r"D:\memnoth-bench-data")
DEFAULT_CHECKPOINT_DIR = DATA_ROOT / "checkpoints" / "ruler"
DEFAULT_RESULTS_DIR = DATA_ROOT / "results"
MAX_CLI_SEED_WORKERS = 8
CLI_TIMEOUT = 60.0


def utc_now() -> str:
    return adapter.utc_now()


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", "replace")).hexdigest()


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


# --------------------------------------------------------------------------- #
# Model-free CLI plumbing (no key pool is imported or touched)
# --------------------------------------------------------------------------- #


def vault_init(bin_path: Path, cli_root: Path | None, vault: Path) -> bool:
    result = mn_json.run_verb(
        "vault",
        ["init", "--vault", str(vault)],
        bin_path,
        timeout=CLI_TIMEOUT,
        cwd=cli_root,
    )
    return bool(result.get("ok")) and "error" not in result


def vault_remember(
    bin_path: Path, cli_root: Path | None, vault: Path, entity: str, text: str
) -> bool:
    result = mn_json.run_verb(
        "vault",
        ["remember", entity, text, "--vault", str(vault)],
        bin_path,
        timeout=CLI_TIMEOUT,
        cwd=cli_root,
    )
    return bool(result.get("ok")) and "error" not in result


def vault_recall(
    bin_path: Path, cli_root: Path | None, vault: Path, query: str, limit: int
) -> list[str]:
    """Recall by the gold-free RULER question; returns ranked entity names."""
    result = mn_json.run_verb(
        "vault",
        ["recall", query, "--limit", str(limit), "--vault", str(vault)],
        bin_path,
        timeout=CLI_TIMEOUT,
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


def vault_asof(
    bin_path: Path, cli_root: Path | None, vault: Path, entity: str
) -> list[str]:
    result = mn_json.run_verb(
        "vault",
        ["asof", "--name", entity, "--vault", str(vault)],
        bin_path,
        timeout=CLI_TIMEOUT,
        cwd=cli_root,
    )
    if not result.get("ok") or "error" in result:
        return []
    facts = (result.get("data") or {}).get("facts") or []
    return [str(fact.get("content")) for fact in facts if isinstance(fact, dict)]


def seed_rows(
    rows: list[dict[str, Any]],
    *,
    bin_path: Path,
    cli_root: Path | None,
    vault: Path,
    workers: int,
) -> dict[str, Any]:
    """Seed every chunk with 8-way CLI parallelism. No gold is ever seeded."""
    tasks: list[tuple[str, str]] = []
    for row in rows:
        for entity, chunk in zip(row["entities"], row["chunks"]):
            tasks.append((entity, chunk))
    detail: dict[str, Any] = {
        "entities_planned": len(tasks),
        "entities_seeded": 0,
        "entities_failed": 0,
        "cli_workers": min(workers, MAX_CLI_SEED_WORKERS),
        "gold_seeded": False,
    }
    if not tasks:
        return detail
    count_lock_workers = min(workers, MAX_CLI_SEED_WORKERS)
    with ThreadPoolExecutor(max_workers=count_lock_workers) as executor:
        futures = {
            executor.submit(vault_remember, bin_path, cli_root, vault, entity, text): entity
            for entity, text in tasks
        }
        for future in as_completed(futures):
            detail["entities_seeded" if future.result() else "entities_failed"] += 1
    return detail


def recall_prediction(
    row: dict[str, Any],
    *,
    bin_path: Path,
    cli_root: Path | None,
    vault: Path,
    limit: int,
) -> tuple[str, dict[str, Any]]:
    """Recall with the question only, then read top entities' fact content."""
    info: dict[str, Any] = {
        "recall_hits": 0,
        "asof_entities_read": 0,
        "pred_chars": 0,
        "query_contains_gold": any(
            str(item).lower() in row["query"].lower() for item in row["gold"]
        ),
    }
    names = vault_recall(bin_path, cli_root, vault, row["query"], limit)
    info["recall_hits"] = len(names)
    parts: list[str] = []
    for name in names[:limit]:
        facts = vault_asof(bin_path, cli_root, vault, name)
        if facts:
            info["asof_entities_read"] += 1
        for fact in facts:
            text = fact.strip()
            if text and text not in parts:
                parts.append(text)
        if sum(len(part) for part in parts) >= adapter.MAX_PRED_CHARS:
            break
    pred = "\n".join(parts)[: adapter.MAX_PRED_CHARS]
    info["pred_chars"] = len(pred)
    return pred, info


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done.add(str(json.loads(line)["case_id"]))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return done


# --------------------------------------------------------------------------- #
# Dry run: prove zero keys / zero model calls / zero network before a real run
# --------------------------------------------------------------------------- #


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    pin = adapter.pin_for(args.task, args.length)
    validation = adapter.validate()
    assert adapter.JUDGE_FREE and adapter.API_KEYS_REQUIRED is False
    plan = {
        "dry_run": True,
        "schema": SCHEMA,
        "version": VERSION,
        "observed_at": utc_now(),
        "official": False,
        "diagnostic": "derived",
        "comparability_tier": "C",
        "score_asserted": False,
        "task": args.task,
        "length": args.length,
        "pin_verified": bool(pin["verified"]),
        "pin_note": pin["note"],
        "max_cases_requested": args.max_cases,
        "cases_planned_if_dataset_available": args.max_cases or "all verified rows (500 for 4K)",
        "cli_seed_workers_planned": min(args.workers, MAX_CLI_SEED_WORKERS),
        "recall_limit_planned": args.recall_limit,
        "network_calls": 0,
        "cli_calls": 0,
        "dataset_bytes_downloaded": 0,
        "keys_loaded": 0,
        "api_keys_required": False,
        "model_calls": 0,
        "judge": None,
        "judge_free": True,
        "pins_checked": validation["pins_checked"],
        "verified_pins": validation["verified_pins"],
        "checkpoint_path": str(
            Path(args.checkpoint_dir) / f"ruler_{args.task}_{args.length}_checkpoint.jsonl"
        ),
        "result_path": str(
            Path(args.results) / f"ruler_{args.task}_{args.length}_max{args.max_cases or 'all'}.json"
        ),
        "note": (
            "Dry run only: no dataset download, no CLI call, no key, no judge. "
            "No score, win or rank is claimed."
        ),
    }
    if not pin["verified"]:
        plan["blocked_reason"] = "unverified pin: " + pin["note"]
    return plan


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="niah_single_1", help="RULER task (default niah_single_1)")
    parser.add_argument(
        "--length", default="4k", help="RULER context length label (default 4k)"
    )
    parser.add_argument(
        "--max-cases", type=int, default=0, help="cases to run (0 = all rows in the pinned file)"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="CLI seed workers, capped at 8"
    )
    parser.add_argument(
        "--recall-limit", type=int, default=adapter.RECALL_LIMIT, help="vault recall --limit"
    )
    parser.add_argument(
        "--data-dir", default=str(adapter.DEFAULT_DATA_DIR), help="external dataset directory"
    )
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="offline plan: no network, no CLI calls, no keys, no scoring",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    if args.max_cases < 0 or args.workers < 1 or args.recall_limit < 1:
        print("error: --max-cases must be >= 0; --workers/--recall-limit must be >= 1")
        return 2

    if args.dry_run:
        # Deliberately side-effect free: no result file, no checkpoint, no data.
        plan = dry_run(args)
        print(json.dumps(plan, indent=2, sort_keys=True))
        print(
            "dry_run counts: network_calls=0 cli_calls=0 keys_loaded=0 model_calls=0 "
            "judge=none cases_planned="
            + str(plan["cases_planned_if_dataset_available"])
        )
        return 0 if plan["pin_verified"] else 2

    try:
        pin = adapter.pin_for(args.task, args.length)
    except ValueError as exc:
        print(f"blocker: {exc}")
        return 2
    if not pin["verified"]:
        print(f"blocker: unverified pin; {pin['note']}")
        return 2
    try:
        dataset_path = adapter.prepare(args.task, args.length, args.data_dir)
        rows = adapter.build_rows(dataset_path, task=args.task, length=args.length, max_cases=args.max_cases or None)
    except Exception as exc:
        print(f"blocker: dataset unavailable ({type(exc).__name__}: {exc})")
        return 2
    if not rows:
        print("blocker: pinned dataset produced zero parseable rows; no score asserted")
        return 2

    bin_path, cli_root = resolve_cli()
    if bin_path is None:
        print("blocker: mn binary not found; set MN_BIN to a public CLI build")
        return 2

    checkpoint_path = (
        Path(args.checkpoint_dir) / f"ruler_{args.task}_{args.length}_checkpoint.jsonl"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(checkpoint_path)
    target = [row for row in rows if row["case_id"] not in done]
    print(
        f"dataset: {dataset_path.name} verified sha256={pin['sha256'][:12]}... "
        f"rows_parsed={len(rows)} selected={len(target)} resumed_skipped={len(done)}"
    )

    result_path = (
        Path(args.results)
        / f"ruler_{args.task}_{args.length}_max{args.max_cases or 'all'}.json"
    )
    if not target and result_path.is_file():
        print("resume no-op: all selected cases already checkpointed; prior result preserved")
        return 0

    gold_by_case = {row["case_id"]: row["gold"] for row in target}
    records: list[dict[str, Any]] = []
    seed_detail: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="mn-bench-ruler-") as temporary:
        vault = Path(temporary) / "vault.db"
        if not vault_init(bin_path, cli_root, vault) and vault.exists():
            print("note: vault init returned an error for an existing path; continuing")
        seed_detail = seed_rows(
            target,
            bin_path=bin_path,
            cli_root=cli_root,
            vault=vault,
            workers=args.workers,
        )
        print(
            "seed: entities_planned={entities_planned} entities_seeded={entities_seeded} "
            "entities_failed={entities_failed} cli_workers={cli_workers}".format(**seed_detail)
        )
        if seed_detail["entities_seeded"] == 0:
            print("blocker: vault seed produced no facts; no score asserted")
            return 2

        with ThreadPoolExecutor(max_workers=min(args.workers, MAX_CLI_SEED_WORKERS)) as executor:
            futures = {
                executor.submit(
                    recall_prediction,
                    row,
                    bin_path=bin_path,
                    cli_root=cli_root,
                    vault=vault,
                    limit=args.recall_limit,
                ): row
                for row in target
            }
            with checkpoint_path.open("a", encoding="utf-8") as checkpoint_file:
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        pred, info = future.result()
                        error = None
                    except Exception as exc:  # never fabricate a verdict on error
                        pred, info, error = "", {}, f"case_error_{type(exc).__name__}"
                    verdict = adapter.score_one(pred, gold_by_case[row["case_id"]])
                    record: dict[str, Any] = {
                        "case_id": row["case_id"],
                        "index": row["index"],
                        "task": args.task,
                        "length": args.length,
                        "query_hash_prefix": sha256_text(row["query"])[:12],
                        "gold_hash_prefix": (
                            sha256_text("\x1f".join(gold_by_case[row["case_id"]]))[:12]
                        ),
                        "pred_hash_prefix": sha256_text(pred)[:12] if pred else None,
                        "pred": pred,  # in-memory only for aggregation; not checkpointed
                        "gold": gold_by_case[row["case_id"]],  # in-memory only
                        "verdict": verdict,
                        "recall": info,
                        "error": error,
                        "observed_at": utc_now(),
                    }
                    records.append(record)
                    checkpoint_file.write(
                        json.dumps(
                            {
                                key: value
                                for key, value in record.items()
                                if key not in {"pred", "gold"}
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    checkpoint_file.flush()

    evaluation = adapter.evaluate(records)
    dataset_manifest = {
        "local_name": dataset_path.name,
        "sha256": pin["sha256"],
        "bytes": pin["bytes"],
        "verified": True,
        "pin_note": pin["note"],
        "url": pin["url"],
    }
    failures = [
        str(record["error"])
        for record in records
        if record["error"] or not record["verdict"]["valid"]
    ]
    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "observed_at": utc_now(),
        "official": False,
        "diagnostic": "derived",
        "comparability_tier": "C",
        "score_asserted": False,
        "judge_free": True,
        "api_keys_required": False,
        "model_calls": 0,
        "keys_loaded": 0,
        "note": NOTE,
        "status": "completed",
        "task": args.task,
        "length": args.length,
        "dataset": dataset_manifest,
        "seed_detail": seed_detail,
        "denominators": {
            "rows_in_file": len(rows),
            "selected": len(target),
            "attempted": len(records),
            "scored": evaluation["scored"],
            "excluded": evaluation["excluded"],
            "resumed_skipped": len(done),
            "failed": len(failures),
        },
        "string_match_all": {
            "correct": evaluation["correct"],
            "denominator": evaluation["scored"],
            "score": evaluation["metrics"]["primary"]["score"],
        },
        "string_match_part": {
            "correct": evaluation["part_correct"],
            "denominator": evaluation["scored"],
            "score": evaluation["metrics"]["string_match_part"]["score"],
        },
        "substring_match_f1": {
            "sum": evaluation["f1_sum"],
            "mean": evaluation["f1_mean"],
            "denominator": evaluation["scored"],
        },
        "evaluation": evaluation,
        "top_failure_reasons": sorted({item: failures.count(item) for item in failures}.items()),
        "workers": min(args.workers, MAX_CLI_SEED_WORKERS),
        "recall_limit": args.recall_limit,
        "wall_time_s": round(time.monotonic() - started, 3),
        "checkpoint_path": str(checkpoint_path),
        "limitations": [
            "Derived harness: vault recall replaces RULER's LLM reader; not an official RULER score.",
            "Lengths are RULER generator target tokens, not vault token counts.",
            "Only the 4K niah_single_1 pin is SHA-verified; 8K-128K pins are refused.",
            "Gold is scored only locally and never seeded, recalled, checkpointed or published.",
            "No judge, no model call, no key; no win, rank or comparison is claimed.",
        ],
    }
    write_json(result_path, result)

    print("ruler: official=false diagnostic=derived score_asserted=false judge_free=true")
    print(
        "denominators: rows_in_file={rows_in_file} selected={selected} attempted={attempted} "
        "scored={scored} excluded={excluded} resumed_skipped={resumed_skipped}".format(
            **result["denominators"]
        )
    )
    print(
        f"string_match_all: {result['string_match_all']['correct']}/"
        f"{result['string_match_all']['denominator']} = {result['string_match_all']['score']}"
    )
    print(
        f"substring_match_f1: sum={result['substring_match_f1']['sum']} "
        f"mean={result['substring_match_f1']['mean']}"
    )
    print(f"keys_loaded=0 model_calls=0 judge=none api_keys_required=false")
    print(f"checkpoint: {checkpoint_path}")
    print(f"result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
