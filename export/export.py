#!/usr/bin/env python3
"""Sanitized export fence for external benchmark results.

Reads ``results/summary/*.json`` from this public repo and renders ONE sanitized
JSON document plus its Markdown representation. It writes to the workspace docs
path and, optionally, a web proof-board fragment.

The default mode is ``--check``: nothing is written, and the generator compares
the desired bytes with the files on disk. ``--write`` is the only mode that
creates or overwrites anything. There is no mode that copies a figure verbatim
out of a result file: every figure is rebuilt field-by-field from an allow-list,
and a figure missing any required provenance field is a hard error.

Exit codes:
    0  check matches (or there are honestly no results yet), write succeeded
    1  drift, invalid/unsafe figure, or missing expected output in check mode
    2  bad command line (argparse)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterator

SCHEMA = "memnoth-external-bench-export/1"
NOTICE = "NOT an official score"
CANONICAL_NOTE = (
    "NOT an official score — informational, derived smoke-slice result only; "
    "do not quote it as an official benchmark result."
)

# Ranking language is never emitted. These are rejected in the INPUT, including
# in free-form note/provenance strings, before anything reaches an output file.
BANNED = re.compile(
    r"(?:#\s*0*1\b|\bbest\b|\btop\b|\bnumber\s+(?:one|1)\b|\bfirst\s+place\b)",
    re.IGNORECASE,
)
# A public export must not carry a private repo path or private crate name.
PRIVATE = re.compile(
    r"(?:memnoth[-_]cloud|memnoth-private|astrivya/private|[/\\]private[/\\]|e:[/\\]memnoth-prod)",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(r"unpinned|placeholder|\bTBD\b|\bTODO\b", re.IGNORECASE)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class FenceError(Exception):
    """A figure or file failed the sanitization fence."""


def iter_strings(value: Any, where: str = "$") -> Iterator[tuple[str, str]]:
    """Yield (json path, string) for every string anywhere in a JSON value."""
    if isinstance(value, str):
        yield where, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_strings(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_strings(item, f"{where}[{index}]")


def scan_text(raw: Any, where: str) -> list[str]:
    """Reject ranking language and private paths anywhere in a source file."""
    errors: list[str] = []
    for path, text in iter_strings(raw, where):
        if BANNED.search(text):
            errors.append(f"{path}: ranking/superiority language is not allowed")
        if PRIVATE.search(text):
            errors.append(f"{path}: private path or private crate name is not allowed")
    return errors


def nonempty_string(fig: dict[str, Any], key: str, where: str, errors: list[str]) -> str:
    value = fig.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where}: missing required string field '{key}'")
        return ""
    return value.strip()


def parse_denominator(value: Any, where: str, errors: list[str]) -> dict[str, Any] | None:
    if isinstance(value, bool):
        errors.append(f"{where}: denominator must be an integer or an object with 'n'")
        return None
    if isinstance(value, int):
        if value < 1:
            errors.append(f"{where}: denominator.n must be >= 1")
            return None
        return {"n": value, "unit": "items"}
    if isinstance(value, dict):
        n = value.get("n")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            errors.append(f"{where}: denominator.n must be an integer >= 1")
            return None
        unit = value.get("unit", "items")
        if not isinstance(unit, str) or not unit.strip():
            errors.append(f"{where}: denominator.unit must be a non-empty string")
            return None
        return {"n": n, "unit": unit.strip()}
    errors.append(f"{where}: denominator must be an integer or an object with 'n'")
    return None


def parse_ci(value: Any, present: bool, where: str, errors: list[str]) -> Any:
    if not present:
        errors.append(f"{where}: missing required field 'ci' (use an interval or \"no CI\")")
        return None
    if value is None:
        return "no CI"
    if isinstance(value, str):
        if value.strip().lower() in {"no ci", "none"}:
            return "no CI"
        errors.append(f"{where}: ci string must be \"no CI\" (or an interval object)")
        return None
    if isinstance(value, dict):
        low, high = value.get("low"), value.get("high")
        if isinstance(low, bool) or isinstance(high, bool) or not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            errors.append(f"{where}: ci.low and ci.high must be numbers")
            return None
        if low > high:
            errors.append(f"{where}: ci.low must be <= ci.high")
            return None
        level = value.get("level", "95%")
        if not isinstance(level, str) or not level.strip():
            errors.append(f"{where}: ci.level must be a non-empty string")
            return None
        return {"low": low, "high": high, "level": level.strip()}
    errors.append(f"{where}: ci must be null/\"no CI\" or an object with low and high")
    return None


def parse_date(value: str, where: str, errors: list[str]) -> str:
    normalized = value.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}T", normalized):
        normalized = normalized[:10]
    if not DATE_RE.match(normalized):
        errors.append(f"{where}: observed_date must be an ISO date (YYYY-MM-DD)")
        return ""
    try:
        date.fromisoformat(normalized)
    except ValueError:
        errors.append(f"{where}: observed_date is not a real calendar date")
        return ""
    return normalized


def sanitize_figure(raw: Any, where: str, errors: list[str]) -> dict[str, Any] | None:
    start = len(errors)
    if not isinstance(raw, dict):
        errors.append(f"{where}: each figure must be a JSON object")
        return None

    fig: dict[str, Any] = raw
    figure_id = nonempty_string(fig, "id", where, errors)
    if figure_id and not ID_RE.match(figure_id):
        errors.append(f"{where}: id may only contain letters, digits, dot, dash, underscore")
    lane = nonempty_string(fig, "lane", where, errors)
    label = nonempty_string(fig, "label", where, errors)

    value = fig.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float, str)) or (isinstance(value, str) and not value.strip()):
        errors.append(f"{where}: value must be a number or a non-empty string")
        value = None
    unit = fig.get("unit", "")
    if not isinstance(unit, str):
        errors.append(f"{where}: unit must be a string")
        unit = ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        display = f"{value}{unit}"
    elif isinstance(value, str):
        display = value.strip()
    else:
        display = "not available"

    denominator = parse_denominator(fig.get("denominator"), where, errors)
    ci = parse_ci(fig.get("ci"), "ci" in fig, where, errors)

    dataset_version = nonempty_string(fig, "dataset_version", where, errors)
    if dataset_version and PLACEHOLDER.search(dataset_version):
        errors.append(f"{where}: dataset_version is an unpinned placeholder")

    judge_version = fig.get("judge_version")
    scorer_version = fig.get("scorer_version")
    if isinstance(judge_version, str) and judge_version.strip() and not PLACEHOLDER.search(judge_version):
        version, kind = judge_version.strip(), "judge"
    elif isinstance(scorer_version, str) and scorer_version.strip() and not PLACEHOLDER.search(scorer_version):
        version, kind = scorer_version.strip(), "scorer"
    else:
        errors.append(f"{where}: missing pinned judge_version or scorer_version")
        version, kind = "", ""

    provenance = nonempty_string(fig, "provenance", where, errors)
    metric_definition = nonempty_string(fig, "metric_definition", where, errors)
    observed_date = ""
    raw_date = fig.get("observed_date")
    if isinstance(raw_date, str) and raw_date.strip():
        observed_date = parse_date(raw_date, where, errors)
    else:
        errors.append(f"{where}: missing required field 'observed_date'")

    if "official" not in fig or fig.get("official") is not False or not isinstance(fig.get("official"), bool):
        errors.append(f"{where}: 'official' must be the literal boolean false")

    note = nonempty_string(fig, "note", where, errors)
    full_note = f"{CANONICAL_NOTE} {note}".strip() if note else CANONICAL_NOTE

    if len(errors) > start:
        return None

    return {
        "id": figure_id,
        "lane": lane,
        "label": label,
        "value": value,
        "value_display": display,
        "denominator": denominator,
        "confidence_interval": ci,
        "dataset_version": dataset_version,
        "judge_or_scorer_version": version,
        "version_kind": kind,
        "provenance": provenance,
        "metric_definition": metric_definition,
        "observed_date": observed_date,
        "official": False,
        "note": full_note,
    }


def load_figures(results_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    figures: list[dict[str, Any]] = []
    files = sorted(results_dir.glob("*.json"))
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: cannot read JSON: {exc}")
            continue
        errors.extend(scan_text(raw, path.name))

        if isinstance(raw, list):
            records = raw
            defaults: dict[str, Any] = {}
        elif isinstance(raw, dict) and isinstance(raw.get("figures"), list):
            records = raw["figures"]
            defaults = {k: v for k, v in raw.items() if k != "figures"}
        elif isinstance(raw, dict):
            records = [raw]
            defaults = {}
        else:
            errors.append(f"{path.name}: expected an object, a figure list, or an object with 'figures'")
            continue

        local_errors: list[str] = []
        for index, record in enumerate(records):
            merged = {**defaults, **record} if isinstance(record, dict) else record
            figure = sanitize_figure(merged, f"{path.name}#{index}", local_errors)
            if figure is not None:
                figures.append(figure)
        # Keep errors scoped to figures actually parsed from this file.
        errors.extend(local_errors)

    seen: set[str] = set()
    for figure in figures:
        if figure["id"] in seen:
            errors.append(f"duplicate figure id '{figure['id']}'")
        seen.add(figure["id"])

    figures.sort(key=lambda item: item["id"])
    return figures, errors


def build_document(figures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "classification": "informational only — derived smoke slices",
        "official": False,
        "note": f"{NOTICE}. Every figure below is a non-official, informational observation.",
        "source": "memnoth-bench results/summary/*.json (public repo; no private paths)",
        "figure_count": len(figures),
        "figures": figures,
    }


def cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(document: dict[str, Any]) -> str:
    lines = [
        "# External benchmark results — informational only",
        "",
        f"> **{NOTICE}.** These are derived smoke-slice observations from the public",
        "> `memnoth-bench` repo. They are not official benchmark scores and must not",
        "> be quoted as such.",
        "",
        "Generated by `memnoth-bench/export/export.py --write`. Do not edit by hand;",
        "run `python export/export.py --check` to verify this file matches the",
        "sanitized source figures. Every figure carries a denominator, a confidence",
        "interval (or an explicit \"no CI\"), a pinned dataset version, a pinned",
        "judge/scorer version, provenance, a metric definition, an observed date,",
        "`official: false`, and the notice above.",
        "",
        "## Figures",
        "",
        "| ID | Lane | Observed | Value | Denominator | CI | Dataset version | Judge/scorer version | Metric definition | Provenance |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for fig in document["figures"]:
        den = fig["denominator"]
        den_text = f"{den['n']} {den['unit']}" if den else "missing"
        ci = fig["confidence_interval"]
        ci_text = ci if isinstance(ci, str) else f"[{ci['low']}, {ci['high']}] {ci['level']}"
        lines.append(
            "| " + " | ".join(
                cell(part)
                for part in (
                    fig["id"],
                    fig["lane"],
                    fig["observed_date"],
                    fig["value_display"],
                    den_text,
                    ci_text,
                    fig["dataset_version"],
                    f"{fig['judge_or_scorer_version']} ({fig['version_kind']})",
                    fig["metric_definition"],
                    fig["provenance"],
                )
            ) + " |"
        )
    lines += [
        "",
        "Each figure's standing note: " + CANONICAL_NOTE,
        "",
        "## Sanitized JSON",
        "",
        "This is the single sanitized JSON document the table above is rendered from.",
        "",
        "```json",
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
    ]
    return "\n".join(lines)


def build_web_fragment(document: dict[str, Any]) -> str:
    figures = document["figures"]
    fragment: dict[str, Any] = {
        "kind": "external",
        "schema": SCHEMA,
        "id": "external-benchmark-lanes",
        "title": "External benchmark lanes — informational only",
        "official": False,
        "note": f"{NOTICE}. Derived smoke slices; not official benchmark scores.",
        "source": "memnoth-bench results/summary/*.json",
        "field_legend": {
            "w": "Display width, fixed at 0% here: informational rows are never normalised, ranked, or compared.",
            "v": "The observed value exactly as sanitized; quote only with its denominator, CI, pins and date.",
            "ours": "true on our own informational run's row.",
            "source": "Provenance from the source manifest (public paths only).",
            "date": "The date the observation was made.",
            "scope": "Always 'subset': these are locally constructed smoke slices, not official splits.",
        },
        "figures": figures,
        "boards": [],
    }
    for fig in figures:
        den = fig["denominator"]
        denominator_text = f"{den['n']} {den['unit']}" if den else "unknown"
        fragment["boards"].append(
            {
                "id": fig["id"],
                "name": fig["label"],
                "sub": f"{denominator_text} · {fig['observed_date']}",
                "detail": denominator_text,
                "delta": "",
                "ours": fig["value_display"],
                "protocol": fig["metric_definition"],
                "note": fig["note"],
                "official": False,
                "rows": [
                    {
                        "name": "Memnoth (informational external run)",
                        "qual": fig["lane"],
                        "w": "0%",
                        "v": fig["value_display"],
                        "ours": True,
                        "source": fig["provenance"],
                        "date": fig["observed_date"],
                        "scope": "subset",
                        "official": False,
                        "denominator": den,
                        "confidence_interval": fig["confidence_interval"],
                        "dataset_version": fig["dataset_version"],
                        "judge_or_scorer_version": fig["judge_or_scorer_version"],
                        "version_kind": fig["version_kind"],
                        "metric_definition": fig["metric_definition"],
                        "observed_date": fig["observed_date"],
                        "note": fig["note"],
                    }
                ],
            }
        )
    text = json.dumps(fragment, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if BANNED.search(text) or PRIVATE.search(text):
        raise FenceError("generated web fragment contains disallowed language or a private path")
    return text


def check_target(path: Path, expected: str, label: str) -> bool:
    if not path.is_file():
        print(f"DRIFT {label}: missing expected file {path}", file=sys.stderr)
        return False
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        print(f"DRIFT {label}: {path} does not match sanitized sources", file=sys.stderr)
        return False
    print(f"OK {label}: {path}")
    return True


def main(argv: list[str] | None = None) -> int:
    script = Path(__file__).resolve()
    repo_root = script.parents[1]
    workspace_root = repo_root.parent

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify outputs match (default)")
    mode.add_argument("--write", action="store_true", help="write outputs (the only writing mode)")
    parser.add_argument(
        "--with-web-fragment",
        action="store_true",
        help="also generate/check web/src/content/proof/external-boards.json",
    )
    parser.add_argument("--results-dir", type=Path, default=repo_root / "results" / "summary")
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=workspace_root / "docs" / "operations" / "benchmark-external-results.md",
    )
    parser.add_argument(
        "--web-out",
        type=Path,
        default=workspace_root / "web" / "src" / "content" / "proof" / "external-boards.json",
    )
    args = parser.parse_args(argv)

    write = args.write
    figures, errors = load_figures(args.results_dir)
    if errors:
        print("export fence: refusing to emit — invalid or unsafe figures:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    if not figures:
        print(
            "export fence: no results yet — "
            f"{args.results_dir} contains no sanitized figure JSON; nothing fabricated."
        )
        stale = [
            path
            for path, enabled in ((args.docs_out, True), (args.web_out, args.with_web_fragment))
            if enabled and path.is_file() and path.stat().st_size > 0
        ]
        if stale:
            for path in stale:
                print(f"DRIFT: stale output with no source results: {path}", file=sys.stderr)
            return 1
        return 0

    document = build_document(figures)
    markdown = render_markdown(document)
    if BANNED.search(markdown) or PRIVATE.search(markdown):
        print("export fence: generated Markdown contains disallowed language or a private path", file=sys.stderr)
        return 1
    web_text = build_web_fragment(document) if args.with_web_fragment else None

    if write:
        args.docs_out.parent.mkdir(parents=True, exist_ok=True)
        args.docs_out.write_text(markdown, encoding="utf-8")
        print(f"WROTE docs: {args.docs_out}")
        if web_text is not None:
            args.web_out.parent.mkdir(parents=True, exist_ok=True)
            args.web_out.write_text(web_text, encoding="utf-8")
            print(f"WROTE web fragment: {args.web_out}")
        return 0

    ok = check_target(args.docs_out, markdown, "docs")
    if web_text is not None:
        ok = check_target(args.web_out, web_text, "web fragment") and ok
    if not ok:
        print("export fence: drift detected; run with --write to regenerate", file=sys.stderr)
        return 1
    print(f"export fence: {len(figures)} figure(s) verified, all non-official")
    return 0


if __name__ == "__main__":
    sys.exit(main())
