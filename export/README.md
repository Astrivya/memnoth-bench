# Sanitized export fence

`export.py` is the **only** way figures from `results/summary/*.json` reach
published surfaces. It reads those files, rebuilds each figure field-by-field
from an allow-list, and renders one sanitized JSON document plus the Markdown
summary generated from it. There is no path that copies a figure verbatim out of
a result file.

## The fence

Every figure must carry all of the following, or the whole run is refused:

| Field | Rule |
|---|---|
| `id`, `lane`, `label` | non-empty, stable identifiers |
| `value` (+ optional `unit`) | number or non-empty string |
| `denominator` | integer `n >= 1`, optionally `{ "n": N, "unit": "cases" }` |
| `ci` | `{ "low": ..., "high": ..., "level": "95%" }`, or `null` / `"no CI"` |
| `dataset_version` | exact pinned revision; `UNPINNED`/placeholder is rejected |
| `judge_version` or `scorer_version` | exact pinned version; placeholder is rejected |
| `provenance` | source manifest / run reference (public paths only) |
| `metric_definition` | what the number actually measures |
| `observed_date` | ISO `YYYY-MM-DD` |
| `official` | the literal boolean `false` |
| `note` | free-form context; a standing "NOT an official score" notice is added |

A top-level object may carry shared fields (`dataset_version`, `judge_version`,
`observed_date`, `official`, `note`, …) with a `figures` list beneath it; those
shared fields are inherited by each figure. A bare figure object and a bare list
of figures are also accepted.

The generator additionally refuses any source string containing ranking or
superiority language (`#1`, `best`, `top`, …), any private repository or crate
name, and any figure with a duplicate id. It re-scans its own generated output
with the same rules before writing or comparing. No output contains a
`generated_at` timestamp or any other drifting value, so `--check` is stable.

## Why there are no hand-copied figures

A bare percentage with no denominator is not a result. Copying a number by hand
drops the run's sample size, interval, pins, judge, and date — the exact context
that makes an informational smoke slice honest. It also silently allows a
private path or an ordinal ranking claim to reach a public page. Keeping the
render loop in one generator means the docs and the web fragment cannot disagree
with the source manifests, and `--check` turns any drift into a failing command.

Both output files carry a standing notice: **NOT an official score**. Never
remove it, and never re-type a figure beside the generated file.

## Exact commands

Run from this repo root (`.../memnoth-bench`):

```bash
# Default: verify the docs file matches results/summary/*.json. Writes nothing.
# Exit 0 = match (or an honest "no results yet"); exit 1 = drift or invalid figure.
python export/export.py --check

# Regenerate the docs summary. This is the ONLY writing mode.
python export/export.py --write

# Also generate/check the optional web proof-board fragment.
python export/export.py --write --with-web-fragment
python export/export.py --check --with-web-fragment
```

Outputs (paths are derived from this file's location, not hardcoded):

- workspace docs: `docs/operations/benchmark-external-results.md`
- web fragment (only with `--with-web-fragment`):
  `web/src/content/proof/external-boards.json`

When `results/summary/` has no JSON yet, both modes print an honest
`no results yet` message and write nothing; `--check` fails only if a stale
non-empty output file is present with no source results to justify it.
