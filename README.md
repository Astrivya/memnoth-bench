# memnoth-bench

Public, reproducible external benchmarks for the Memnoth CLI and MCP server.

**Informational only — NOT an official benchmark score.** Every result carries a
manifest with pinned versions, denominators, judge metadata, and a run date.
No "#1" or "best" claim is made from these lanes without a paired test on the
same pinned harness.

This repo calls the **public** `mn` binary (`--json`) and the MCP stdio surface.
It never imports private crates and never sees a private path. The vendored
control-plane repository stays private; nothing from it belongs here.

## Repo layout

- `targets/cli.lock` — CLI commit + version this harness was validated against
- `adapters/` — public-surface adapters (`mn --json`, MCP stdio, pinned judge)
- `harnesses/` — pinned adapters for LongMemEval, LoCoMo, MemoryAgentBench
- `fixtures/` — pinned question IDs / small slices (committed, small)
- `runner/` — one command: `bench run --lane ... --confirm-spend`
- `cache/` — GITIGNORED; judge + dataset caches keyed by pins
- `results/raw/` — GITIGNORED; per-item responses, uploaded as CI artifacts
- `results/summary/` — committed, sanitized, with denominators / CI / date
- `export/` — generator writing sanitized figures to root `docs/` and the web proof boards, with `--check`

## Quick start

```bash
# Model-free smoke only — no keys, no spend.
python runner/bench.py smoke

# A pinned external lane (requires a key by NAME in env, never a value in code).
MEMNOTH_BENCH=1 python runner/bench.py run --lane memoryagentbench-mh --confirm-spend
```

## Rules

1. Keys are read from env by name. Only `.env.example` is tracked. Never print a value.
2. Datasets are downloaded at runtime from pinned URLs/revisions and verified by SHA-256. Never vendor them.
3. A manifest is written **before** any result file.
4. A failed or un-runnable lane is recorded as a limitation, never as a score.
5. Every external failure that reproduces is promoted to a fast local fixture in `CLI/`.
6. Judge responses are cached by `(question hash, prompt hash, judge version, answer hash)`.
7. Policy, docs map, and definition of done live in the workspace root `docs/`. This README defers to them.

## License

Apache-2.0, matching the public CLI. See `LICENSE`.
