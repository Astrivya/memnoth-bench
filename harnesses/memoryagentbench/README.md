# MemoryAgentBench — pinned harness adapter (FactConsolidation)

**Informational diagnostic only. Not an official benchmark score. This is our highest-ROI lane.**

## Source and pins

- Source: `https://github.com/Yukun-Cui/MemoryAgentBench`
- MQUAKE true+rewrite pairs: `https://github.com/stanfordnlp/MQUAKE`
- Upstream commit: `PIN_BEFORE_USE` (`UPSTREAM_COMMIT`).
- MQUAKE commit: `PIN_BEFORE_USE` (`MQUAKE_COMMIT`).
- Dataset URL: `PIN_BEFORE_USE` in `adapter.py` (`DATASET_URL`).
- Dataset SHA-256: `PIN_BEFORE_USE` in `adapter.py` (`DATASET_SHA256`). Compute it from the exact downloaded artifact; **never invent a hash**.
- Judge pin (optional, semantic verdict only): `gpt-4o-2024-08-06`, temperature `0` (`JUDGE`). Records the pin only; `enforced_by_adapter: false`.

`prepare()` refuses to download until the `PIN_BEFORE_USE` values are replaced. Downloads land only in gitignored `data/`, are hashed in a temp file, and are moved into place only after SHA-256 verification. Dataset bytes are never vendored.

## Slice

FactConsolidation: SH 24 + MH 16 cases at 4096 and 8192 context tokens, built from MQUAKE true + rewrite pairs, from `fixtures/pinned-ids.json`. This is a **locally constructed slice, not an official split** (`official_split: false`). Planned fixture counts are echoed with `authoritative: false`; the observed valid denominator is always authoritative and is never inflated to match the plan.

## Metrics

- **Primary — structured serial-number conflict resolution (exact match).** The gold serial is the serial of the newer MQUAKE rewrite fact; returning the superseded true-fact serial, a paraphrase, or prose fails. Strict `str().strip()` equality, no fuzzy matching. Reported overall and by subset, by context token, and by subset × context, each with its own denominator.
- **Secondary — plain answer exact match** after whitespace/case normalization. Reported second, with the same groupings.
- **Optional judge** — only when a `judge_fn` is injected, kept separate from both exact-match metrics.

Invalid rows are counted (`invalid_rows`) and excluded from denominators rather than silently dropped. Every result carries `official: false`, `diagnostic: derived`, an `observed_at` UTC stamp, denominators, and no bare percentages or "#1" claim.

## Exact smoke command

From the repository root, no keys, no network, no spend:

```bash
python -m py_compile harnesses/__init__.py harnesses/common.py harnesses/memoryagentbench/__init__.py harnesses/memoryagentbench/adapter.py
python -m harnesses.memoryagentbench.adapter --smoke
```

## License note

MemoryAgentBench and MQUAKE datasets and code are **research-only, third-party material**. Memnoth's Apache-2.0 license does **not** grant redistribution rights to upstream data. Never commit dataset bytes or upstream checkouts; commit only pins, IDs, and derived diagnostics. Verify the upstream licenses before any internal use and honor their terms.

## What the metric does NOT prove

- Not an official MemoryAgentBench result; the slice, counts and denominators are local.
- Serial exact match measures the structured field only — not a natural-language answer or real user behavior.
- The adapter cannot verify offline that `gold_serial` truly came from the pinned MQUAKE true+rewrite pair; that depends on `prepare()` using the real pinned artifact.
- 4096 vs 8192 token rows are descriptive, not a causal context-length claim.
- A canonical-match diagnostic exists but is never substituted for the strict primary metric.
- Judge accuracy depends on an unverified injected callable and is never promoted over the primary metric.
- Passing this lane does not prove conflict resolution across arbitrary schemas, tenants, or long histories. No "#1" or "best" claim follows.
