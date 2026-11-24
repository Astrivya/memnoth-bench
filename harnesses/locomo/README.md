# LoCoMo — pinned harness adapter

**Informational diagnostic only. Not an official benchmark score.**

## Source and pins

- Source: `https://github.com/microsoft/LoCoMo`
- Upstream commit: `PIN_BEFORE_USE` — fill with the exact commit the conversation artifact came from.
- Dataset URL: `PIN_BEFORE_USE` in `adapter.py` (`DATASET_URL`).
- Dataset SHA-256: `PIN_BEFORE_USE` in `adapter.py` (`DATASET_SHA256`). Compute it from the exact downloaded artifact; **never invent a hash**.
- Judge pin (optional lane): `gpt-4o-2024-08-06`, temperature `0` (`JUDGE`). Records the pin only; `enforced_by_adapter: false`.

`prepare()` refuses to download until the `PIN_BEFORE_USE` values are replaced. Downloads land only in gitignored `data/`, are hashed in a temp file, and are moved into place only after SHA-256 verification. Dataset bytes are never vendored.

## Slice

One conversation, five questions from `fixtures/pinned-ids.json`. This is a **locally constructed slice, not an official LoCoMo split** (`official_split: false`).

## Metrics

- **Token-level F1** (SQuAD-style): deterministic, judge-free, reported as a sum and mean over an explicit denominator, plus an exact-token-match count.
- **Judge accuracy**: only when a `judge_fn` is injected; kept separate from F1.
- **Tokens / latency / context tokens**: descriptive statistics with `n` and missing counts, reported with the context note that they depend on hardware, provider, context length and caching.

Every result carries `official: false`, `diagnostic: derived`, an `observed_at` UTC stamp, denominators, and no bare percentages or "#1" claim.

## Exact smoke command

From the repository root, no keys, no network, no spend:

```bash
python -m py_compile harnesses/__init__.py harnesses/common.py harnesses/locomo/__init__.py harnesses/locomo/adapter.py
python -m harnesses.locomo.adapter --smoke
```

## License note

The LoCoMo dataset and code are **research-only, third-party material**. Memnoth's Apache-2.0 license does **not** grant redistribution rights to upstream data. Never commit dataset bytes or upstream checkouts; commit only pins, IDs, and derived diagnostics. Verify the upstream license before any internal use and honor its terms.

## What the metric does NOT prove

- Not an official LoCoMo result; five questions and one conversation are a local diagnostic.
- Token F1 rewards lexical overlap, not semantic correctness or factual accuracy.
- A single conversation cannot characterize long-horizon memory across many users or sessions.
- Token counts and latency are not a speed or efficiency claim and are not comparable across hardware/providers.
- Judge accuracy depends on an unverified injected callable; it is not an upstream score.
- No "#1" or "best" claim follows from this lane.
