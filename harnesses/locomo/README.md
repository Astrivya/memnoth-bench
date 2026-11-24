# LoCoMo — pinned harness adapter

**Informational diagnostic only. Not an official benchmark score.**

## Source and pins

- Official source (dataset and code): `https://github.com/snap-research/locomo`
- Upstream commit (pinned): `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`, obtained from
  `gh api repos/snap-research/locomo/commits/main --jq .sha`.
- Upstream artifact path: `data/locomo10.json` (the official 10-conversation release;
  the download was checked to contain 10 samples).
- Pinned artifact URL (`DATASET_URL` in `adapter.py`):
  `https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json`
- Artifact SHA-256 (`DATASET_SHA256` in `adapter.py`):
  `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
  Computed from one real download of the exact pinned URL to
  `D:/temp/opencode/locomo10.json` (outside the repository) with
  `Get-FileHash -Algorithm SHA256 D:/temp/opencode/locomo10.json`. The artifact is
  **not** vendored.
- Local path after verification: `data/locomo/locomo_conversation.json` (gitignored).
- Judge pin (optional lane): `gpt-4o-2024-08-06`, temperature `0` (`JUDGE`). Records the pin only; `enforced_by_adapter: false`.

`prepare()` uses only the verified pins above. Downloads land only in gitignored
`data/`, are hashed in a temp file, and are moved into place only after SHA-256
verification. Dataset bytes are never committed.

## Upstream evaluation commands and published metric

Confirmed from the upstream `README.MD` and scripts at the pinned commit:

- `bash scripts/generate_observations.sh` regenerates RAG observations;
  `bash scripts/generate_session_summaries.sh` regenerates session summaries.
- QA evaluation entry points are `bash scripts/evaluate_gpts.sh`,
  `bash scripts/evaluate_claude.sh`, `bash scripts/evaluate_gemini.sh`,
  `bash scripts/evaluate_hf_llm.sh` and `bash scripts/evaluate_rag_gpts.sh`;
  they call `python3 task_eval/evaluate_qa.py`.
- The published QA metric is **token-level F1** (SQuAD-style with normalization and
  Porter stemming: `f1_score` / `f1` in `task_eval/evaluation.py`), reported per
  category and overall by `task_eval/evaluation_stats.py` (the `*_f1` keys). Upstream
  also computes exact match, BERTScore, ROUGE-L and evidence recall. Its evaluation
  scripts do **not** use an LLM judge.

This lane's F1 is deterministic and judge-free, but it is not the upstream scorer,
and this one-conversation / five-question slice is not the upstream ten-conversation
evaluation. The numbers are not comparable to published LoCoMo results.

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

The LoCoMo dataset and code are **third-party, research-only material**. The
upstream `LICENSE.txt` at the pinned commit is **Creative Commons Attribution-
NonCommercial 4.0 International (CC BY-NC 4.0)** — non-commercial use, with
attribution to the LoCoMo authors and their ACL 2024 paper required. Memnoth's
Apache-2.0 license does **not** grant redistribution rights to upstream data.
Never commit dataset bytes or upstream checkouts; commit only pins, IDs, and
derived diagnostics. Honor the upstream license terms before any internal use.

## What the metric does NOT prove

- Not an official LoCoMo result; five questions and one conversation are a local diagnostic, and this adapter is not the upstream scorer.
- Token F1 rewards lexical overlap, not semantic correctness or factual accuracy.
- A single conversation cannot characterize long-horizon memory across many users or sessions.
- Token counts and latency are not a speed or efficiency claim and are not comparable across hardware/providers.
- Judge accuracy depends on an unverified injected callable; it is not an upstream score.
- No "#1" or "best" claim follows from this lane.
