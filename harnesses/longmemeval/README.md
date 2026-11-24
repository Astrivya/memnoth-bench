# LongMemEval — pinned harness adapter

**Informational diagnostic only. Not an official benchmark score.**

## Source and pins

- Source: `https://github.com/xiaowu0162/LongMemEval`
- Paper: `https://arxiv.org/abs/2410.10813`
- Upstream commit: `PIN_BEFORE_USE` — fill with the exact commit the dataset artifact was built from.
- Dataset URL: `PIN_BEFORE_USE` in `adapter.py` (`DATASET_URL`).
- Dataset SHA-256: `PIN_BEFORE_USE` in `adapter.py` (`DATASET_SHA256`). Compute it from the exact downloaded artifact; **never invent a hash**.
- Judge pin: `gpt-4o-2024-08-06`, temperature `0` (`JUDGE` in `adapter.py`). The adapter records this pin but cannot verify the injected `judge_fn` actually called it (`enforced_by_adapter: false`).

`prepare()` refuses to download until the `PIN_BEFORE_USE` values are replaced. Downloads go only to the gitignored `data/` directory, are hashed in a temp file, and are moved into place only after the SHA-256 matches. Dataset bytes are never vendored.

## Slice

Oracle variant, `question_type == temporal-reasoning`, 30 questions from `fixtures/pinned-ids.json`. This is a **locally constructed slice, not an official LongMemEval split** (`official_split: false`).

## Metrics

- **Recall@k / NDCG@k** with `k in (1, 3, 5, 10)`: computed directly from the `answer_session_ids` labels against the ranked `retrieved_session_ids`. Judge-free and model-free.
- **Judge accuracy**: computed only when a `judge_fn` is injected; indeterminate verdicts are excluded from its denominator.

Every result carries `official: false`, `diagnostic: derived`, an `observed_at` UTC stamp, counts with denominators, and no bare percentages or "#1" claim.

## Exact smoke command

From the repository root, no keys, no network, no spend:

```bash
python -m py_compile harnesses/__init__.py harnesses/common.py harnesses/longmemeval/__init__.py harnesses/longmemeval/adapter.py
python -m harnesses.longmemeval.adapter --smoke
```

## License note

The LongMemEval dataset and code are **research-only, third-party material**. Memnoth's Apache-2.0 license does **not** grant redistribution rights to upstream data. Never commit dataset bytes or upstream harness checkouts; commit only pins, IDs, and derived diagnostics. Verify the upstream license before any internal use and honor its terms.

## What the metric does NOT prove

- Not an official LongMemEval result; the slice and denominator are local.
- Recall@k / NDCG@k measure retrieved session IDs only — not answer correctness.
- Label quality is inherited from upstream; a wrong or ambiguous `answer_session_ids` label propagates.
- Judge accuracy reflects whatever `judge_fn` actually calls, which the adapter cannot verify.
- No comparison to published numbers is valid unless data, labels, judge version and prompt all match.
- A better number here does not prove the vault is correct on real user histories.
