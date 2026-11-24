# LongMemEval — pinned harness adapter

**Informational diagnostic only. Not an official benchmark score.**

## Source and pins

All pins below were verified on 2026-11-24 against the official upstream sources.

- Source (upstream repository): `https://github.com/xiaowu0162/LongMemEval`
- Paper: `https://arxiv.org/abs/2410.10813`
- Official dataset: `https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned`
- Upstream commit (`UPSTREAM_COMMIT` in `adapter.py`): `9e0b455f4ef0e2ab8f2e582289761153549043fc` — the exact HEAD of the GitHub `main` branch at pin time, resolved with `gh api repos/xiaowu0162/LongMemEval/commits/main --jq .sha`. A resolved commit, not a branch name.
- HF dataset commit: `98d7416c24c778c2fee6e6f3006e7a073259d48f` — the pinned revision embedded in `DATASET_URL`; this is the commit that actually serves the artifact and is distinct from the GitHub commit above. Confirmed via `https://huggingface.co/api/datasets/xiaowu0162/longmemeval-cleaned/commits/main`.
- Artifact path: `longmemeval_oracle.json` within the HF dataset. It is the only released variant carrying the `answer_session_ids` labels this adapter reads; the `_s` and `_m` variants are not used.
- Dataset URL (`DATASET_URL` in `adapter.py`): `https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json`
- Dataset SHA-256 (`DATASET_SHA256`): `821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c` (artifact size 15,388,478 bytes). Verified two independent ways: the HF tree API reports the same value as `lfs.oid` for this file at the pinned commit (`https://huggingface.co/api/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f`), and `Get-FileHash -Algorithm SHA256` on the single downloaded artifact produced the same digest. No hash was invented.
- Judge pin: `gpt-4o-2024-08-06`, temperature `0` (`JUDGE` in `adapter.py`). Verified against upstream `src/evaluation/evaluate_qa.py` at the pinned commit: `model_zoo['gpt-4o']` resolves to `gpt-4o-2024-08-06` and the completion kwargs set `temperature: 0`. The adapter records this pin but cannot verify the injected `judge_fn` actually called it (`enforced_by_adapter: false`).

Upstream's exact eval command, recorded from the pinned README for reference only (this harness never runs it):

```bash
export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_ORGANIZATION=YOUR_ORGANIZATION     # may be omitted if your key belongs to only one organization
cd src/evaluation
python3 evaluate_qa.py gpt-4o your_hypothesis_file ../../data/longmemeval_oracle.json
```

`prepare()` downloads only after the pins are confirmed non-placeholder, hashes the bytes in a temp file, and moves them into place only when the SHA-256 matches. Downloads go only to the gitignored `data/` directory; dataset bytes are never vendored.

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
- The pinned commit and SHA-256 prove only that the bytes are the exact upstream artifact; they do not validate the upstream labels or the local slice.
- A better number here does not prove the vault is correct on real user histories.
