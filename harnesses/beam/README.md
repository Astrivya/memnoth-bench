# BEAM harness (1M / 10M)

Derived diagnostic only. **No score is claimed.** `adapter.py` is structural +
proxy-scoring; it does not run a model or a judge by itself.

- Paper: *Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs*,
  ICLR 2026, arXiv:2510.27246 — <https://arxiv.org/abs/2510.27246>
- Upstream: <https://github.com/mohammadtavakoli78/BEAM>
- Verified upstream commit: `b2da22eac88bb0874c64665f13457eb99835774a`
  (resolved via `https://api.github.com/repos/mohammadtavakoli78/BEAM/commits/main`)

## Verified dataset pins

Every SHA-256 below was verified two ways on 2026-11-24: the HuggingFace tree API
reports it as the Git-LFS `oid` at the pinned commit, and it was recomputed with
`hashlib.sha256` over the full downloaded artifact (byte counts match exactly).
The probe questions are embedded in these same parquet files under
`probing_questions`; there is no separate probe artifact. No hash was invented.

| Lane | HF repo (immutable commit) | File | Bytes | SHA-256 |
|---|---|---|---|---|
| `beam-1m` | `Mohammadta/BEAM` @ `3205395e897e7318c7b094ef4e6047b9b82dbb03` | `data/1M-00000-of-00001.parquet` | 66,156,374 | `41b5acbbb55a586b1305514ef9d9fb03365d9b3331b598a1c2dd7603d93ef533` |
| `beam-10m` | `Mohammadta/BEAM-10M` @ `9b2096193fe74e2837e4713e483351e19817773c` | `data/10M-00000-of-00002.parquet` | 153,844,664 | `31d96fd47ec56221d202e68792f26c00e49467dd4b36ee105c36ebd19ef78ad5` |
| `beam-10m` | `Mohammadta/BEAM-10M` @ same commit | `data/10M-00001-of-00002.parquet` | 189,980,875 | `a4f13fe25af51d57405ae41008689c31d1421377f3efde56a024b441deb2ee65` |

Resolve URL pattern:

```
https://huggingface.co/datasets/<repo>/resolve/<commit>/<file>
```

Provenance endpoints used:

- `https://huggingface.co/api/datasets/Mohammadta/BEAM/commits/main`
- `https://huggingface.co/api/datasets/Mohammadta/BEAM/tree/<commit>/data`
- `https://huggingface.co/api/datasets/Mohammadta/BEAM-10M/commits/main`
- `https://huggingface.co/api/datasets/Mohammadta/BEAM-10M/tree/<commit>/data`

The 1M tier has 35 conversations × 20 questions = 700; the 10M tier has
10 × 20 = 200. Ten memory abilities, two questions each. The HF viewer labels
the small split `100K` while the paper says `128K`; this adapter pins only 1M
and 10M, so that naming difference does not affect it. The pinned upstream
evaluation script at the commit above uses nugget scoring (0.0/0.5/1.0 per rubric
item, Kendall tau-b for event ordering) with `gpt-4.1-mini` — that is **not** the
proxy implemented here.

## Frozen harness — do not vary per run

| Role | Model | Provider | Constant |
|---|---|---|---|
| Reader | `nvidia/nemotron-3-ultra-550b-a55b:free` (550B, 1M ctx) | Kilo | `READER_MODEL` |
| Judge | `baidu/ernie-5.5-flash-preview:free` | Kilo (different provider from reader) | `JUDGE_MODEL` |

- `reader_call(question, context, *, api_key, ...)` and
  `judge_call(question, answer, gold_answer, rubric, *, api_key, ...)` are real
  OpenAI-compatible gateway calls against `https://ai-gateway.kilo.ai/api/v1`.
  Pass a key from `KeyPool.get()`; no key is stored, logged, or printed.
- `build_rows` marks every row `answer_source = "reader over CLI-recalled facts"`
  and records `reader_model` / `judge_model`, so any figure is attributable.
- Raw judge responses are retained in `judge_raw` / `judge_response` and
  published; the judge never replaces the deterministic primary metric. We do
  not retry with a different judge, and we do not select the most lenient one.
- All free Kilo models set `mayTrainOnYourPrompts:true`. That is acceptable for
  synthetic benchmark content only — never send private data through them.

## Comparability — Tier C

- The BEAM paper does **not** fix a reader or judge. Published paper baselines
  (LIGHT / RAG with Llama-4-Maverick and the paper scorer) are **not comparable**
  to vendor rows or to this harness.
- Third-party BEAM reports variously use `gpt-5`-family judges; the pinned
  upstream script uses `gpt-4.1-mini`. Same answers have been observed to rescore
  roughly 76.83 / 78.18 / 82.23 across three scorers. None of those pins is
  reproduced here.
- Our result is therefore **Tier C derived** until a paired, same-pin run
  exists: `official=False`, `diagnostic="derived"`, `comparability_tier="C"`.
- The primary metric is a deterministic **SubEM/F1 proxy** against the dataset's
  ideal-answer strings. It is not the official nugget score and must not be
  presented as one.

## Encoded retrieval profile

`RETRIEVAL_PROFILE` freezes the configuration the surrounding system is expected
to use. The adapter itself performs no recall; it only formats facts it is given.

- absolute-date grounding and chronological evidence layout
- atomic facts with speaker + timestamp (`format_context`)
- hybrid recall + reciprocal-rank fusion (`rrf_fuse`) + rerank, `top_k = 15`
  (K=15 is reported best for BEAM; K=20 degrades)
- multi-query decomposition with a sufficiency check
- temporal invalidation / versioned facts via `valid_from` / `valid_to`
  (superseded intervals are marked in the rendered context)

## No gold leakage

The reader receives only the question text and recalled conversation facts.
Ideal answers, rubric nuggets and other probing-question annotations are used
only by `evaluate` and `judge_call`, never in a reader prompt. `build_rows`
attaches `gold_answer` to the result row for scoring, with
`gold_withheld_from_reader: True`; the reader callback signature is
`reader_fn(question, chat)` and receives no gold field.

## Interface

Mirrors `harnesses/longmemeval/adapter.py`:

```
LANE, SOURCE_URL, PAPER_URL, UPSTREAM_COMMIT
DATASET_URL, DATASET_SHA256            # 1M defaults; multi-shard tiers use TIERS
TIERS, ABILITIES, READER_MODEL, JUDGE_MODEL
prepare(local_dir="data/beam", tier=None) -> Path
select(max_cases=None, tier=None) -> dict
validate(record) -> list[str]
build_rows(dataset, *, reader_fn=None, max_cases=None, abilities=None, tier=None) -> (rows, attempted, failures)
reader_call(...) / judge_call(...)
evaluate(results, judge_fn=None, *, tier=None) -> dict
self_test() -> dict
```

`stamps()` supplies `official=False`, `diagnostic="derived"`, `observed_at`;
denominators count only rows actually supplied, and there are no bare rates.

## Limitations

- No model, judge, or benchmark run was executed here; pins and structure were
  verified, not scores.
- The 10M tier needs both parquet shards (~344 MB total on disk); `prepare()`
  downloads and verifies both before returning.
- `reader_call` / `judge_call` cannot confirm the remote model identity; the
  caller must enforce the frozen pins and record them.
- Dataset redistribution is CC BY-SA 4.0. Only pins, IDs and derived diagnostics
  live in this repository; upstream bytes are never vendored.
