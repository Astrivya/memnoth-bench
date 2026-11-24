# HotpotQA — answer-only EM/F1 derived diagnostic

**Status:** harness built and verified offline (2026-11-24). No score asserted,
no rank claimed, no win claim. Every artifact is stamped `official: false`,
`diagnostic: derived`, `comparability_tier: C`, `score_asserted: false` and
carries the standing **"NOT an official score"** note.

HotpotQA (Yang et al., 2018) is a public, CC BY-SA 4.0 **reading-comprehension**
benchmark, not a memory benchmark. This lane uses the distractor dev set as a
judge-free, multi-hop retrieval diagnostic: a pinned extractive reader sees a
question plus recalled paragraph text, and answer Exact Match / token F1 are
computed locally with the official SQuAD-style normalisation. There is no LLM
judge, so the largest source of score drift in the other lanes is absent.

## Verified pins

Public provenance (verified 2026-11-24): HF revision
`1908d6afbbead072334abe2965f91bd2709910ab`, file
`distractor/validation-00000-of-00001.parquet`, SHA-256
`c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6`. The pins
are transcribed as constants in `adapter.py`:

| Item | Value |
|---|---|
| HF dataset / config / split | `hotpotqa/hotpot_qa`, `distractor`, `validation` (dev) |
| Immutable HF revision | `1908d6afbbead072334abe2965f91bd2709910ab` |
| File | `distractor/validation-00000-of-00001.parquet` |
| URL | pinned `resolve/<revision>/…` — never `resolve/main/` |
| SHA-256 | `c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6` |
| Bytes / rows | `27,452,575` / `7,405` (all `level = hard`) |
| Columns | `id, question, answer, type, level, supporting_facts, context` |

Independent local re-verification on 2026-11-24 confirmed the byte count, hash,
row count, column schema (`context` = `title[10]` + `sentences[10][…]`), the
first/last five sorted IDs, and the 150-row ID-list digest below.

The pinned slice is **n = 150**: sort all 7,405 `id`s lexicographically, take the
first 150. SHA-256 of those IDs joined by `\n` =
`5e3e6eb0e03ff6a866aa0ace6b0eaf569d1b32f89f6ad87c7cdda99e2ee960e9`.
Type mix: 117 bridge / 33 comparison; answer length 1–9 words, median 2; zero
yes/no answers in the slice. This is a locally derived slice, **not** the
published split.

## Metric (exact answer-only semantics)

From the pinned `hotpot_evaluate_v1.py` (transcribed, not vendored):

```python
normalize_answer(s) = white_space_fix(remove_articles(remove_punc(lower(s))))
# lower -> strip string.punctuation -> remove standalone a|an|the -> collapse whitespace
```

* **EM** = `normalize_answer(pred) == normalize_answer(gold)`.
* **F1** = `Counter` overlap on normalised tokens; `2PR/(P+R)`; zero on no overlap.
* **Yes/no guard**: if either normalised side is `yes|no|noanswer` and they differ,
  F1 and precision/recall are all zero. (The 150-row slice has no yes/no answers;
  a full-dev run would exercise the guard.)
* A missing/abstaining prediction is scored **0 explicitly** and counted, never
  skipped. Numerators are always published with denominators (e.g. `63/150`).

Supporting-fact (`Sup`) and `Joint` scores are deliberately **not** computed:
this lane is answer-only and does not claim to reproduce the official full
protocol. The official scorer copy/hash has **not** been frozen; the adapter
reports `scorer_sha256: None` rather than inventing a hash.

## Pipeline (per case)

1. **Ingest (no gold).** `adapter.paragraph_records()` reads only `context` and
   emits all 10 distractor paragraphs (title + sentences) as deterministic vault
   entities. `answer` and `supporting_facts` are never read here.
2. **Recall (no gold).** The runner creates one temp vault per case, seeds those
   10 paragraphs via `mn vault remember`, and queries recall with the **question
   text only**, capped at `K = 10`.
3. **Reader (no gold).** A pinned extractive prompt asks for only the short answer
   span. The reader receives the question plus recalled paragraph text — never
   the gold answer, supporting-fact labels, `id`, `type` or `level`. Default
   reader: `particle-ai/particle-1:free`; an override is recorded in the result.
4. **Score locally.** `adapter.evaluate()` recomputes EM/F1 from the official
   normaliser and reports real denominators. Gold exists in memory only and is
   never written to a checkpoint or result.

Because the per-case vault contains only that question's 10 paragraphs, this is
the **oracle-context-style arm**, not the cross-question vault/KG-recall arm.
No paired arms are implemented here, so no retrieval-vs-reader attribution is
made and no advantage claim follows.

## Running

```bash
# Offline verification: pins, deterministic slice, no calls, no score.
python runner/hotpotqa_run.py --dry-run

# Offline unit checks for the scorer and the gold-leakage guard.
python harnesses/hotpotqa/adapter.py --smoke

# Real derived diagnostic (needs the mn binary and at least one gateway key).
python runner/hotpotqa_run.py --max-cases 150 --workers 8 --keys 8
```

`prepare()` streams the pinned parquet to
`D:\memnoth-bench-data\datasets\hotpotqa` (outside the repo), enforces the
80 GB bench-data cap and disk-space margin, and verifies SHA-256 **and** byte
count before moving bytes into place. Pass a `local_dir` to redirect it; it must
stay under the approved data root. Checkpoints are JSONL with exactly five
fields — `question_id`, `pred`, `gold_hash_prefix`, `em`, `f1` — so no gold value
or key value can be written. `--dry-run` makes no network, CLI or model call.

## Limitations / honesty

* A 150-question derived slice with a generic LLM reader is **not** comparable to
  published leaderboard rows (which use the full 7,405-row split and mostly
  fine-tuned extractive readers). Published figures are never restated as ours.
* The vault ingestion of "all 10 paragraphs per question" is contrived; any
  number produced here is a retrieval/reader diagnostic, not a product capability
  claim and not a memory-quality result.
* The reader transport reuses `locomo_reader._post_json` (rate gate, bounded
  retries, short error codes); the gateway does not attest model identity, so
  the model string is a recorded call argument, not a verified snapshot.
* Context is bounded (2000 chars per paragraph, K=10). Recall quality, prompt
  bytes and reader model are pinned but not tuned on the scored slice.
* **Wording to use:** "Derived diagnostic on 150 pinned HotpotQA distractor-dev
  questions with a pinned extractive reader; judge-free EM/F1; `official: false`,
  `diagnostic: derived`, `comparability_tier: C`, `score_asserted: false`. NOT an
  official score. No ranking against published systems is made."
