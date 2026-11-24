# MemoryAgentBench Conflict_Resolution — official-style SubEM lane

**New lane `mab-cr-subem`. Informational Tier C only. No rank, no win, no score asserted.**

This is deliberately separate from `harnesses/memoryagentbench/`. That lane computes a
**derived serial-number exact match** at 4096/8192 local tokens. It is a gate, not a
MemoryAgentBench score. This lane computes the **published** metric —
`substring_exact_match` (SubEM) — on the pinned `Conflict_Resolution` parquet, using a
reader and **no judge**.

## Pins (public provenance only; verified 2026-11-24)

| Pin | Value | Source |
|---|---|---|
| HF dataset revision | `7ea066982b140a19337e17e60d45d4076e042faf` | <https://huggingface.co/api/datasets/ai-hyz/MemoryAgentBench/revision/7ea066982b140a19337e17e60d45d4076e042faf> |
| CR artifact | `data/Conflict_Resolution-00000-of-00001.parquet` | <https://huggingface.co/api/datasets/ai-hyz/MemoryAgentBench/tree/7ea066982b140a19337e17e60d45d4076e042faf/data> |
| `DATASET_URL` | revision-pinned `resolve/.../data/Conflict_Resolution-00000-of-00001.parquet?download=true` | <https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/7ea066982b140a19337e17e60d45d4076e042faf/data/Conflict_Resolution-00000-of-00001.parquet?download=true> |
| `DATASET_BYTES` | `1,491,588` | HF tree `size` + recomputed bytes |
| `DATASET_SHA256` | `24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45` | HF Git-LFS `lfs.oid` **and** independently recomputed from the downloaded bytes |
| Upstream code commit | `538026089d1a8a8eff05121d0db89b388f360eba` | <https://github.com/HUST-AI-HYZ/MemoryAgentBench/commit/538026089d1a8a8eff05121d0db89b388f360eba> |
| Metric source | `utils/eval_other_utils.py` at that commit | <https://github.com/HUST-AI-HYZ/MemoryAgentBench/blob/538026089d1a8a8eff05121d0db89b388f360eba/utils/eval_other_utils.py> |
| Paper | ICLR 2026, arXiv:2507.05257 (Table 5, Table 6, Appendix B.4.2) | <https://arxiv.org/abs/2507.05257> |

The hash is embedded in the immutable revision URL, so the bytes cannot drift with
`main`. `prepare()` streams to a temporary file and verifies both byte count and
SHA-256 before moving it into `D:\memnoth-bench-data\datasets\mab\`. Nothing is
written inside the repository; the whole bench-data tree is capped at 80 GiB and a
download that would cross the cap is refused.

## Verified row structure (from the pinned parquet)

Eight rows, four columns (`context`, `questions`, `answers`, `metadata`):

```
factconsolidation_{sh,mh}_{6k,32k,64k,262k}   100 questions each = 800 questions
```

`context` is one long string of numbered facts (`0. …`, `1. …`, …), contiguous and
0-based; the SH and MH rows at a given context length share the same fact list.
`questions` is a list of 100 short natural-language strings; `answers` is a list of
100 lists, each holding one short gold string in this pinned file. `metadata.source`
carries the exact subset/length label used for filtering. Some optional metadata
fields declared in the parquet schema (`question_ids`, `question_types`, `keypoints`)
are `null` in this revision; they are not needed for this lane.

## What is measured

1. **Seed once.** Every numbered fact is planted verbatim into a temporary vault as a
   numbered entity. The serial lives in the entity name, never in the fact text.
2. **Query many.** Each question is used for a vault recall sweep; the recalled
   entities are reassembled as serial-prefixed evidence (newer = larger serial,
   matching the benchmark's conflict instruction) and handed to the frozen reader.
3. **Read.** `nvidia/nemotron-3-ultra-550b-a55b:free`, temperature 0, short-answer
   prompt, via `runner.locomo_reader` transport and `runner.kilo_pool` key rotation.
   Gold is never seeded and never sent to the reader.
4. **Score.** The published `substring_exact_match`: lowercase → strip ASCII
   punctuation → remove articles → collapse whitespace, then check that normalized
   gold is contained in normalized prediction, max over gold answers. For CR the
   pinned `post_process` falls through to `default_post_process`, so the score is
   `max(raw SubEM, parse_output SubEM)`. **No judge is called.**

Denominators are reported overall, by subset (SH/MH), by context length
(6k/32k/64k/262k), and by subset × length. A missing prediction is an error and is
never converted into a hit.

## Why this is still Tier C

The scoring function is the published one, but a benchmark result also pins a reader,
prompt, chunking, retrieval depth and sampling. Our harness supplies all of those.
The paper's Table 3 rows are self-reported under each vendor's own reader; there is
no working published-backbone key under this harness, so no paired same-model run
exists. Therefore every result carries:

```
official: false          # our harness, our reader — not an official MAB run
diagnostic: derived
comparability_tier: C
score_asserted: false
```

No `#1`, no "beats published", no rank column. The four context lengths are not four
independent systems, and 100 questions per row gives coarse resolution.

## Run

```bash
python -m py_compile harnesses/mab_cr/adapter.py runner/mab_cr_run.py
python -m harnesses.mab_cr.adapter --smoke
python runner/mab_cr_run.py --dry-run
python runner/mab_cr_run.py --subset all --context all --max-cases 0 --workers 8
```

`--dry-run` prepares the dataset and builds the exact case plan, then returns before
any key is loaded, any vault is opened, or any reader call is made; it writes a
result envelope with zeroed call counters as proof. Checkpoints are JSONL rows
(`case_id`, `pred`, `gold_hash_prefix`, `subem`, plus subset/context/error); results
are timestamped JSON under `D:\memnoth-bench-data\results`.

## Limitations

- Judge-free is not model-free: SubEM still depends on the reader's answer text.
- Vault recall returns a bounded set of fact entities; facts outside that sweep are
  not shown to the reader. Top-k and prompt are pinned in the runner's result file.
- `parse_output` is the pinned max(raw, parsed) rule; a label-prefixed answer can
  pass SubEM even though it would fail exact match. This is upstream behavior, not
  leniency added here.
- 64k/262k rows exist in the pinned file but have no published Table 5 validation
  row; numbers there are descriptive only.
- Upstream `main` may have moved since the commit pin; re-pin before any future run.
- Dataset bytes are third-party research material. Never commit them; commit only
  pins, code and derived diagnostics.
