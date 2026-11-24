# RULER — derived, judge-free retrieval diagnostic

This harness uses [NVIDIA/RULER](https://github.com/NVIDIA/RULER)'s deterministic
scoring rule to measure one thing only: can this product's vault recall a short
planted value from a large synthetic haystack? It is **not** RULER's official
runner and produces **no official RULER score**.

## What is pinned

- Code: `db43f92ff634bbce1c33d1fb15858aec4bba509a` (classic script pipeline on
  `main`). SHA-256 values for nine upstream files are recorded in
  `SOURCE_FILE_PINS`; they are provenance only and no upstream code is run.
- Data: third-party mirror `self-long/RULER-llama3-1M`, revision
  `5da49e4000224d3289ffb14fc32cbd20211765ed`. Only `niah_single_1:4k` has a
  real, locally recomputed SHA-256
  (`03b988e7d1eb9b380b7a658641caf5692dc8d99be89712a87e65003626f417f2`,
  416,867 bytes). **8K/16K/32K/64K/128K are UNVERIFIED**: `prepare()` refuses
  them before any network call, and no digest is invented from HF LFS metadata.
- Metric: `string_match_all` / `string_match_part` from the pinned
  `scripts/eval/synthetic/constants.py` — case-insensitive **substring
  containment**, not strict equality. Predictions are post-processed exactly as
  upstream does (`strip()`, control characters to `\n`, `strip()`).
- Default generation parameters: `random_seed=42`, `NUM_SAMPLES=500`. Lengths
  are RULER generator target tokens, not vault token counts.

## Verified 4K file structure

The pinned parquet was inspected locally (read-only, SHA verified). Its
columns are `index`, `input` (string), `answers` (`list<string>`), `length`,
`predictions` (`struct` of third-party model outputs, never read). The RULER
question is embedded in the tail of `input`, immediately before
`<|eot_id|><|start_header_id|>assistant`; the assistant answer prefix is
excluded. `build_rows()` parses that template, treats everything between the
instruction line and the question as the haystack, keeps each line intact and
groups lines deterministically into chunks (`CHUNK_LINES = 5`) with stable
entity ids `ruler_<length>_<index>_<chunkNNN>`. Gold is taken only from
`answers`; a row whose question contains a gold value is dropped by the
no-leakage guard.

## Judge-free by construction

There is no `judge_fn` parameter, no key-pool import, no reader model and no
completion endpoint in this adapter or its runner. Scoring is string
containment plus a derived per-sample substring-match F1 (precision uses
numeric answer-shaped candidates in the prediction; when that is not
applicable, F1 reduces to containment recall). No API key is needed at any
stage — including dataset preparation, which only downloads a pinned public
parquet.

Gold is used only by the scorer. It is never seeded into the vault, never sent
to `vault recall`, and never written to a checkpoint or result (only a hash
prefix). The recall query is the RULER question: key only, never the value.

## API

```python
from harnesses.ruler import adapter

adapter.validate()                     # offline pin check, no network
path = adapter.prepare("niah_single_1", "4k")  # verified download, external dir
rows = adapter.build_rows(path, max_cases=20)  # deterministic chunks + gold-free queries
verdict = adapter.evaluate([{"case_id": ..., "pred": ..., "gold": [...], ...}])
adapter.self_test()                    # offline synthetic smoke; no keys, no network
```

`evaluate()` emits `official: false`, `diagnostic: "derived"`,
`comparability_tier: "C"`, `score_asserted: false`, real denominators
(`supplied`, `scored`, `excluded`) and per-sample hash-only verdicts.

## Runner

```bash
python runner/ruler_run.py --dry-run --max-cases 20   # zero keys/network/CLI calls
python runner/ruler_run.py --task niah_single_1 --length 4k --max-cases 20 --workers 8
```

The real runner seeds chunks into a temporary vault with eight-way CLI
parallelism (`mn vault remember`), recalls with the question (`mn vault recall
--limit 10`) and reads the top entities' fact content (`mn vault asof`). All CLI
calls go through `adapters.mn_json`, which gives the child an isolated home and
no inherited provider keys. Checkpoints are JSONL under
`D:\memnoth-bench-data\checkpoints\ruler`; results under
`D:\memnoth-bench-data\results`. Datasets are written only under
`D:\memnoth-bench-data\datasets\ruler` with an 80 GB directory cap.

## Limits

- Vault recall is not RULER's LLM reader; any number here is a property of this
  harness, not a comparable RULER row and not a rank against any vendor.
- Needle-in-a-haystack is saturated upstream; there is no headline to claim.
- The harness does not test as-of reasoning, contradiction handling,
  provenance or KG traversal.
- `qa_1`/`qa_2` are excluded: their gold derives from SQuAD/HotpotQA and is not
  fully synthetic.
