# AMB / PrecisionMemBench retrieval lane

`adapter.py` pins AMB's `precisionmembench` dataset in `--mode retrieval` and
provides the missing glue between AMB's Python `MemoryProvider` contract and our
public CLI's `vault remember` / `vault recall` surface.

This is a **judge-free derived diagnostic**. There is no answer LLM, no judge and
no reranker. The only signal is whether a belief ID comes back in the top-k.
Every number produced here is stamped `official=false`, `diagnostic=derived`,
`comparability_tier=C`, `score_asserted=false`, `judge=null`, with the standing
**"NOT an official score"** note. No win, rank or leaderboard comparison is
claimed or implied.

## Why an adapter exists

AMB expects `ingest(documents)` / `retrieve(query, k, user_id)` with document IDs
that round-trip to gold belief IDs. Our MCP has no such contract: `mem_ingest`
takes a directory, `mem_add_fact` has no `beliefId`, and `mem_search` has no
per-call `user_id`. This adapter supplies, deterministically:

1. **Belief-ID mapping** — identity (`b-db-decision` stays `b-db-decision`). The
   map is built once from the pinned corpus and asserted bijective. Identity is
   chosen because PMB IDs are already unique, lowercase, whitespace-free and
   FTS-friendly; a synthetic prefix would add noise tokens that hurt short-alias
   queries such as `k8s`, `ts`, `gha` and `kube`.
2. **Per-user vault isolation** — a case is seeded only with beliefs whose
   `user_id` equals the case's `user_id`. The `other-user` belief therefore never
   enters a `test-user` vault; the cold-start case gets an empty vault.
3. **Fresh-vault reset** — a new temp vault file per case. The bi-temporal vault
   invalidates rather than deletes, so a fresh file is the honest `/reset`.

Gold belief IDs are used **only** by the local scorer. `recall_fn` receives the
case's `query` text verbatim; no gold ID, ID map or expectation is ever sent to
it, and checkpoints carry only a `gold_hash_prefix`.

## Verified pins

Every hash below was computed locally from bytes downloaded at the exact
`resolve/<commit>/` URL. Mutable branch URLs are never used.

| Item | Verified value |
|---|---|
| AMB repo (public) | `https://github.com/vectorize-io/agent-memory-benchmark` |
| AMB HEAD | `01bd83f3fa5245d6be3986cfab81fdd828f26acd` |
| Upstream PrecisionMemBench commit | `b95d6abb471c0d591c440172283dd74e7af000df` |
| Dataset | `precisionmembench` / `single-turn`, `--mode retrieval` |
| Corpus / cases | 35 beliefs / 77 cases |
| AMB self-check split | 43 active / 25 structural / 9 trivially-empty |

Vendored `.json.gz` files at AMB HEAD, under `data/precisionmembench/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `info.json.gz` | 182 | `e2b02701e3accece8a98ce2eb1842f19dd623ea28628752c48213c02dcdb87ad` |
| `single-turn/categories.json.gz` | 1,351 | `d5ff4a1df28138d90bb430d4713b36ec56d6913d063093ac67909acae9f83c9a` |
| `single-turn/documents.json.gz` | 3,604 | `a7bb7ed587f1fee494559211313a75a23133170863070c1ef9bb13e95df11b46` |
| `single-turn/queries.json.gz` | 7,261 | `fdbd66f0bb907215cc21282f7879bdae88fb499b874ede615d3c0acb3c9fbfa3` |
| `single-turn/stats.json.gz` | 388 | `c5637f8204805421a7a0f495b5ff0c0520ea287624ee44a50a584742bb96b575` |

Upstream fixtures at the pinned PMB commit, under `fixtures/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `beliefs.seed.json` | 33,303 | `35659b3b9a581b644fc6746e1b4e902c3e7b56c0973b546bf7b0816a69e3de98` |
| `retrieval.cases.json` | 35,940 | `09e99f0852280538abe34b03570f65583fdb10df3c554b7150f0f665d5db709c` |

The vendored `documents.json.gz` projects each belief down to
`{id, content, user_id, timestamp}`. The full `beliefs.seed.json` supplies the
metadata (`pinned`, `scope`, `type`, `resolved_at`, `superseded_by`,
`participants`) that AMB's provider-independent scoring tiers need. Both sources
are verified pins and their query ID sets are cross-checked at load time.

**Retired MemBench is explicitly UNVERIFIED — NOT COMPUTED.** Its test files
ship through mutable links with no published digest, and its AMB adapter was
removed on 2026-08-03. `refuse_membench_data()` actively refuses it; no hash is
invented.

## Data flow

```
prepare(local_dir)            download + SHA/byte verify 7 pinned artifacts
        │                     into D:\memnoth-bench-data\datasets\amb
        ▼
load_payload(data_dir)        -> (beliefs, cases, documents)
        │                     offline; re-verifies SHA; asserts 35/77/35,
        │                     stats agreement, ID-set agreement, 43/25/9 split
        ▼
build_rows(cases, beliefs,    per case: seed_fn(entity, text) one belief at a time
           seed_fn, recall_fn)-> recall_fn(query, k) returns entity names
        │                     -> score_case() ports AMB score_retrieval
        ▼
evaluate(rows)                belief-ID case pass + active/structural/empty
                              denominators, precision, recall, honesty stamps
```

`seed_fn` and `recall_fn` are injected: the adapter is pure logic and owns no
process, key or model. `runner/amb_run.py` binds them to a fresh temp vault and
the public `mn vault ... --json` surface.

## Scoring semantics (ported from the pinned AMB adapter)

AMB splits each case into four tiers. Only one is provider-dependent:

- `personaPrelude` — a fixed harness constant, non-empty for `test-user`.
- `pinnedFacts` — derived locally from pinned, in-scope, current beliefs.
- `openQuestions` — derived locally from pinned, in-scope open questions.
- `relevantBeliefs` — **the measurement**: our recalled belief IDs, with pinned
  beliefs removed, relation participants expanded one hop, and capped at
  `min(top-k, maxBeliefs)`.

Assertions handled: `mustInclude`, `mustExclude`, `shouldInclude`,
`shouldOnlyInclude`, `minCount`, `maxCount`, `orderedBefore` (relevant tier);
`mustInclude`/`mustExclude` (pinned and question tiers); `nonEmpty`, `isNull`,
`contains`, `mustNotContain` (persona tier). Precision and recall are computed
over `relevantBeliefs` only, exactly as upstream does.

Case pass types follow AMB's three-way split:

- **active** — satisfying the case requires query-dependent belief IDs. This is
  the comparable number (`active_passes / 43` on the full pinned set).
- **structural** — nothing relevant is expected, but a local-tier assertion
  (scope isolation, supersession, routing) still decides the case.
- **trivially-empty** — an empty relevant tier and no other assertion; only a
  provider that abstains passes.

The default `--top-k` is 10 while AMB's pinned `maxBeliefs` budget is 20. The
effective budget is `min(top-k, 20)` and is recorded per case. This is a declared
operating point, not a claim of comparability to a published row.

## Running

```bash
# Offline: pins, selection, stamps; no network, CLI, model or keys.
python runner/amb_run.py --dry-run

# Offline synthetic checks of pins, ID map, gold isolation and scoring.
python -c "from harnesses.amb.adapter import self_test; print(self_test())"

# Real run (needs a public `mn` binary; no API keys):
python runner/amb_run.py --max-cases 77 --top-k 10 --workers 8
```

A real run writes JSONL checkpoints with exactly four fields
(`case_id`, `retrieved_ids`, `gold_hash_prefix`, `hit`) and a result JSON stamped
as above. `--dry-run` proves `keys_loaded=0`, `model_calls=0` and `judge=null`.

## Files

- `adapter.py` — pins, download/verify, offline loading, ID map, AMB scoring,
  `validate`, `evaluate`, `select`, `self_test`.
- `README.md` — this file.

Dataset bytes never enter the repository: they land under
`D:\memnoth-bench-data` with a hard 80 GB cap. See the build report in the
bench-data `reports/` directory for verification counts and limitations.
