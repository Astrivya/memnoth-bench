# MemoryAgentBench — pinned harness adapter (FactConsolidation)

**Informational diagnostic only. Not an official benchmark score. This is our highest-ROI lane.**

## Source and pins

All pins verified 2026-11-24 from public sources. Nothing below is vendored.

| Pin | Value | Where verified |
|---|---|---|
| Official code repo | `HUST-AI-HYZ/MemoryAgentBench` — <https://github.com/HUST-AI-HYZ/MemoryAgentBench> (MIT, ICLR 2026) | Paper <https://arxiv.org/abs/2507.05257> links "Source Code" to this repo; `gh api repos/HUST-AI-HYZ/MemoryAgentBench` |
| `UPSTREAM_COMMIT` | `538026089d1a8a8eff05121d0db89b388f360eba` | <https://github.com/HUST-AI-HYZ/MemoryAgentBench/commit/538026089d1a8a8eff05121d0db89b388f360eba> (`refs/heads/main` HEAD, 2026-09-25, "Merge pull request #25 from vbcherepanov/feat/total-agent-memory-method"). `main` can move; re-pin before any future run. |
| MQUAKE source | `princeton-nlp/MQuAKE` — <https://github.com/princeton-nlp/MQuAKE> | Paper Appendix B.4.2 cites MQUAKE (Zhong et al., 2023b), "MQuAKE: assessing knowledge editing in language models via multi-hop questions", arXiv:2305.14795 — authors Z. Zhong, Z. Wu, C. D. Manning, C. Potts, D. Chen; `gh api repos/princeton-nlp/MQuAKE` |
| `MQUAKE_COMMIT` | `fb43dadc2d8cd19d08ce81c63d957b59deb3f3cd` | <https://github.com/princeton-nlp/MQuAKE/commit/fb43dadc2d8cd19d08ce81c63d957b59deb3f3cd> (`refs/heads/main` HEAD, 2024-09-12, "update mquake-cf-3k-v2 (#12)") |
| HF dataset | `ai-hyz/MemoryAgentBench` — <https://huggingface.co/datasets/ai-hyz/MemoryAgentBench> | Linked from the official repo README; `gh api` equivalent: `https://huggingface.co/api/datasets/ai-hyz/MemoryAgentBench` |
| HF revision (immutable) | `7ea066982b140a19337e17e60d45d4076e042faf` | `sha` field of the HF dataset API, last modified 2026-06-21. The revision is embedded in the resolve URL, so the bytes cannot drift with `main`. |
| Artifact path in that dataset | `data/Conflict_Resolution-00000-of-00001.parquet` | HF tree API: <https://huggingface.co/api/datasets/ai-hyz/MemoryAgentBench/tree/main/data> |
| `DATASET_URL` | <https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/7ea066982b140a19337e17e60d45d4076e042faf/data/Conflict_Resolution-00000-of-00001.parquet?download=true> | Revision-pinned resolve URL for the exact file hashed below |
| `DATASET_SHA256` | `24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45` | Computed locally from the 1,491,588 downloaded bytes; independently matches the `lfs.oid` SHA-256 reported by the HF tree API for that file. Traceable, not invented. |
| Judge pin (optional, semantic verdict only) | `gpt-4o-2024-08-06`, temperature `0` (`JUDGE`) | Records the pin only; `enforced_by_adapter: false`. The adapter cannot verify a remote call. |

**How the hash was computed (reproducible):** the revision-pinned URL was downloaded with `curl -L` to `D:/temp/opencode/mab/Conflict_Resolution-00000-of-00001.parquet` (outside the repo), then hashed with PowerShell `Get-FileHash -Algorithm SHA256`. The result above equals the Git-LFS object ID published by Hugging Face for the same path, so two independent sources agree.

**Why this artifact:** the HF dataset ships four split parquets; `Conflict_Resolution` is the only one containing FactConsolidation and is the smallest independently downloadable slice that carries **both** SH and MH (1.4 MB). It has 8 rows — `factconsolidation_{sh,mh}_{6k,32k,64k,262k}` with 100 questions each. There is no smaller per-subset artifact and no separate 6K/32K-only file, so this is the tightest real pin available. `prepare()` stores it as gitignored `data/memoryagentbench/Conflict_Resolution-00000-of-00001.parquet`; dataset bytes are never committed.

`prepare()` refuses to download on placeholder or malformed pins. Downloads are streamed to a temp file, hashed, and moved into place only after the SHA-256 matches. It is never called at import time or by `--smoke`.

## Research context (from the paper, not from this adapter)

- The parent competency is **Selective Forgetting**: FactConsolidation is built from MQUAKE counterfactual true + rewrite pairs, ordering the rewritten (new) fact after the original. Contexts are 6K, 32K, 64K and 262K tokens; SH is single-hop, MH is multi-hop. (§3.1, Appendix B.4.2 of <https://arxiv.org/html/2507.05257v4>; ICLR 2026 PDF <https://proceedings.iclr.cc/paper_files/paper/2026/file/fd1eff9dd295df50a41f2521942fa31d-Paper-Conference.pdf>.)
- The **serial-number guardrail** is explicit in the benchmark prompt: facts are indexed by serial numbers and agents are told "newer facts have larger serial numbers", then required to resolve conflicts by selecting the newest fact (§3.3; full templates in Appendix D). That is the construct our primary metric probes.
- The **published metric is SubEM (substring exact match)**, per the repo's metric table (`fact_sh` / `fact_mh` → `substring_exact_match`) and Appendix B.4.2. Our strict serial-field exact match is a **derived, judge-free diagnostic**, not that metric.
- Published validation (paper Table 5, <https://arxiv.org/html/2507.05257v4>): **o4-mini SH 100.0 @ 6K / 61.0 @ 32K; MH 80.0 @ 6K / 14.0 @ 32K** (GPT-4o: SH 92/88, MH 28/10). The paper uses these reasoning-model results to show the dataset is solvable at 6K but degrades sharply at 32K.
- Published agent ceiling (paper Table 3 and §4.2): all evaluated methods reach **at most 28% on FC-MH** (that 28% is GPT-5-mini, a long-context model); the memory agents themselves are far below it (MemGPT 3%, MIRIX 2%, Mem0 2%, Cognee 3%, Zep 3%, GraphRAG 2%, etc.). The paper's conclusion is that selective forgetting remains unsolved by current memory agents.

## Slice

FactConsolidation: SH 24 + MH 16 cases at 4096 and 8192 context tokens, from `fixtures/pinned-ids.json`. This is a **locally constructed slice, not an official split** (`official_split: false`); its 4096/8192 settings are **not** the paper's published 6K/32K settings, so no number here is comparable to Table 5. Planned fixture counts are echoed with `authoritative: false`; the observed valid denominator is always authoritative and is never inflated to match the plan.

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

MemoryAgentBench and MQUAKE datasets and code are **research-only, third-party material**. The official repo is MIT; the HF dataset card is tagged `license: mit`, while the paper's ethics statement says code is released under MIT and datasets/benchmark artifacts under CC BY 4.0 with third-party materials retaining their original licenses — treat the stricter reading as binding. Memnoth's Apache-2.0 license does **not** grant redistribution rights to upstream data. Never commit dataset bytes or upstream checkouts; commit only pins, IDs, and derived diagnostics. Verify the upstream licenses before any internal use and honor their terms.

## What the metric does NOT prove

- Not an official MemoryAgentBench result; the slice, counts and denominators are local.
- Our strict serial exact match is **not** the benchmark's published SubEM metric, and our 4096/8192 local settings are **not** the published 6K/32K settings. The o4-mini / Mem-agent figures above are context from the paper, not results produced by this adapter and not a target to compare against.
- Serial exact match measures a structured field only — not a natural-language answer, not real user behavior, and not retrieval or reasoning quality in general.
- The adapter cannot verify offline that `gold_serial` truly came from the pinned MQUAKE true+rewrite pair; that depends on `prepare()` using the real pinned artifact.
- 4096 vs 8192 token rows are descriptive, not a causal context-length claim.
- A canonical-match diagnostic exists but is never substituted for the strict primary metric.
- Judge accuracy depends on an unverified injected callable and is never promoted over the primary metric.
- Passing this lane does not prove conflict resolution across arbitrary schemas, tenants, or long histories. No "#1" or "best" claim follows.
- The upstream `main` commits are point-in-time pins; they may move upstream and must be re-verified before a future run. The HF revision and SHA-256 are immutable for the pinned artifact.
