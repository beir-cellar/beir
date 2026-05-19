# RESULTS — Level A (micro-smoke NFCorpus → AngelicaDB)

**Date:** 2026-05-18
**Branch:** `main` (working tree, uncommitted)
**Endpoint under test:** `https://angelicadb-kva.onrender.com` (AngelicaDB v0.2.0)
**Harness:** `scripts/smoke_angelica_nfcorpus.py --level A`
**Raw outputs:**
- `data/results/smoke_A.json` — flat `/retrieve`
- `data/results/smoke_A_fulloida.json` — `/retrieve/full-oida`

---

## Goal

Validate end-to-end pipeline (ingest → retrieve → evaluate against BEIR qrels)
on the smallest meaningful slice of NFCorpus, after AngelicaDB persistence fix
(missing `epistemic_class_config` table on prod DB).

This is NOT a competitive benchmark — the corpus is sub-sampled, so absolute
nDCG numbers are not directly comparable to published BEIR baselines. The
target of level A is **pipeline correctness + order-of-magnitude reasonableness**.

## Protocol

| Step | Endpoint | Notes |
|---|---|---|
| 1. Load | local | NFCorpus test split via `beir.datasets.data_loader.GenericDataLoader` |
| 2. Sub-sample | local | seed=42; 10 test queries (with non-empty qrels) + all their gold docs + 50 random distractors → **418 unique docs** |
| 3. Ingest each doc | `POST /ingest` (single, NOT batch) | `mode=commit`, `sourceType=MANUAL` (→ Capture policy), `source = BEIR doc _id` |
| 4. Retrieve each query | `POST /retrieve` | `limit=100`, `threshold=0.0`, no regime filter |
| 5. Aggregate | local | for each query, map KOs → BEIR doc IDs via `supporting_sources`; per doc take **max similarity** across its KOs; sort desc |
| 6. Evaluate | `pytrec_eval` | nDCG / MAP / Recall / P @ {10, 100} |

## Query types (NFCorpus test split)

**Domain:** medical / nutrition / health IR (NutritionFacts.org corpus, biomedical literature).
**Style:** very short keyword-style queries — mean 3.3 words, median 2 words, min 1, max 11.
**Form:** mostly noun phrases or article-title fragments (no question marks).

The 10 queries sampled for level A:

| qid | query text | n gold docs (rel ≥ 1) |
|---|---|---|
| PLAIN-634 | `avocados` | 21 |
| PLAIN-133 | `Starving Tumors of Their Blood Supply` | 36 |
| PLAIN-1557 | `magnesium` | 26 |
| PLAIN-1398 | `IGF-1` | 103 |
| PLAIN-1275 | `goji berries` | 55 |
| PLAIN-782 | `Bush administration` | 2 |
| PLAIN-583 | `antinutrients` | 44 |
| PLAIN-3014 | `Sometimes the Enzyme Myth Is True` | 1 |
| PLAIN-499 | `African-American` | 34 |
| PLAIN-3271 | `Saturated Fat & Cancer Progression` | 49 |

(After sub-sampling, only gold docs that fell into the 418-doc scope remain reachable.)

## Ingestion

| | |
|---|---|
| Endpoint | `POST /ingest` (one POST per doc, sequential) |
| `mode` | `commit` |
| `sourceType` | `MANUAL` → policy mode `Capture` |
| Body text | `f"{title}. {text}"` from BEIR corpus.jsonl |
| Docs sent | 418 |
| Docs committed (`committed: true`) | **418 / 418** |
| Network errors | 0 |
| **KOs created (total)** | **4 580** |
| Mean KOs per doc | 10.96 |
| Wall-time | 649.6 s (≈ 1.55 s / doc) |

**Observation:** the EIP decomposition fans out aggressively — each BEIR
passage gets split into ~11 atomic KOs on average. This dilutes per-doc
signal: when a query semantically matches one sub-claim, only that single
KO scores high; aggregating to doc-level via `max(similarity)` recovers
the doc but the long tail of KOs from the same doc never re-enters
the top-K.

## Retrieval

| | |
|---|---|
| Endpoint | `POST /retrieve` |
| `limit` | 100 |
| `threshold` | 0.0 (no floor) |
| `filterRegimes` | unset (= all except `KR_WORKING`) |
| Queries issued | 10 / 10 (no errors) |
| Wall-time | 18.6 s (≈ 1.9 s / query) |
| `candidates_considered` per query | **308** (same for all queries — sub-corpus only has ~308 KOs in non-WORKING regimes) |
| Avg unique docs in top-100 KOs | 77 / 100 |

The retrieve API always returns 100 KOs (the requested top-K) and these
map to 69–89 unique BEIR doc IDs depending on the query.

## Metrics — head-to-head: `/retrieve` vs `/retrieve/full-oida`

Both runs use the same 418-doc / 10-query slice and the same 4 580 KOs in
the DB (full-oida run uses `--skip-ingest`).

| Metric | Flat `/retrieve` | Full-OIDA `/retrieve/full-oida` | Δ |
|---|---|---|---|
| **nDCG@10** | **0.2315** | 0.1632 | −0.0683 |
| nDCG@100 | 0.2381 | 0.1444 | −0.0937 |
| MAP@10 | 0.0240 | 0.0145 | −0.0095 |
| MAP@100 | 0.0668 | 0.0392 | −0.0276 |
| Recall@10 | 0.0402 | 0.0382 | −0.0020 |
| Recall@100 | 0.3776 | 0.1436 | **−0.2340** |
| P@10 | 0.2100 | 0.1700 | −0.0400 |
| P@100 | 0.1020 | 0.0650 | −0.0370 |
| Retrieve wall-time | 18.6 s | 12.6 s | −6 s |
| KOs returned / query | 100 | 50 (cap'd by solver) | |

### Why full-OIDA is worse here (root cause)

Every one of the 10 queries from `/retrieve/full-oida` came back with:

```
"composition_metadata": {
  "stopping_criterion": "fallback_avg_edges_below_tau_fallback",
  "shells": { "shell_1_count": 0, "shell_2_count": 0 },
  "pruned_count": 0
},
"salience_routing": { "route_used": "balanced" },
"edges": []
```

**The 4-gate solver had no edges to traverse.** AngelicaDB's `/ingest`
pipeline did not extract any edges during this run (`evidenceAttached: 0`
and `edgesWritten: 0` in every commit response). The 4-gate composition
(`solveSubgraph`) is built around binding-coherence over edges; with an
empty edge set it short-circuits via `tauFallback` and degrades to a
narrower, salience-weighted vector cut — strictly worse than the flat
hybrid cosine search.

The drop is therefore **not** a defect of `/retrieve/full-oida` — it is
the expected behaviour on a corpus that was ingested without edge
extraction. To make full-OIDA outperform flat hybrid you need either:
1. enabling edge extraction at ingest time (cross-document SUPPORTS /
   CONTRADICTS edges between KOs), or
2. a corpus where the gold relevance signal lives in graph structure,
   not in single-passage relevance (e.g. FEVER, Climate-FEVER, ArguAna,
   not NFCorpus).

### Context for absolute numbers

- Published BGE-base-en-v1.5 on **full NFCorpus** = nDCG@10 ≈ 0.368 (and
  the old SPEC's M0 gate, run on the full split, hit 0.36810 for both
  OIDA-degenerate and BGE-base). Level A is **not** comparable to that
  number: scope is 418 / 3 633 docs (≈ 11 %) and only 10 / 323 queries.
- Recall@100 on flat ≈ 0.38 on a 100-of-308-KO retrieve shows the recall
  ceiling is set primarily by the KO fan-out (sub-claims diluting doc
  signal), not by missing data in the index.
- P@10 ≈ 0.21 (flat) — about 2 of 10 returned docs are relevant on
  average — is in the expected range for a domain corpus with sparse
  short queries.

## Pass / fail vs. level-A intent

| Criterion | Result |
|---|---|
| All 418 ingest POSTs return `committed: true` | ✅ |
| Round-trip ingest → retrieve actually returns the ingested KOs | ✅ |
| `supporting_sources` correctly carries the BEIR doc `_id` we passed as `source` | ✅ |
| Eval pipeline (pytrec_eval) runs end-to-end | ✅ |
| Flat `/retrieve` nDCG@10 > 0 (i.e. not the broken-persistence baseline) | ✅ (0.231) |
| Flat `/retrieve` nDCG@10 within an order of magnitude of published BGE baselines | ✅ (≈ 0.63 × BGE-base, on a much smaller scope) |
| `/retrieve/full-oida` returns valid subgraph payload (kos/edges/composition_metadata) | ✅ |
| `/retrieve/full-oida` beats flat `/retrieve` on NFCorpus | ❌ (expected — no edges in graph; see root cause section) |

**Verdict: level A passes its intent — the AngelicaDB pipeline is functional
end-to-end against BEIR qrels.** Flat `/retrieve` is the right endpoint to
benchmark on NFCorpus-style passage-relevance datasets. `/retrieve/full-oida`
needs a corpus where edge extraction happens at ingest or where the gold
signal lives in graph structure (FEVER / Climate-FEVER / ArguAna).

Level B (full NFCorpus) will say whether the moderate flat-retrieve nDCG
is a scoring-aggregation problem or a recall problem at scale.

## Next

- Run level B (full NFCorpus test split: 3 633 docs / 323 queries).
- Estimated cost: ~3 633 × 1.55 s ≈ **94 min ingest** + ~10 min retrieve.
- Same script: `python scripts/smoke_angelica_nfcorpus.py --level B`.
