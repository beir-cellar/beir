# RESULTS — Mini-test ArguAna → AngelicaDB

**Date:** 2026-05-18
**Endpoint under test:** `https://angelicadb-kva.onrender.com` (v0.2.0)
**Harness:** `scripts/smoke_angelica_nfcorpus.py --dataset arguana --level mini`
**Raw outputs:**
- `data/results/mini_arguana_flat.json` — flat `/retrieve`
- `data/results/mini_arguana_fulloida.json` — `/retrieve/full-oida`

---

## Goal

1. **Sanity-check** that KO ingestion on AngelicaDB still works for a different
   dataset shape (long argumentative passages, not short medical abstracts).
2. **Re-probe `/retrieve/full-oida`** on a dataset that semantically *should*
   suit it (counter-argument retrieval = epistemic structure native to OIDA).

## Dataset profile — ArguAna vs NFCorpus

| | ArguAna | NFCorpus (level A) |
|---|---|---|
| Task | Find best **counter-argument** to a given argument | Find passages relevant to a medical query |
| Corpus size (full) | 8 674 | 3 633 |
| Test queries (full) | 1 406 | 323 |
| Query length (mean / max) | 193.6 / 868 words | 3.3 / 11 words |
| Doc length (typ.) | ~1 800 chars | ~700 chars |
| Gold docs per query | **always 1** (the specific counter-arg) | 1 – 100+ |

## Protocol (mini level)

- 3 random queries from test split + their 3 gold docs + 15 distractor docs
  → **18 unique docs** in scope (seed = 42)
- `POST /ingest` (commit, sourceType=MANUAL, source=BEIR _id)
- `POST /retrieve` and `POST /retrieve/full-oida` (limit=100)
- Aggregate KOs → BEIR docs via `supporting_sources` (max similarity / kge_score)
- pytrec_eval at K=10, 100

## KO write — sanity check

| | Value |
|---|---|
| Docs sent | 18 |
| `committed: true` | **18 / 18** |
| Total KOs created | **177** |
| Mean KOs per doc | 9.8 |
| Network errors | 0 |
| Wall-time | 26.5 s (≈ 1.5 s / doc, same as NFCorpus) |
| Round-trip check (retrieve finds ingested KOs?) | ✅ all 3 queries return 100 KOs |
| `supporting_sources` correctly carries BEIR _id | ✅ |

**→ KO write is functioning correctly.** Nothing is broken upstream; the
EIP pipeline accepts the longer ArguAna passages, decomposes them into ~10
sub-claim KOs per doc, and persists them — identical behaviour to NFCorpus.

## Head-to-head — flat vs full-oida

Same 18-doc / 3-query scope, same 177 KOs in DB.

| Metric | Flat `/retrieve` | Full-OIDA `/retrieve/full-oida` | Δ |
|---|---|---|---|
| **nDCG@10** | **0.5000** | 0.3290 | −0.1710 |
| nDCG@100 | 0.5000 | 0.3290 | −0.1710 |
| MAP@10 | 0.4444 | 0.2222 | −0.2222 |
| Recall@10 | 0.6667 | 0.6667 | 0 |
| Recall@100 | 0.6667 | 0.6667 | 0 |
| P@10 | 0.0667 | 0.0667 | 0 |
| Wall-time | 2.5 s | n/a (cached after first run) | |
| KOs returned per query | 100 | 50 (solver cap) | |

Identical recall — full-OIDA **finds the same gold docs** but ranks them
worse (loses MAP and nDCG, keeps Recall). Across all 3 queries:

```
edges: []
stopping_criterion: "fallback_avg_edges_below_tau_fallback"
shells: { shell_1_count: 0, shell_2_count: 0 }
route_used: "balanced"
```

**Same exact failure mode as NFCorpus** — the solver never expands beyond
the seed set because there are no edges in the DB. The persistence path
`/ingest` does not generate edges (every response: `edgesWritten: 0`,
`evidenceAttached: 0`), and the auto-bond trigger has not run a backfill
on the newly written KOs.

## Why this matters for the L2 thesis

ArguAna was supposed to be the dataset where full-OIDA shines: counter-arg
retrieval is intrinsically about **epistemic relationships** (one argument
opposes another). The gold-relevance signal is graph-structural.

But full-OIDA **cannot exploit that** on the current ingestion path,
because the EIP doesn't extract inter-doc edges and no backfill step is
documented as part of the standard flow. So even on the dataset designed
to play to its strengths, full-OIDA collapses to "salience re-ranking
over flat similarity" — which is strictly worse than the flat hybrid on
both NFCorpus and ArguAna.

## Verdict

| Question | Answer |
|---|---|
| Does KO ingestion work? | ✅ Yes, 18/18 committed, 177 KOs, retrievable |
| Does flat `/retrieve` work? | ✅ Yes, nDCG@10 = 0.500 on mini scope, recall 0.667 |
| Does `/retrieve/full-oida` work? | ⚠ Mechanically yes (returns valid subgraph), substantively no — solver short-circuits because no edges |
| Is the issue dataset-specific (NFCorpus vs ArguAna)? | ❌ No — same `stopping_criterion` on both, regardless of domain |
| Is the issue the `/ingest` pipeline? | ✅ Yes — it never writes edges. To unlock full-OIDA we need a separate edge-population step |

## Next steps (proposed)

1. **Server-side fix (AngelicaDB)**: run `batch-detect-edges` (or equivalent
   auto-bond backfill) on the `default` project so the ~5 000 KO already
   ingested across NFCorpus + ArguAna get edges. Then re-run **mini ArguAna
   full-oida** with `--skip-ingest` — that isolates "only the edge gap" from
   "the formula too".
2. **Decision point**: only after step 1 do we know whether full-OIDA's
   `kge_score` (with binding component activated) is competitive on a real
   L2 dataset. If yes → full ArguAna run (level A or full). If no → we
   document that full-OIDA needs either curated KB scale or a different
   scoring weight schedule for BEIR-style benchmarks.
