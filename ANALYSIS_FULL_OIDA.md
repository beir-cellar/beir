# Analysis — Why `/retrieve/full-oida` underperformed on Level A

**Date:** 2026-05-18
**Question:** Cosa è andato storto nell'ingestion per il path full-OIDA, e con la configurazione corretta di KO + edge il risultato sarebbe stato diverso?

## TL;DR

Due cause concorrenti, entrambe strutturali — non bug, ma misalignment tra
shape del corpus ingerito e ipotesi del solver 4-gate:

1. **Zero edge nel DB per i 4 580 KO di NFCorpus** → `solveSubgraph` non
   può espandere, si ferma sui seed e cade su `fallback_avg_edges_below_tau_fallback`.
2. **`kge_score` re-ranks contro la similarity quando la similarity è
   debole** — su NFCorpus le top similarity sono 0.18–0.24 (corpus
   semanticamente disperso); il peso 0.30 sulla salience domina e
   l'ordine prodotto è quasi ortogonale al ground-truth BEIR.

Anche se avessimo avuto gli edge "giusti" (auto-bond attivo + backfill),
**probabilmente full-OIDA non avrebbe battuto flat su NFCorpus.** Per
vincere serve un dataset dove il segnale gold vive nella struttura del
grafo (ArguAna / FEVER), non in NFCorpus dove vive nella relevance
puntuale per passaggio.

---

## 1. Evidenza empirica dal payload reale

Query `"avocados"` (qid `PLAIN-634`), 10 top KO dal subgraph
`/retrieve/full-oida` sui dati ingeriti in Level A:

```
kge_score:    0.553  0.522  0.497  0.495  0.493  0.490  0.489  0.485  0.476  0.475
similarity:   0.187  0.186  0.195  0.238  0.188  0.183  0.179  0.222  0.208  0.237
ron_subgraph: seed   seed   seed   seed   seed   seed   seed   seed   seed   seed
edges:        []
```

E le `composition_metadata`:

```json
{
  "expansion_path": [],
  "shells": { "shell_1_count": 0, "shell_2_count": 0 },
  "pruned_count": 0,
  "stopping_criterion": "fallback_avg_edges_below_tau_fallback"
}
```

`salience_routing`:

```json
{
  "route_used": "balanced",
  "lane_outputs": { "solver_dominant_signal": "salience_fallback",
                    "solver_composition_score": 0 }
}
```

Tre osservazioni dure:

- **Tutti i 10 KO sono `seed`** — niente è stato espanso oltre il seed
  set. Stage 2/3/4 della pipeline girano ma sono no-op.
- **Top KO** (`kge_score = 0.553`) ha `similarity = 0.187`. Il claim è:
  *"Studies using germ-free mice confirmed a critical role for dietary
  choline and gut flora in TMAO production..."* — niente avocado. Lo
  ha promosso la salience (`s_epi ≈ 0.81`, `s_act ≈ 0.73`), non la
  rilevanza alla query.
- **Similarity NON ordinate**: il KO con similarity più alta (0.238) è
  in posizione 4 secondo `kge_score`. Il re-ranking inverte
  attivamente l'ordine prodotto dalla cosine.

## 2. Perché non ci sono edge nel DB

Il pipeline EIP che gira su `/ingest` **non fa cross-document edge
extraction**. Ogni risposta in Level A aveva `evidenceAttached: 0` e
`edgesWritten: 0`. È coerente con la doc: l'EIP estrae candidati,
classifica, e scrive KO + context frame — gli edge sono materia di un
processo separato.

I tre meccanismi documentati per popolare edge:

| Meccanismo | Stato sul nostro DB |
|---|---|
| Trigger SQL `trg_ko_auto_bond` (auto-edge `CORROBORATES` se cosine > 0.60 a insert-time) | Installato **dopo** il nostro ingest di Level A → i 4 580 KO non hanno passato per il trigger |
| Batch script `scripts/batch-detect-edges.ts` lato AngelicaDB | Non eseguito sul project |
| Post-write hook EIP (claim contraddittori inseriti insieme → edge `CONTRADICTS`) | Non applicabile: ogni doc NFCorpus è stato ingerito in chiamata separata, niente cross-signal contradiction detection |

Anche se il trigger fosse stato attivo prima del run, su NFCorpus non
avremmo visto molti edge: la soglia 0.60 di cosine è alta per un corpus
di abstract medici eterogenei. La distribuzione delle similarity nelle
nostre query (0.18–0.24 top) suggerisce che le similarity *inter-KO*
sono in media ancora più basse.

## 3. Perché `kge_score` peggiora il ranking su questa workload

Formula documentata (dalla reference inviata):

```
kge_score = 0.35 · similarity
          + 0.20 · regime_adjusted_score
          + 0.30 · salience  (mix di s_epi / s_act / gravity)
          + 0.15 · binding   (coerenza degli edge incidenti)
```

Con `binding = 0` (no edges) restano tre componenti. Su NFCorpus:

- **similarity media top-10 ≈ 0.20** — segnale debole
- **regime_adjusted ≈ similarity** (la gravity moltiplica 1.0 o ~1.4
  per `KR_EVENT` / `KR_CANONICAL`) — non aggiunge informazione utile per
  la rilevanza al query
- **salience ≈ 0.70** su quasi tutti i KO medici — DOMINA il punteggio
  con il peso 0.30, ma è invariante alla query (è una property del KO,
  non della relazione query↔KO)

Risultato netto: il `kge_score` collassa in un intervallo stretto
(0.475–0.553 sui top 10) dominato dalla salience invariante; la
similarity (l'unico segnale query-dipendente) entra al 35 % di peso ma
parte da magnitudini troppo basse per sopravvivere all'aggregazione.

Questo è coerente con il `solver_dominant_signal: "salience_fallback"` —
il solver stesso si auto-diagnostica e dice "ho deciso per salience
perché nessun altro segnale era affidabile".

## 4. Controfattuale — e se avessimo avuto edge "giusti"?

Tre scenari diversi, in ordine crescente di impatto:

### Scenario A — Auto-bond + backfill sui 4 580 KO esistenti

Riapro lo stesso run dopo aver lanciato `batch-detect-edges` sul
project. Edge `CORROBORATES` si creerebbero tra KO con cosine > 0.60.

| | Previsione |
|---|---|
| Edge totali generati | bassi, forse 100–300 sui 4 580 KO (corpus eterogeneo) |
| KO `seed` con almeno 1 edge | < 30 % stimato |
| Effetto su `kge_score` | binding aggiunge fino a +0.15 a chi è nel cluster; cambia l'ordine *all'interno* dei seed ma non porta nuovi doc in top-10 |
| nDCG@10 atteso | da 0.163 → ~0.18–0.20 (recupera in parte la salience-only baseline) |
| Vince contro flat (0.231)? | **No**, ancora sotto |

L'auto-bond CORROBORATES su un corpus passage-only crea "guilt by
association" — utile per coerenza, neutro per rilevanza.

### Scenario B — Stesso pipeline ma su dataset L2 (ArguAna / Climate-FEVER)

ArguAna è "counter-argument retrieval": per ogni argomento, trovare il
contro-argomento. La ground truth ha struttura **inerentemente
contraddittoria**.

| | Previsione |
|---|---|
| Edge `CONTRADICTS` generati | dipende dall'EIP; potenzialmente alto su coppie native arg/counter-arg |
| Edge `SUPPORTS` | massivi via auto-bond (claim simili dentro la stessa stance) |
| Effetto su salience | `s_epi` differenzia argomenti forti vs. deboli → segnale informativo |
| Effetto su routing | route_used dovrebbe diventare `epistemic_first` (non balanced) |
| nDCG@10 vs flat | **full-OIDA dovrebbe vincere** per design — è esattamente il caso d'uso L2 del SPEC originale |

### Scenario C — Corpus tipo demo della reference (310 KO con high similarity)

Nell'esempio di reference la top similarity era 0.63 (vs i nostri 0.20).
Con similarity alte:

- la componente 0.35 · similarity contribuisce ~0.22 al `kge_score`
  invece dei nostri 0.07
- la salience smette di dominare in modo decisivo
- la rilevanza al query torna ad essere il segnale principale

È il regime per cui il `kge_score` è tarato: knowledge base curata,
piccola, con contenuto strettamente legato al dominio.

## 5. Risposta sintetica alle due domande

**Q1: Quali criticità nell'ingestion per `/retrieve/full-oida`?**

a. **Niente edge backfill dopo l'ingest**: `/ingest` scrive solo KO +
   context frame, non genera edge. Il trigger auto-bond era assente o
   installato dopo, e nessun batch detection è stato lanciato.
b. **Sub-task implicito**: per ogni progetto destinato a uso
   full-OIDA serve un secondo step (`batch-detect-edges` o un hook
   server-side) — non documentato come obbligatorio nell'API doc che
   abbiamo seguito.
c. **Corpus mismatch**: gli atomi-KO prodotti dalla decomposizione EIP
   da passaggi NFCorpus sono eterogenei al loro interno; le similarity
   inter-KO sono basse → poche soglie di auto-bond superate.

**Q2: Con la config corretta (KO + edge giusti) il risultato sarebbe stato diverso?**

- **Su NFCorpus**: marginalmente diverso, ancora perdente vs flat.
  L'aggiunta di edge sposta il binding score da 0 a qualche frazione
  positiva, ma il `kge_score` resta dominato dalla salience (che è
  invariante alla query) e il segnale similarity (l'unico
  query-dipendente) è troppo debole su questo corpus.
- **Su un dataset L2 (ArguAna / Climate-FEVER)**: sì, drasticamente
  diverso. Lì la struttura del grafo *è* il segnale gold; il binding
  e la routing epistemica avrebbero molto da dire. Ed è esattamente la
  separazione L1/L2/L3 che il vecchio SPEC.md prevedeva (sezione 2.2).

## 6. Raccomandazioni operative

1. **Prima di rifare /retrieve/full-oida su NFCorpus**: lanciare
   `batch-detect-edges` lato AngelicaDB sul `default` project per
   popolare gli edge sui 4 580 KO già ingeriti, e ri-runnare il test
   con `--skip-ingest --retrieve-endpoint full-oida`. Questo isola
   "è solo la mancanza di edge?" da "è anche la formula?".

2. **Per benchmarkare seriamente full-OIDA**: ingerire ArguAna
   (~8 700 docs / 1 400 queries) o un sub-sample di Climate-FEVER.
   Stima di costo simile a NFCorpus Level B.

3. **Documentazione AngelicaDB**: l'API doc consultata non menziona
   che `/retrieve/full-oida` ha senso solo dopo un edge-population
   step. Vale la pena aggiungerlo nelle "Note operative" §5.
