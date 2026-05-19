"""Smoke harness: ingest NFCorpus into AngelicaDB and evaluate /retrieve vs qrels.

Levels:
  A — micro: 10 test queries + their gold docs + 50 random distractors (~100-200 docs total)
  B — full: entire NFCorpus test split (3,633 docs / 323 queries)

Each BEIR doc is sent as a single POST /ingest commit (NOT /ingest/batch).
sourceId = BEIR doc _id  → returned later in /retrieve as supporting_sources[i].

Evaluation:
  - For each query, call POST /retrieve with limit=100, threshold=0.0
  - Aggregate retrieved KOs by sourceId (= BEIR doc _id), taking max similarity
  - Evaluate the resulting doc ranking against BEIR qrels using pytrec_eval
  - Metrics: nDCG@10, nDCG@100, Recall@10, Recall@100, MAP@10, P@10

Usage:
    python scripts/smoke_angelica_nfcorpus.py --level A [--queries 10] [--distractors 50]
    python scripts/smoke_angelica_nfcorpus.py --level B
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # for vendored `beir` package

BASE = "https://angelicadb-kva.onrender.com"
DATA_DIR = REPO_ROOT / "data" / "beir_raw"
RESULTS_DIR = REPO_ROOT / "data" / "results"
DATASET_URL_TEMPLATE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke")


def load_dataset(name: str, split: str = "test"):
    from beir.datasets.data_loader import GenericDataLoader
    from beir.util import download_and_unzip
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = DATASET_URL_TEMPLATE.format(name=name)
    data_path = download_and_unzip(url, str(DATA_DIR))
    if not os.path.isdir(data_path):
        raise FileNotFoundError(data_path)
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split=split)
    return corpus, queries, qrels


def subsample(corpus, queries, qrels, n_queries: int, n_distractors: int, seed: int = 42):
    rng = random.Random(seed)
    qids_with_qrels = [q for q in queries if q in qrels and qrels[q]]
    chosen_qids = rng.sample(qids_with_qrels, min(n_queries, len(qids_with_qrels)))
    gold_doc_ids = set()
    for q in chosen_qids:
        for d, rel in qrels[q].items():
            if rel > 0:
                gold_doc_ids.add(d)
    pool = [d for d in corpus if d not in gold_doc_ids]
    distractor_ids = set(rng.sample(pool, min(n_distractors, len(pool))))
    keep_doc_ids = gold_doc_ids | distractor_ids
    sub_corpus = {d: corpus[d] for d in keep_doc_ids}
    sub_queries = {q: queries[q] for q in chosen_qids}
    sub_qrels = {q: {d: r for d, r in qrels[q].items() if d in keep_doc_ids} for q in chosen_qids}
    return sub_corpus, sub_queries, sub_qrels


def format_doc(doc: dict) -> str:
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return f"{title}. {text}".strip(". ").strip() if title else text


def ingest_doc(session, key: str, doc_id: str, text: str, timeout: int = 60) -> dict:
    payload = {
        "text": text,
        "source": doc_id,
        "sourceType": "MANUAL",
        "mode": "commit",
        "sensitivity": 1,
        "visibleTo": [],
    }
    r = session.post(
        f"{BASE}/ingest",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def retrieve_query(session, key: str, query: str, top_k: int = 100, threshold: float = 0.0,
                   endpoint: str = "flat", timeout: int = 120) -> dict:
    if endpoint == "full-oida":
        path = "/retrieve/full-oida"
        payload = {
            "query": query,
            "limit": top_k,
            "include": ["edges", "contradictions", "salience_metadata",
                        "composition_metadata", "dialectic_resolutions"],
        }
    else:
        path = "/retrieve"
        payload = {"query": query, "limit": top_k, "threshold": threshold}
    r = session.post(
        f"{BASE}{path}",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def extract_kos(retrieve_response: dict, endpoint: str) -> list[dict]:
    if endpoint == "full-oida":
        return retrieve_response.get("subgraph", {}).get("kos", [])
    return retrieve_response.get("results", [])


def aggregate_kos_to_docs(kos: list[dict]) -> dict[str, float]:
    """Map retrieved KOs back to BEIR doc IDs via supporting_sources, max similarity per doc.

    For /retrieve/full-oida, prefer `kge_score` if present, else fall back to `similarity`.
    For /retrieve, only `similarity` is present.
    """
    by_doc: dict[str, float] = {}
    for ko in kos:
        score = ko.get("kge_score")
        if score is None:
            score = ko.get("regime_adjusted_score")
        if score is None:
            score = ko.get("similarity", 0.0)
        score = float(score or 0.0)
        for src in ko.get("supporting_sources", []):
            if src and (src not in by_doc or score > by_doc[src]):
                by_doc[src] = score
    return by_doc


def evaluate(qrels: dict, run: dict, k_values=(10, 100)) -> dict:
    import pytrec_eval
    metrics = {f"ndcg_cut.{','.join(str(k) for k in k_values)}",
               f"map_cut.{','.join(str(k) for k in k_values)}",
               f"recall.{','.join(str(k) for k in k_values)}",
               f"P.{','.join(str(k) for k in k_values)}"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    per_query = evaluator.evaluate(run)
    out = {}
    for measure_set in metrics:
        for k in k_values:
            mname = f"{measure_set.split('.')[0]}_{k}"
            mkey = f"{measure_set.split('.')[0]}{'_cut_' if 'cut' in measure_set else '_'}{k}"
            # pytrec_eval keys: ndcg_cut_10, map_cut_10, recall_10, P_10
            if measure_set.startswith("ndcg"):
                mkey = f"ndcg_cut_{k}"; label = f"nDCG@{k}"
            elif measure_set.startswith("map"):
                mkey = f"map_cut_{k}"; label = f"MAP@{k}"
            elif measure_set.startswith("recall"):
                mkey = f"recall_{k}"; label = f"Recall@{k}"
            elif measure_set.startswith("P"):
                mkey = f"P_{k}"; label = f"P@{k}"
            vals = [per_query[q][mkey] for q in per_query]
            out[label] = sum(vals) / len(vals) if vals else 0.0
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="nfcorpus", choices=["nfcorpus", "arguana"])
    p.add_argument("--split", default="test")
    p.add_argument("--level", choices=["mini", "A", "B"], required=True)
    p.add_argument("--queries", type=int, default=10, help="(level mini/A only) test queries to sample")
    p.add_argument("--distractors", type=int, default=50, help="(level mini/A only) random distractor docs")
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--retrieve-endpoint", choices=["flat", "full-oida"], default="flat",
                   help="flat → POST /retrieve; full-oida → POST /retrieve/full-oida")
    p.add_argument("--skip-ingest", action="store_true",
                   help="Skip ingestion (assume KOs are already in the DB from a prior run with the same seed)")
    p.add_argument("--out", default=None, help="JSON output path (default: data/results/smoke_<level>_<ts>.json)")
    args = p.parse_args()

    key = os.environ.get("ADB_ADMIN_KEY")
    if not key:
        log.error("ADB_ADMIN_KEY not set in env")
        return 2

    log.info("Loading %s (split=%s)", args.dataset, args.split)
    corpus, queries, qrels = load_dataset(args.dataset, split=args.split)
    log.info("Full: %d docs | %d queries | %d qrels", len(corpus), len(queries), len(qrels))

    if args.level in ("mini", "A"):
        corpus, queries, qrels = subsample(corpus, queries, qrels, args.queries, args.distractors, seed=args.seed)
        log.info("Subsampled: %d docs | %d queries | %d qrels", len(corpus), len(queries), len(qrels))

    session = requests.Session()

    # Ingest
    ingest_log: list[dict] = []
    ingest_errors = 0
    if args.skip_ingest:
        t_ingest = 0.0
        committed = 0
        total_kos = 0
        log.info("Skipping ingest (--skip-ingest); reusing KOs already in the DB")
    else:
        t0 = time.time()
        log.info("Ingesting %d docs via POST /ingest commit (sourceType=MANUAL → Capture mode)", len(corpus))
        for doc_id in tqdm(list(corpus.keys()), desc="ingest"):
            text = format_doc(corpus[doc_id])
            if not text.strip():
                continue
            try:
                resp = ingest_doc(session, key, doc_id, text)
                ingest_log.append({
                    "doc_id": doc_id,
                    "committed": resp.get("committed", False),
                    "kosCreated": resp.get("kosCreated", 0),
                    "candidatesTotal": resp.get("candidatesTotal", 0),
                })
            except Exception as e:
                ingest_errors += 1
                ingest_log.append({"doc_id": doc_id, "error": str(e)[:200]})
        t_ingest = time.time() - t0
        committed = sum(1 for x in ingest_log if x.get("committed"))
        total_kos = sum(x.get("kosCreated", 0) for x in ingest_log)
        log.info("Ingest done in %.1fs: committed=%d / %d, total KOs created=%d, errors=%d",
                 t_ingest, committed, len(corpus), total_kos, ingest_errors)

    # Retrieve
    endpoint_path = "/retrieve/full-oida" if args.retrieve_endpoint == "full-oida" else "/retrieve"
    log.info("Retrieving (endpoint=%s top_k=%d threshold=%.2f) for %d queries",
             endpoint_path, args.top_k, args.threshold, len(queries))
    t0 = time.time()
    run: dict[str, dict[str, float]] = {}
    retrieve_log: list[dict] = []
    for qid in tqdm(list(queries.keys()), desc="retrieve"):
        qtext = queries[qid]
        try:
            resp = retrieve_query(session, key, qtext,
                                  top_k=args.top_k, threshold=args.threshold,
                                  endpoint=args.retrieve_endpoint)
            kos = extract_kos(resp, args.retrieve_endpoint)
            by_doc = aggregate_kos_to_docs(kos)
            run[qid] = by_doc
            log_entry = {
                "qid": qid,
                "n_kos_returned": len(kos),
                "n_unique_docs": len(by_doc),
            }
            if args.retrieve_endpoint == "full-oida":
                sub = resp.get("subgraph", {})
                cm = sub.get("composition_metadata", {})
                sr = sub.get("salience_routing", {})
                log_entry.update({
                    "n_edges": len(sub.get("edges", [])),
                    "stopping_criterion": cm.get("stopping_criterion"),
                    "shells": cm.get("shells"),
                    "pruned_count": cm.get("pruned_count"),
                    "route_used": sr.get("route_used"),
                    "elapsed_ms": resp.get("metadata", {}).get("elapsed_ms"),
                })
            else:
                log_entry["candidates_considered"] = resp.get("metadata", {}).get("candidates_considered", 0)
            retrieve_log.append(log_entry)
        except Exception as e:
            run[qid] = {}
            retrieve_log.append({"qid": qid, "error": str(e)[:200]})
    t_retrieve = time.time() - t0
    log.info("Retrieve done in %.1fs", t_retrieve)

    # Evaluate
    int_qrels = {q: {d: int(r) for d, r in v.items() if r > 0} for q, v in qrels.items()}
    int_qrels = {q: v for q, v in int_qrels.items() if v}
    metrics = evaluate(int_qrels, run, k_values=(10, 100)) if run and int_qrels else {}

    out = {
        "level": args.level,
        "dataset": args.dataset,
        "split": args.split,
        "n_docs_in_scope": len(corpus),
        "n_queries_in_scope": len(queries),
        "n_qrels_in_scope": len(int_qrels),
        "ingest": {
            "endpoint": "POST /ingest",
            "sourceType": "MANUAL",
            "mode": "commit",
            "duration_sec": round(t_ingest, 2),
            "committed_count": committed,
            "total_kos_created": total_kos,
            "errors": ingest_errors,
        },
        "retrieve": {
            "endpoint": f"POST {endpoint_path}",
            "top_k": args.top_k,
            "threshold": args.threshold,
            "duration_sec": round(t_retrieve, 2),
            "queries_with_results": sum(1 for q in run if run[q]),
            "queries_empty": sum(1 for q in run if not run[q]),
        },
        "metrics": metrics,
        "ingest_log_sample": ingest_log[:20],
        "retrieve_log_sample": retrieve_log[:20],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"smoke_{args.level}_{int(time.time())}.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
