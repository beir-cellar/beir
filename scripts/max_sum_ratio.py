"""Compare max-aggregation vs sum-aggregation of KO scores → BEIR doc ranking.

For each query in the mini-ArguAna scope (seed=42), call /retrieve and
/retrieve/full-oida, then aggregate KO similarities to docs in three ways:
  - max (current harness default)
  - sum (alternative)
  - ratio max/sum (lower = score spread over many KOs; closer to 1 = dominated by one KO)

Also reports edges-written sanity check per query.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_angelica_nfcorpus import (  # type: ignore
    BASE, load_dataset, subsample, retrieve_query, extract_kos,
)
import requests
import pytrec_eval


def aggregate(kos, mode: str) -> dict[str, float]:
    by_doc: dict[str, list[float]] = defaultdict(list)
    for ko in kos:
        s = ko.get("kge_score")
        if s is None:
            s = ko.get("regime_adjusted_score")
        if s is None:
            s = ko.get("similarity", 0.0)
        s = float(s or 0.0)
        for src in ko.get("supporting_sources", []):
            if src:
                by_doc[src].append(s)
    if mode == "max":
        return {d: max(v) for d, v in by_doc.items()}
    if mode == "sum":
        return {d: sum(v) for d, v in by_doc.items()}
    raise ValueError(mode)


def eval_run(qrels, run) -> dict:
    ev = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut.10", "recall.10,100", "map_cut.10", "P.10"}
    )
    per = ev.evaluate(run)
    if not per:
        return {}
    keys = {"ndcg_cut_10": "nDCG@10", "recall_10": "R@10",
            "recall_100": "R@100", "map_cut_10": "MAP@10", "P_10": "P@10"}
    out = {}
    for k, label in keys.items():
        vals = [per[q][k] for q in per]
        out[label] = round(sum(vals) / len(vals), 4)
    return out


def main():
    key = os.environ["ADB_ADMIN_KEY"]
    corpus, queries, qrels = load_dataset("arguana", split="test")
    corpus, queries, qrels = subsample(corpus, queries, qrels, 3, 15, seed=42)
    int_qrels = {q: {d: int(r) for d, r in v.items() if r > 0} for q, v in qrels.items()}
    int_qrels = {q: v for q, v in int_qrels.items() if v}

    session = requests.Session()
    report = {}
    for endpoint in ("flat", "full-oida"):
        run_max: dict[str, dict[str, float]] = {}
        run_sum: dict[str, dict[str, float]] = {}
        per_query_diag = []
        for qid, qtext in queries.items():
            resp = retrieve_query(session, key, qtext, top_k=100,
                                  threshold=0.0, endpoint=endpoint)
            kos = extract_kos(resp, endpoint)
            run_max[qid] = aggregate(kos, "max")
            run_sum[qid] = aggregate(kos, "sum")
            ratios = []
            for doc, score_sum in run_sum[qid].items():
                if score_sum > 0:
                    ratios.append(run_max[qid][doc] / score_sum)
            mean_ratio = sum(ratios) / len(ratios) if ratios else 0.0
            edges = len(resp.get("subgraph", {}).get("edges", [])) if endpoint == "full-oida" else None
            per_query_diag.append({
                "qid": qid,
                "n_kos": len(kos),
                "n_unique_docs": len(run_max[qid]),
                "mean_max_over_sum_ratio": round(mean_ratio, 4),
                "edges": edges,
                "stopping_criterion": (
                    resp.get("subgraph", {}).get("composition_metadata", {}).get("stopping_criterion")
                    if endpoint == "full-oida" else None
                ),
            })
        report[endpoint] = {
            "metrics_max_agg": eval_run(int_qrels, run_max),
            "metrics_sum_agg": eval_run(int_qrels, run_sum),
            "per_query": per_query_diag,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
