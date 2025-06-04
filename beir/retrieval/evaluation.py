from __future__ import annotations

import glob
import logging

import pytrec_eval

from .custom_metrics import hole, mrr, recall_cap, top_k_accuracy
from .search.base import BaseSearch

logger = logging.getLogger(__name__)


class EvaluateRetrieval:
    def __init__(
        self,
        retriever: BaseSearch = None,
        k_values: list[int] = [1, 3, 5, 10, 100, 1000],
        score_function: str | None = "cos_sim",
    ):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.retriever = retriever
        self.score_function = score_function

    def retrieve(
        self, corpus: dict[str, dict[str, str]], queries: dict[str, str], **kwargs
    ) -> dict[str, dict[str, float]]:
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        return self.retriever.search(corpus, queries, self.top_k, self.score_function, **kwargs)

    def encode_and_retrieve(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        encode_output_path: str = "./embeddings/",
        overwrite: bool = False,
        query_filename: str = "queries.pkl",
        corpus_filename: str = "corpus.*.pkl",
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """Encode the corpus and queries, save them and then retrieve results."""
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        # Encode corpus and queries using encode function and save them to disk
        self.retriever.encode(
            corpus,
            queries,
            encode_output_path=encode_output_path,
            overwrite=overwrite,
            query_filename=query_filename,
            corpus_filename=corpus_filename,
            **kwargs,
        )

        # Retrieve results using faiss-cpu search function using the saved embeddings
        query_embeddings_file = f"{encode_output_path}/{query_filename}"
        corpus_embeddings_files = glob.glob(f"{encode_output_path}/{corpus_filename}")

        return self.retriever.search_from_files(
            query_embeddings_file=query_embeddings_file,
            corpus_embeddings_files=corpus_embeddings_files,
            top_k=self.top_k,
            **kwargs,
        )

    @staticmethod
    def evaluate(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        ignore_identical_ids: bool = True,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
        if ignore_identical_ids:
            logger.info(
                "For evaluation, we ignore identical query and document ids (default), please explicitly set ``ignore_identical_ids=False`` to ignore this."
            )
            popped = []
            for qid, rels in results.items():
                for pid in list(rels):
                    if qid == pid:
                        results[qid].pop(pid)
                        popped.append(pid)

        ndcg = {}
        _map = {}
        recall = {}
        precision = {}

        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0

        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

        for eval in [ndcg, _map, recall, precision]:
            logger.info("\n")
            for k in eval.keys():
                logger.info(f"{k}: {eval[k]:.4f}")

        return ndcg, _map, recall, precision

    @staticmethod
    def evaluate_custom(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
        k_values: list[int],
        metric: str,
    ) -> tuple[dict[str, float]]:
        if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
            return mrr(qrels, results, k_values)

        elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
            return recall_cap(qrels, results, k_values)

        elif metric.lower() in ["hole", "hole@k"]:
            return hole(qrels, results, k_values)

        elif metric.lower() in [
            "acc",
            "top_k_acc",
            "accuracy",
            "accuracy@k",
            "top_k_accuracy",
        ]:
            return top_k_accuracy(qrels, results, k_values)
