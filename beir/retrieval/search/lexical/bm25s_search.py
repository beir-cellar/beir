"""
This code snippet is a modified version of BM25s: https://github.com/xhluca/bm25s/blob/main/examples/retrieve_nq_with_batching.py
This uses BM25s to retrieve the top-k results for Natural Questions (NQ) dataset in batches to reduce memory usage.

To run this example, you need to install the following dependencies:

```bash
pip install bm25s[core]
```
"""

from __future__ import annotations

import logging

import bm25s
import Stemmer
from tqdm import tqdm

from .. import BaseSearch

logger = logging.getLogger(__name__)


class BM25SSearch(BaseSearch):
    def __init__(
        self,
        batch_size: int = 20,
        method: str = "bm25",
        method_kwargs: dict = {},
        stemmer: str = "english",
        stopwords: str = "english",
        backend: str = "numba",
        index_dir: str = None,
    ):
        self.batch_size = batch_size
        self.index_dir = index_dir
        self.stopwords = stopwords
        self.tokenizer = bm25s.tokenization.Tokenizer(stopwords=stopwords, stemmer=Stemmer.Stemmer(stemmer))
        self.retriever = bm25s.BM25(method=method, backend=backend, **method_kwargs)
        self.timer = bm25s.utils.benchmark.Timer("[BM25S]")
        self.results = {}

    def _postprocess_results_for_eval(self, results, scores, query_ids):
        """
        Given the queried results and scores output by BM25S, postprocess them
        to be compatible with BEIR evaluation functions.
        query_ids is a list of query ids in the same order as the results.
        """

        results_record = [
            {"id": qid, "hits": results[i], "scores": list(scores[i])} for i, qid in enumerate(query_ids)
        ]

        result_dict_for_eval = {
            res["id"]: {doc["id"]: float(score) for doc, score in zip(res["hits"], res["scores"])}
            for res in results_record
        }

        return result_dict_for_eval

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        *args,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        mem_use = bm25s.utils.benchmark.get_max_memory_usage()
        logging.info(f"Initial memory usage: {mem_use:.2f} GB")

        # corpus indexing
        corpus_records = [{"id": k, "title": v["title"], "text": v["text"]} for k, v in corpus.items()]
        corpus_lst = [r["title"] + " " + r["text"] for r in corpus_records]

        logging.info(f"Tokenizing {len(corpus_lst)} documents using BM25s...")
        corpus_tokenized = self.tokenizer.tokenize(corpus_lst, update_vocab=True)

        # index the corpus
        self.retriever.corpus = corpus_records
        self.retriever.index(corpus_tokenized)

        # Save the index
        # if self.index_dir:
        # self.retriever.save(self.index_dir)
        # self.tokenizer.save_vocab(self.index_dir)
        # self.tokenizer.save_stopwords(self.index_dir)
        # logging.info(f"Saved the index to {self.index_dir}")

        # retrieve the top-k results
        num_docs = self.retriever.scores["num_docs"]

        mem_use = bm25s.utils.benchmark.get_max_memory_usage()
        print(f"Memory usage after loading the index: {mem_use:.2f} GB")

        # retrieve results from BM25
        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in query_ids]
        logging.info(f"Tokenizing {len(queries)} queries using BM25s...")
        queries_tokenized = self.tokenizer.tokenize(queries)

        logging.info("Retrieving the top-k results...")
        t = self.timer.start("Retrieving")

        batches = []

        for i in tqdm(range(0, len(query_ids), self.batch_size)):
            batches.append(
                self.retriever.retrieve(
                    queries_tokenized[i : i + self.batch_size],
                    k=top_k,
                )
            )

            # reload the corpus and scores to free up memory
            self.retriever.load_scores(save_dir=self.index_dir, mmap=True, num_docs=num_docs)
            if isinstance(self.retriever.corpus, bm25s.utils.corpus.JsonlCorpus):
                self.retriever.corpus.load()

        results = bm25s.Results.merge(batches)
        self.results = self._postprocess_results_for_eval(results.documents, results.scores, query_ids)

        self.timer.stop(t, show=True, n_total=len(query_ids))

        # get memory usage
        mem_use = bm25s.utils.benchmark.get_max_memory_usage()
        logging.info(f"Final (peak) memory usage: {mem_use:.2f} GB")
        return self.results
