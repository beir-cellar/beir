"""Evaluate KB Arena retrieval strategies on BEIR datasets.

KB Arena (https://github.com/xmpuspus/kb-arena) is an open-source benchmark
that ships nine architecturally distinct retrieval strategies (naive vector,
contextual vector, QnA pairs, knowledge graph, hybrid RRF, RAPTOR, PageIndex,
BM25, rerank-vector). This example wires any KB Arena Strategy into BEIR's
EvaluateRetrieval flow so the same nine strategies can be scored on the BEIR
datasets using BEIR's canonical IR metrics (NDCG, MAP, Recall, Precision).

Install KB Arena alongside BEIR:

    pip install kb-arena

Usage:

    python evaluate_kb_arena.py

By default this runs BM25 against the SciFact split because BM25 needs no
embedding-provider or LLM API keys. To switch to ``naive_vector``,
``contextual_vector``, or any other strategy, change the import below and
provide the embedding-provider env vars KB Arena expects (see KB Arena README).
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search import BaseSearch

# KB Arena public surface
from kb_arena.models.document import Document, Section
from kb_arena.strategies.base import Strategy
from kb_arena.strategies.bm25 import BM25Strategy

# retriever_lab's LLM-stub context manager lets BM25/vector strategies run
# retrieval-only without calling the underlying LLM. KB Arena uses the same
# helper internally for its retriever-lab benchmark.
from kb_arena.benchmark.retriever_lab import _PatchLLMClient


class KBArenaSearch(BaseSearch):
    """Adapter that wraps any KB Arena Strategy as a BEIR BaseSearch backend.

    BEIR's BaseSearch expects ``search(corpus, queries, top_k)`` and returns
    ``{query_id: {doc_id: score}}``. We translate BEIR's corpus dict into
    KB Arena's Document/Section model, build the strategy index once, and then
    run ``Strategy.query`` per query under retriever_lab's LLM-stub patch so
    no API keys are spent on the generation step.
    """

    def __init__(self, strategy: Strategy, corpus_name: str = "beir") -> None:
        self.strategy = strategy
        self.corpus_name = corpus_name

    def _to_kb_documents(self, corpus: dict[str, dict[str, str]]) -> list[Document]:
        docs: list[Document] = []
        for doc_id, fields in corpus.items():
            title = fields.get("title", "") or doc_id
            body = fields.get("text", "")
            section = Section(
                id=f"{doc_id}/main",
                title=title,
                content=body,
                heading_path=[title] if title else [],
                level=1,
            )
            docs.append(
                Document(
                    id=doc_id,
                    source=doc_id,
                    corpus=self.corpus_name,
                    title=title,
                    sections=[section],
                )
            )
        return docs

    async def _aindex(self, documents: list[Document]) -> None:
        await self.strategy.build_index(documents)

    async def _aquery(
        self, queries: dict[str, str], top_k: int
    ) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        with _PatchLLMClient():
            for qid, qtext in queries.items():
                answer = await self.strategy.query(qtext, top_k=top_k)
                trace = answer.retrieval
                if trace is None:
                    results[qid] = {}
                    continue
                results[qid] = {
                    chunk.doc_id: float(chunk.score) for chunk in trace.retrieved
                }
        return results

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        *args,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        loop = asyncio.new_event_loop()
        try:
            documents = self._to_kb_documents(corpus)
            loop.run_until_complete(self._aindex(documents))
            return loop.run_until_complete(self._aquery(queries, top_k))
        finally:
            loop.close()


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    dataset = "scifact"
    url = (
        f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    )
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)

    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

    # Swap BM25Strategy() for any other KB Arena strategy:
    #   from kb_arena.strategies.naive_vector import NaiveVectorStrategy
    #   strategy = NaiveVectorStrategy()
    # (vector strategies require an embedding provider; see KB Arena README.)
    model = KBArenaSearch(strategy=BM25Strategy(), corpus_name=dataset)

    retriever = EvaluateRetrieval(model, k_values=[1, 3, 5, 10, 100])
    results = retriever.retrieve(corpus, queries)

    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values
    )

    logging.info("NDCG: %s", ndcg)
    logging.info("MAP: %s", _map)
    logging.info("Recall: %s", recall)
    logging.info("Precision: %s", precision)


if __name__ == "__main__":
    main()
