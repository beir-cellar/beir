from __future__ import annotations

import torch

from beir.retrieval.search.dense import DenseRetrievalExactSearch


class DummyModel:
    def encode_queries(self, queries, **kwargs):
        assert queries == ["alpha"]
        return torch.tensor([[1.0, 0.0]])

    def encode_corpus(self, corpus, **kwargs):
        assert [doc["text"] for doc in corpus] == ["alpha", "beta"]
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]])


def test_exact_search_handles_single_query():
    retriever = DenseRetrievalExactSearch(DummyModel(), show_progress_bar=False)

    results = retriever.search(
        corpus={
            "d1": {"title": "", "text": "alpha"},
            "d2": {"title": "", "text": "beta"},
        },
        queries={"q1": "alpha"},
        top_k=1,
        score_function="dot",
    )

    assert results == {"q1": {"d1": 1.0}}
