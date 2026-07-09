from __future__ import annotations

import torch
from datasets import Dataset

from beir.retrieval.search.dense import DenseRetrievalExactSearch
from beir.retrieval.search.dense.util import pickle_load


class DatasetModel:
    def encode_queries(self, queries, **kwargs):
        assert queries == ["alpha", "gamma"]
        return torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def encode_corpus(self, corpus, **kwargs):
        vectors = []
        for doc in corpus:
            if doc["text"] == "alpha":
                vectors.append([1.0, 0.0])
            elif doc["text"] == "gamma":
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.0, 0.0])
        return torch.tensor(vectors)


def _hf_inputs():
    corpus = Dataset.from_dict(
        {
            "id": ["d1", "d2", "d3"],
            "title": ["", "", ""],
            "text": ["alpha", "beta", "gamma"],
        }
    )
    queries = Dataset.from_dict({"id": ["q1", "q2"], "text": ["alpha", "gamma"]})
    return corpus, queries


def test_exact_search_accepts_hf_dataset_inputs():
    corpus, queries = _hf_inputs()
    retriever = DenseRetrievalExactSearch(DatasetModel(), show_progress_bar=False)

    results = retriever.search(
        corpus=corpus,
        queries=queries,
        top_k=1,
        score_function="dot",
    )

    assert results == {"q1": {"d1": 1.0}, "q2": {"d3": 1.0}}


def test_exact_search_encode_accepts_hf_dataset_inputs(tmp_path):
    corpus, queries = _hf_inputs()
    retriever = DenseRetrievalExactSearch(
        DatasetModel(), corpus_chunk_size=2, show_progress_bar=False
    )

    retriever.encode(
        corpus=corpus,
        queries=queries,
        encode_output_path=str(tmp_path),
        overwrite=True,
    )

    _, query_ids = pickle_load(tmp_path / "queries.pkl")
    _, corpus_ids_0 = pickle_load(tmp_path / "corpus.0.pkl")
    _, corpus_ids_1 = pickle_load(tmp_path / "corpus.1.pkl")

    assert query_ids == ["q1", "q2"]
    assert set(corpus_ids_0 + corpus_ids_1) == {"d1", "d2", "d3"}
