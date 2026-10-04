from __future__ import annotations

import json

from beir.datasets.data_loader import GenericDataLoader


def test_load_reuses_corpus_across_splits_without_leaking_queries_or_qrels(tmp_path):
    (tmp_path / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "", "text": "document"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "queries.jsonl").write_text(
        "\n".join(json.dumps({"_id": qid, "text": text}) for qid, text in (("test-q", "test"), ("train-q", "train")))
        + "\n",
        encoding="utf-8",
    )
    qrels_folder = tmp_path / "qrels"
    qrels_folder.mkdir()
    (qrels_folder / "test.tsv").write_text("query-id\tcorpus-id\tscore\ntest-q\td1\t1\n", encoding="utf-8")
    (qrels_folder / "train.tsv").write_text("query-id\tcorpus-id\tscore\ntrain-q\td1\t1\n", encoding="utf-8")

    loader = GenericDataLoader(data_folder=str(tmp_path))
    corpus, test_queries, test_qrels = loader.load(split="test")
    train_corpus, train_queries, train_qrels = loader.load(split="train")
    repeated_corpus, repeated_queries, repeated_qrels = loader.load(split="test")

    assert train_corpus is corpus is repeated_corpus
    assert test_queries == repeated_queries == {"test-q": "test"}
    assert train_queries == {"train-q": "train"}
    assert test_qrels == repeated_qrels == {"test-q": {"d1": 1}}
    assert train_qrels == {"train-q": {"d1": 1}}
