from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from beir.datasets.data_loader_hf import HFDataLoader


def test_hub_revisions_are_passed_to_their_respective_repositories():
    loader = HFDataLoader(
        hf_repo="BeIR/scifact",
        hf_repo_qrels="BeIR/scifact-qrels",
        revision="corpus-commit",
        qrels_revision="qrels-commit",
    )
    dataset = MagicMock()
    dataset.values.return_value = iter([MagicMock()])
    dataset.cast_column.return_value = dataset
    dataset.rename_column.return_value = dataset
    dataset.remove_columns.return_value = dataset
    dataset.cast.return_value = dataset
    dataset.column_names = ["id", "text", "title"]

    with patch("beir.datasets.data_loader_hf.load_dataset") as load_dataset:
        load_dataset.return_value = {"test": dataset}
        loader._load_corpus()
        loader._load_queries()
        loader._load_qrels("test")

    assert load_dataset.call_args_list == [
        call("BeIR/scifact", "corpus", keep_in_memory=False, streaming=False, revision="corpus-commit"),
        call("BeIR/scifact", "queries", keep_in_memory=False, streaming=False, revision="corpus-commit"),
        call("BeIR/scifact-qrels", keep_in_memory=False, streaming=False, revision="qrels-commit"),
    ]


def test_hub_revisions_default_to_current_dataset_versions():
    loader = HFDataLoader(hf_repo="BeIR/scifact")
    dataset = MagicMock()
    dataset.values.return_value = iter([MagicMock()])
    dataset.cast_column.return_value = dataset
    dataset.rename_column.return_value = dataset
    dataset.remove_columns.return_value = dataset
    dataset.cast.return_value = dataset
    dataset.column_names = ["id", "text", "title"]

    with patch("beir.datasets.data_loader_hf.load_dataset") as load_dataset:
        load_dataset.return_value = {"test": dataset}
        loader._load_corpus()
        loader._load_queries()
        loader._load_qrels("test")

    assert [args.kwargs["revision"] for args in load_dataset.call_args_list] == [None, None, None]
