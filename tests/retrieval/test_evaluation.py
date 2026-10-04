from __future__ import annotations

import logging

import pytest

from beir.retrieval.evaluation import EvaluateRetrieval


@pytest.mark.parametrize("missing_results", [None, {}], ids=["omitted", "empty"])
def test_evaluate_counts_queries_without_results_as_zero(missing_results):
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}}
    results = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
    if missing_results is not None:
        results["q3"] = missing_results

    ndcg, mean_ap, recall, precision = EvaluateRetrieval.evaluate(qrels, results, [1, 10])

    assert ndcg == {"NDCG@1": 0.66667, "NDCG@10": 0.66667}
    assert mean_ap == {"MAP@1": 0.66667, "MAP@10": 0.66667}
    assert recall == {"Recall@1": 0.66667, "Recall@10": 0.66667}
    assert precision == {"P@1": 0.66667, "P@10": 0.06667}


@pytest.mark.parametrize("results", [{}, {"unjudged": {"d1": 1.0}}], ids=["empty", "unjudged-only"])
def test_evaluate_returns_zero_when_no_judged_queries_have_results(results):
    metrics = EvaluateRetrieval.evaluate({"q1": {"d1": 1}}, results, [1, 10])

    for metric in metrics:
        assert list(metric.values()) == [0.0, 0.0]


def test_evaluate_warns_about_omitted_judged_queries(caplog):
    with caplog.at_level(logging.WARNING, logger="beir.retrieval.evaluation"):
        EvaluateRetrieval.evaluate({"q1": {"d1": 1}, "q2": {"d2": 1}}, {"q1": {"d1": 1.0}}, [1])

    assert "1 judged queries are missing from results" in caplog.text


def test_evaluate_does_not_warn_for_explicitly_empty_results(caplog):
    with caplog.at_level(logging.WARNING, logger="beir.retrieval.evaluation"):
        EvaluateRetrieval.evaluate({"q1": {"d1": 1}}, {"q1": {}}, [1])

    assert not caplog.records


def test_evaluate_ignores_unjudged_queries_in_average():
    metrics = EvaluateRetrieval.evaluate({"q1": {"d1": 1}}, {"q1": {"d1": 1.0}, "unjudged": {"d2": 1.0}}, [1])

    for metric in metrics:
        assert list(metric.values()) == [1.0]


def test_evaluate_rejects_empty_qrels():
    with pytest.raises(ValueError, match="qrels must contain at least one judged query"):
        EvaluateRetrieval.evaluate({}, {}, [1])
