from __future__ import annotations

from beir.retrieval.custom_metrics import top_k_accuracy, top_k_success
from beir.retrieval.evaluation import EvaluateRetrieval


def test_success_at_k_counts_queries_with_a_relevant_top_k_hit():
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d2": 1},
        "q3": {"d3": 1},
    }
    results = {
        "q1": {"irrelevant": 2.0, "d1": 1.0},
        "q2": {"d2": 2.0},
    }
    expected = {"Success@1": 0.33333, "Success@2": 0.66667}

    assert top_k_success(qrels, results, [1, 2]) == expected
    for alias in ("success", "success@k", "top_k_success"):
        assert EvaluateRetrieval.evaluate_custom(qrels, results, [1, 2], alias) == expected


def test_accuracy_at_k_remains_backward_compatible():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    results = {"q1": {"d1": 1.0}, "q2": {"irrelevant": 1.0}}
    expected = {"Accuracy@1": 0.5}

    assert top_k_accuracy(qrels, results, [1]) == expected
    for alias in ("acc", "top_k_acc", "accuracy", "accuracy@k", "top_k_accuracy"):
        assert EvaluateRetrieval.evaluate_custom(qrels, results, [1], alias) == expected
