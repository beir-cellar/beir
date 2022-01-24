#! /usr/bin/env python3

import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical.vespa_search import VespaLexicalSearch
from beir.retrieval.evaluation import EvaluateRetrieval
from pandas import DataFrame


def download_and_unzip_dataset(data_dir, dataset_name):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset_name
    )
    data_path = util.download_and_unzip(url, data_dir)
    print("Dataset downloaded here: {}".format(data_path))
    return data_path


def prepare_data(data_path):
    corpus, queries, qrels = GenericDataLoader(data_path).load(
        split="test"
    )  # or split = "train" or "dev"
    return corpus, queries, qrels


def get_search_results(dataset_name, corpus, queries, qrels):
    initialize = True
    deployment_parameters = None
    metrics = []
    for match_phase in ["or", "weak_and"]:
        for rank_phase in ["bm25", "native_rank"]:
            model = VespaLexicalSearch(
                application_name=dataset_name,
                match_phase=match_phase,
                rank_phase=rank_phase,
                initialize=initialize,
                deployment_parameters=deployment_parameters,
            )
            initialize = False
            deployment_parameters = {"url": "http://localhost", "port": 8089}
            retriever = EvaluateRetrieval(model)
            results = retriever.retrieve(corpus, queries)
            ndcg, _map, recall, precision = retriever.evaluate(
                qrels, results, retriever.k_values
            )
            metric = {
                "dataset_name": dataset_name,
                "match_phase": match_phase,
                "rank_phase": rank_phase,
            }
            metric.update(ndcg)
            metric.update(_map)
            metric.update(recall)
            metric.update(precision)
            metrics.append(metric)
    return metrics


def benchmark_vespa_lexical(data_dir, dataset_names):
    result = []
    for dataset_name in dataset_names:
        print("Dataset: {}".format(dataset_name))
        data_path = download_and_unzip_dataset(
            data_dir=data_dir, dataset_name=dataset_name
        )
        corpus, queries, qrels = prepare_data(data_path=data_path)
        metrics = get_search_results(
            dataset_name=dataset_name, corpus=corpus, queries=queries, qrels=qrels
        )
        output_file = os.path.join(data_dir, "metrics.csv")
        if os.path.isfile(output_file):
            DataFrame.from_records(metrics).to_csv(
                output_file, mode="a", header=False, index=False
            )
        else:
            DataFrame.from_records(metrics).to_csv(
                output_file, mode="w", header=True, index=False
            )
        print(metrics)
        result.extend(metrics)
    return result


if __name__ == "__main__":

    data_dir = "/Users/tmartins/projects/sw/beir/data"
    dataset_names = ["scifact", "nfcorpus"]

    result = benchmark_vespa_lexical(data_dir=data_dir, dataset_names=dataset_names)
