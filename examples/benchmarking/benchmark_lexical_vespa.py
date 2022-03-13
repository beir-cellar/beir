#! /usr/bin/env python3

import os
import shutil
from typing import Tuple, Optional, List, Dict
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical.vespa_search import VespaLexicalSearch
from beir.retrieval.evaluation import EvaluateRetrieval
from pandas import DataFrame


def download_and_unzip_dataset(data_dir: str, dataset_name: str) -> str:
    """
    Download and unzip dataset

    :param data_dir: Folder path to hold the downloaded files
    :param dataset_name: Name of the dataset according to BEIR benchmark

    :return: Return the path of the folder containing the unzipped dataset files.
    """
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(
        dataset_name
    )
    data_path = util.download_and_unzip(url, data_dir)
    print("Dataset downloaded here: {}".format(data_path))
    return data_path


def prepare_data(data_path: str, split_type: str = "test") -> Tuple:
    """
    Extract corpus, queries and qrels from the test dataset.

    :param data_path: Folder path that contains the unzipped dataset files.
    :param split_type: One of 'train', 'dev' or 'test' set.

    :return: a tuple containing 'corpus', 'queries' and 'qrels'.
    """
    corpus, queries, qrels = GenericDataLoader(data_path).load(
        split=split_type
    )  # or split = "train" or "dev"
    return corpus, queries, qrels


def parse_match_phase_argument(match_phase: Optional[List[str]] = None) -> List[str]:
    """
    Parse match phase argument.

    :param match_phase: An optional list of match phase types to use in the experiments.
        Currently supported types are 'weak_and' and 'or'. By default the experiments will use
        'weak_and'.

    :return: A list with all the match phase types to use in the experiments.
    """
    if not match_phase:
        match_phase_list = ["weak_and"]
    else:
        assert all(
            [x in ["or", "weak_and"] for x in match_phase]
        ), "match_phase must be a list containing 'weak_and' and/or 'or'."
        match_phase_list = match_phase
    return match_phase_list


def parse_rank_phase_argument(rank_phase: Optional[List[str]] = None) -> List[str]:
    """
    Parse rank phase argument.

    :param rank_phase: An optional list of rank phase types to use in the experiments.
        Currently supported types are 'bm25' and 'native_rank'. By default the experiments will use
        'bm25'.

    :return: A list with all the match phase types to use in the experiments.
    """
    if not rank_phase:
        rank_phase_list = ["bm25"]
    else:
        assert all(
            [x in ["native_rank", "bm25"] for x in rank_phase]
        ), "rank_phase must be a list containing 'native_rank' and/or 'bm25'."
        rank_phase_list = rank_phase
    return rank_phase_list


def get_search_results(
    dataset_name: str,
    corpus: Dict[str, Dict[str, str]],
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    match_phase: Optional[List[str]] = None,
    rank_phase: Optional[List[str]] = None,
    initialize: bool = True,
    remove_app: bool = True,
):
    """
    Deploy an Vespa app, feed, query and compute evaluation metrics

    :param dataset_name: Name of the dataset according to BEIR benchmark
    :param corpus: Corpus used to feed the app.
    :param queries: Queries used to query the app.
    :param qrels: Labeled data used to evaluate the query results.
    :param match_phase: An optional list of match phase types to use in the experiments.
        Currently supported types are 'weak_and' and 'or'. By default the experiments will use
        'weak_and'.
    :param rank_phase: An optional list of rank phase types to use in the experiments.
        Currently supported types are 'bm25' and 'native_rank'. By default the experiments will use
        'bm25'.
    :param initialize: Deploy and feed the app on the first run of the experiments. Default to True.
    :param remove_app: Stop and remove the app after the experiments are run. Default to True.
    """
    if initialize:
        deployment_parameters = None
    else:
        deployment_parameters = {"url": "http://localhost", "port": 8089}
    match_phase_list = parse_match_phase_argument(match_phase=match_phase)
    rank_phase_list = parse_rank_phase_argument(rank_phase=rank_phase)
    metrics = []
    for match_phase in match_phase_list:
        for rank_phase in rank_phase_list:
            model = VespaLexicalSearch(
                application_name=dataset_name,
                match_phase=match_phase,
                rank_phase=rank_phase,
                initialize=initialize,
                deployment_parameters=deployment_parameters,
            )
            initialize = False  # only initialize the first run
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
    if remove_app:
        try:
            model.remove_app()
        except:  # todo: could not find how to increase container.remove() timeout (https://github.com/docker/docker-py/issues/2951)
            pass
    return metrics


def benchmark_vespa_lexical(
    data_dir: str,
    dataset_names: List[str],
    split_type: str = "test",
    match_phase: Optional[List[str]] = None,
    rank_phase: Optional[List[str]] = None,
    initialize: bool = True,
    remove_dataset: bool = True,
    remove_app: bool = True,
):
    """
    Benchmark Vespa lexical search app against a suite of BEIR datasets.

    A metrics.csv file will be created at 'data_dir' containing the metrics computed in the experiments.

    :param data_dir: Folder path to hold the downloaded files
    :param dataset_names: A list of dataset names according to the BEIR benchmark.
    :param split_type: One of 'train', 'dev' or 'test' set.
    :param match_phase: An optional list of match phase types to use in the experiments.
        Currently supported types are 'weak_and' and 'or'. By default the experiments will use
        'weak_and'.
    :param rank_phase: An optional list of rank phase types to use in the experiments.
        Currently supported types are 'bm25' and 'native_rank'. By default the experiments will use
        'bm25'.
    :param initialize: Deploy and feed the app on the first run of the experiments. Default to True.
    :param remove_dataset: Remove dataset files after the experiments are run. Default to True.
    :param remove_app: Stop and remove the app after the experiments are run. Default to True.
    """
    result = []
    for dataset_name in dataset_names:
        print("Dataset: {}".format(dataset_name))
        data_path = download_and_unzip_dataset(
            data_dir=data_dir, dataset_name=dataset_name
        )
        corpus, queries, qrels = prepare_data(
            data_path=data_path, split_type=split_type
        )
        metrics = get_search_results(
            dataset_name=dataset_name,
            corpus=corpus,
            queries=queries,
            qrels=qrels,
            match_phase=match_phase,
            rank_phase=rank_phase,
            initialize=initialize,
            remove_app=remove_app,
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
        if remove_dataset:
            shutil.rmtree(os.path.join(data_dir, dataset_name))
            os.remove(os.path.join(data_dir, dataset_name + ".zip"))
    return result


if __name__ == "__main__":

    data_dir = os.environ.get("DATA_DIR", os.getcwd())
    dataset_names = [
        "scifact",
        "trec-covid",
        "nfcorpus",
        "nq",
        "fiqa",
        "arguana",
        "webis-touche2020",
        "cqadupstack",
        "quora",
        "dbpedia-entity",
        "scidocs",
        "fever",
        "climate-fever",
        "hotpotqa",
    ]
    _ = benchmark_vespa_lexical(
        data_dir=data_dir,
        dataset_names=dataset_names,
        split_type="test",
        match_phase=["weak_and"],
        rank_phase=["bm25"],
        initialize=True,
        remove_dataset=True,
        remove_app=True,
    )
    #
    # MS MARCO is the only dataset which uses the dev set to compute the metrics
    #
    _ = benchmark_vespa_lexical(
        data_dir=data_dir,
        dataset_names=["msmarco"],
        split_type="dev",
        match_phase=["weak_and"],
        rank_phase=["bm25"],
        initialize=True,
        remove_dataset=True,
        remove_app=True,
    )
