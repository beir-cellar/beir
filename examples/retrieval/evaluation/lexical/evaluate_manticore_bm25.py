"""
This example shows how to evaluate Manticore BM25f model in BEIR.
To install and run Manticore server on your local machine, follow the instruction from this manual - 
https://manual.manticoresearch.com/Installation

The code doesn't require GPU to run.

Usage: python evaluate_manticore_bm25.py
:option --data_dir -d: A folder path for downloaded dataset files. 
:option --dataset-name -n: A dataset(s) to be used in the benchmark.
:option --host -h: Hostname and port your Manticore server is running on, e.g. localhost:9308
:option --outfile -o: Filepath to save benchmarking results
:option --store-datasets: Store downloaded dataset files after benchmarking is completed.
:option --store-indexes: Store created indexes after benchmarking is completed.  
"""

import os
import shutil
import typer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical.manticore_search import ManticoreLexicalSearch
from beir.retrieval.evaluation import EvaluateRetrieval
from pandas import DataFrame
from typing import List, Optional
from wasabi import msg


def load_datasets( data_dir: str, dataset_names: List[str] ) -> List[str]:
    """
    Download necessary datasets
    
    :param data_dir: A folder path for downloaded files.
    :param dataset_names: A list of dataset names to be used in the benchmark.
    :return: A list of filepathes to downloaded datasets.
    """
    print("Loading datasets:")
    url_tmpl = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
    data_pathes = [
         util.download_and_unzip( url_tmpl.format(name), data_dir) for name in dataset_names
    ]
    print("Done") 
    return data_pathes


def remove_datasets( data_dir: str, dataset_names: List[str] ):
    """
    Remove downloaded datasets
    
    :param data_dir: A folder path for downloaded dataset files.
    :param dataset_names: A list of dataset names to be removed.
    """
    for name in dataset_names:
        shutil.rmtree( os.path.join(data_dir, name) )
        os.remove( os.path.join(data_dir, name + ".zip") )


def save_results(metrics: List, outfile: str):
    """
    Save calculated metrics data
    
    :param metrics: A list of calculated metrics.
    :param outfile: Path to outfile.
    """
    if os.path.isfile(outfile):
        DataFrame.from_records(metrics).to_csv(
            outfile, mode="a", header=False, index=False
        )
    else:
        DataFrame.from_records(metrics).to_csv(
            outfile, mode="w", header=True, index=False
        )
    savepath = os.getcwd() + "/" + outfile
    msg.good("Benchmarking results are saved to " + savepath)


def benchmark(
        data_dir: str = typer.Option( os.getcwd(), '--data-dir', '-d'),
        dataset_names: List[str] = typer.Option( [
            "msmarco",
            "scifact",
            "trec-covid",
            "nfcorpus",
            "nq",
            "fiqa",
            "arguana",
            "webis-touche2020",
            "dbpedia-entity",
            "scidocs",
            "fever",
            "climate-fever",
            "hotpotqa",
        ], '--dataset-name', '-n' ),
        host: str = typer.Option( "http://localhost:9308", '--host', '-h' ),
        outfile: Optional[str] = typer.Option(None, '--outfile', '-o'),
        store_datasets: bool = False,
        store_indexes: bool = False,
        ):
    """
    Benchmark Manticore BM25 search relevance across a collection of BEIR datasets.
    
    :param data_dir: A folder path for downloaded files. By default, set to the current script's folder.    
    
    :param dataset_names: A list of dataset names to be used in the benchmark. By default,
     all the datasets available for download from the BEIR's leaderboard are used.
    
    :param host: Hostname and port your Manticore server is running on. By default, 
    set to  http://localhost:9308
    
    :param store_datasets: Store downloaded dataset files after benchmarking is completed. By default, 
    set to False.
    
    :param store_indexes: Store created indexes after benchmarking is completed. By default, 
    set to False.
    
    :param outfile: File to save benchmark results. By default, set to None
    """
    print("Benchmarking is started\n")
    metrics = []
    data_pathes = load_datasets(data_dir, dataset_names)
    for i,name in enumerate(dataset_names):
        print("\nDataset " + name + ":")
        # Create an evaluation model for Manticore search
        model = ManticoreLexicalSearch(
            index_name=name,
            host=host,
            store_indexes=store_indexes,
            )
        # Msmarco is the only dataset using "dev" set for its evaluation 
        split_type = 'dev' if name == 'msmarco' else 'test'
        # Extract corpus, queries and qrels from dataset.
        corpus, queries, qrels = GenericDataLoader( data_pathes[i] ).load(split=split_type)
        # Performing evaluations with the set of metrics given( NDCG and so on )
        retriever = EvaluateRetrieval(model)
        results = retriever.retrieve(corpus, queries)
        ndcg, _map, recall, precision = retriever.evaluate(
            qrels, results, retriever.k_values
        )
        metric = {"Dataset": name}
        metric.update(ndcg)
        metric.update(_map)
        metric.update(recall)
        metric.update(precision)
        metrics.append(metric)
    if not store_datasets:
        remove_datasets(data_dir, dataset_names)
    # Output benchmark results
    if outfile is not None:
        save_results(metrics, outfile)
    print( "\n" + DataFrame(data=metrics).to_markdown(tablefmt='grid')  + "\n" )
    msg.good("Benchmarking is successfully finished")


if __name__ == "__main__":
    typer.run(benchmark)
    