"""
This example shows how to evaluate docTTTTTquery in BEIR.

Since Anserini uses Java 11, we would advise you to use docker for running Pyserini.
To be able to run the code below you must have docker locally installed in your machine.
To install docker on your local machine, please refer here: https://docs.docker.com/get-docker/

After docker installation, please follow the steps below to get docker container up and running:

1. docker pull docker pull beir/pyserini-fastapi
2. docker build -t pyserini-fastapi .
3. docker run -p 8000:8000 -it --rm pyserini-fastapi

Once the docker container is up and running in local, now run the code below.

For the example, we use the "castorini/doc2query-t5-base-msmarco" model for query generation.
In this example, we generate 3 questions per passage and append them with passage used for BM25 retrieval.

Usage: python evaluate_anserini_docT5query.py --dataset <dataset>
"""

import argparse
import json
import logging
import os
import pathlib
import random
import requests
import torch
import torch.multiprocessing as mp

from tqdm import tqdm

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.generation.models import QGenModel

CHUNK_SIZE_MP = 100
CHUNK_SIZE_GPU = 64  # memory-bound, this should work for most GPUs
DEVICE_CPU = 'cpu'
DEVICE_GPU = 'cuda'
NUM_QUERIES_PER_PASSAGE = 5
PYSERINI_URL = "http://127.0.0.1:8000"

DEFAULT_MODEL_ID = 'BeIR/query-gen-msmarco-t5-base-v1' # https://huggingface.co/BeIR/query-gen-msmarco-t5-base-v1
DEFAULT_DEVICE = DEVICE_GPU

# noinspection PyArgumentList
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO, handlers=[LoggingHandler()])


def init_process(device, model_id):
    """Initializes a worker process."""

    global model

    if device == DEVICE_GPU:
        # Assign the GPU process ID to bind this process to a specific GPU
        # This is a bit fragile and relies on CUDA ordinals being the same
        # See: https://stackoverflow.com/questions/63564028/multiprocess-pool-initialization-with-sequential-initializer-argument
        proc_id = int(mp.current_process().name.split('-')[1]) - 1
        device = f'{DEVICE_GPU}:{proc_id}'

    model = QGenModel(model_id, use_fast=True, device=device)


def _decide_device(cpu_procs):
    """Based on command line arguments, sets the device and number of processes to use."""

    if cpu_procs:
        return DEVICE_CPU, cpu_procs
    else:
        assert torch.cuda.is_available(), "No GPUs available. Please set --cpu-procs or make GPUs available"
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        return DEVICE_GPU, torch.cuda.device_count()


def _download_dataset(dataset):
    """Downloads a dataset and unpacks it on disk."""

    url = 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip'.format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), 'datasets')
    return util.download_and_unzip(url, out_dir)


def _generate_query(corpus_list):
    """Generates a set of queries for a given document."""

    documents = [document for _, document in corpus_list]
    generated_queries = model.generate(corpus=documents,
                                       ques_per_passage=NUM_QUERIES_PER_PASSAGE,
                                       max_length=64,
                                       temperature=1,
                                       top_k=10)

    for i, (_, document) in enumerate(corpus_list):
        start_index = i * NUM_QUERIES_PER_PASSAGE
        end_index = start_index + NUM_QUERIES_PER_PASSAGE
        document["queries"] = generated_queries[start_index:end_index]

    return dict(corpus_list)


def _add_generated_queries_to_corpus(num_procs, device, model_id, corpus):
    """Using a pool of workers, generate queries to add to each document in the corpus."""

    # Chunk input so we can maximize the use of our GPUs
    corpus_list = list(corpus.items())
    chunked_corpus = [corpus_list[pos:pos + CHUNK_SIZE_GPU] for pos in range(0, len(corpus_list), CHUNK_SIZE_GPU)]

    pool = mp.Pool(num_procs, initializer=init_process, initargs=(device, model_id))
    for partial_corpus in tqdm(pool.imap_unordered(_generate_query, chunked_corpus, chunksize=CHUNK_SIZE_MP), total=len(chunked_corpus)):
        corpus.update(partial_corpus)

    return corpus


def _write_pyserini_corpus(pyserini_index_file, corpus):
    """Writes the in-memory corpus to disk in the Pyserini format."""

    with open(pyserini_index_file, 'w', encoding='utf-8') as fOut:
        for doc_id, document in corpus.items():
            data = {
                'id': doc_id,
                'title': document.get('title', ''),
                'contents': document.get('text', ''),
                'queries': ' '.join(document.get('queries', '')),
            }
            json.dump(data, fOut)
            fOut.write('\n')


def _index_pyserini(pyserini_index_file, dataset):
    """Uploads a Pyserini index file and indexes it into Lucene."""

    with open(pyserini_index_file, 'rb') as fIn:
        r = requests.post(f'{PYSERINI_URL}/upload/', files={'file': fIn}, verify=False)

    r = requests.get(f'{PYSERINI_URL}/index/', params={'index_name': f'beir/{dataset}'})


def _search_pyserini(queries, k):
    """Searches an index in Pyserini in bulk."""

    qids = list(queries)
    query_texts = [queries[qid] for qid in qids]
    payload = {
        'queries': query_texts,
        'qids': qids,
        'k': k,
        'fields': {'contents': 1.0, 'title': 1.0, 'queries': 1.0},
    }

    r = requests.post(f'{PYSERINI_URL}/lexical/batch_search/', json=payload)
    return json.loads(r.text)['results']


def _print_retrieval_examples(corpus, queries, results):
    """Prints retrieval examples for inspection."""

    query_id, scores_dict = random.choice(list(results.items()))
    logging.info(f"Query: {queries[query_id]}\n")

    scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    for rank in range(10):
        doc_id = scores[rank][0]
        logging.info(
            "Doc %d: %s [%s] - %s\n" % (rank + 1, doc_id, corpus[doc_id].get('title'), corpus[doc_id].get('text')))


def main():
    parser = argparse.ArgumentParser(prog='evaluate_anserini_docT5query_parallel')
    parser.add_argument('--dataset', required=True, help=f"The dataset to use. Example: scifact")
    parser.add_argument('--model-id',
                        default=DEFAULT_MODEL_ID, help=f"The model ID to use. Default: {DEFAULT_MODEL_ID}")
    parser.add_argument('--cpu-procs', default=None, type=int,
                        help=f"Use CPUs instead of GPUs and use this number of cores. Leaving this unset (default) "
                             "will use all available GPUs. Default: None")
    args = parser.parse_args()

    device, num_procs = _decide_device(args.cpu_procs)

    # Download and load the dataset into memory
    data_path = _download_dataset(args.dataset)
    pyserini_index_file = os.path.join(data_path, 'pyserini.jsonl')
    corpus, queries, qrels = GenericDataLoader(data_path).load(split='test')

    # Generate queries per document and create Pyserini index file if does not exist yet
    if not os.path.isfile(pyserini_index_file):
        _add_generated_queries_to_corpus(num_procs, device, args.model_id, corpus)
        _write_pyserini_corpus(pyserini_index_file, corpus)

    # Index into Pyserini
    _index_pyserini(pyserini_index_file, args.dataset)

    # Retrieve and evaluate
    retriever = EvaluateRetrieval()
    results = _search_pyserini(queries, k=max(retriever.k_values))
    retriever.evaluate(qrels, results, retriever.k_values)

    _print_retrieval_examples(corpus, queries, results)


if __name__ == "__main__":
    main()
