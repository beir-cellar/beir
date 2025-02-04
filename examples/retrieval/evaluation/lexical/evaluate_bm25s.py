"""
This example show how to evaluate BM25s (https://github.com/xhluca/bm25s) in BEIR.
To be able to run BM25s, you should have BM25s installed locally (on your desktop) along with ``pip install bm25s[core]``.
This code doesn't require GPU to run.

For more details, refer to the BM25s paper & code: https://arxiv.org/abs/2407.03618

Usage: python evaluate_bm25s.py
"""

import logging
import os
import pathlib
import random

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25SSearch

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "nfcorpus"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where scifact has been downloaded and unzipped to the data loader
# data folder would contain these files:
# (1) scifact/corpus.jsonl  (format: jsonlines)
# (2) scifact/queries.jsonl (format: jsonlines)
# (3) scifact/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#### Lexical Retrieval using Bm25s (Python native) ####
# we use the default BM25 variant in our example
method = "robertson"
method_kwargs = {"k1": 1.5, "b": 0.75}
backend = "numba"  # for memory efficiency, you can also use "python"

index_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "index")
os.makedirs(index_dir, exist_ok=True)

retriever = EvaluateRetrieval(
    BM25SSearch(
        method=method,
        method_kwargs=method_kwargs,
        stemmer="english",
        stopwords="english",
        backend=backend,
        index_dir=index_dir,
    ),
    score_function=None,
)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Retrieval Example ####
query_id, scores_dict = random.choice(list(results.items()))
logging.info(f"Query : {queries[query_id]}\n")

scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
for rank in range(10):
    doc_id = scores[rank][0]
    logging.info(f"Rank {rank + 1}: {doc_id} [{corpus[doc_id].get('title')}] - {corpus[doc_id].get('text')}\n")
