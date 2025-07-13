import datetime
import logging
import os
import pathlib
import random

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.opensearch.dense import NeuralSearch as Neural

random.seed(42)

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

#### /print debug information to stdout

#### Download dbpedia-entity.zip dataset and unzip the dataset
dataset = "dbpedia-entity"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Loading test queries and corpus in DBPedia
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
corpus_ids, query_ids = list(corpus), list(queries)

#### Randomly sample 1M pairs from Original Corpus (4.63M pairs)
#### First include all relevant documents (i.e. present in qrels)
corpus_set = set()
for query_id in qrels:
    corpus_set.update(list(qrels[query_id].keys()))
# Take only first 50 relevant documents if there are more
corpus_set = set(list(corpus_set)[:50])
corpus_new = {corpus_id: corpus[corpus_id] for corpus_id in corpus_set}

#### Remove already seen k relevant documents and sample (100 - k) docs randomly
remaining_corpus = list(set(corpus_ids) - corpus_set)
target_size = 1000000  # Total corpus size of 100 documents
sample = target_size - len(corpus_set)

for corpus_id in random.sample(remaining_corpus, sample):
    corpus_new[corpus_id] = corpus[corpus_id]

#### Provide parameters for OpenSearch
hostname = "localhost:9200"
index_name = dataset
model = Neural(index_name=index_name, hostname=hostname)
neural = EvaluateRetrieval(model)

#### Index 1M passages into the index (separately)
#neural.retriever.index(corpus_new)

#### Saving benchmark times
neural.evaluate(qrels=qrels, results=neural.retrieve(corpus=corpus_new, queries=queries), k_values=[1, 3, 5, 10, 100])
neural.retriever.cleanup()
