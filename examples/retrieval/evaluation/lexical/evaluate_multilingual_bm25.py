"""
This example show how to evaluate BM25 model (Elasticsearch) in BEIR for German.
This script can be used to any evaluate any language by just changing language name.
To find languages supported by Elasticsearch, please refer below:
https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html

To be able to run Elasticsearch, you should have it installed locally (on your desktop) along with ``pip install beir``.
Depending on your OS, you would be able to find how to download Elasticsearch. I like this guide for Ubuntu 18.04 -
https://linuxize.com/post/how-to-install-elasticsearch-on-ubuntu-18-04/ 
For more details, please refer here - https://www.elastic.co/downloads/elasticsearch. 

This code doesn't require GPU to run.

If unable to get it running locally, you could try the Google Colab Demo, where we first install elastic search locally and retrieve using BM25
https://colab.research.google.com/drive/1HfutiEhHMJLXiWGT8pcipxT5L2TpYEdt?usp=sharing#scrollTo=nqotyXuIBPt6


Usage: python evaluate_multilingual_bm25.py
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25

import pathlib, os, random
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "germanquad"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where scifact has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) scifact/corpus.jsonl  (format: jsonlines)
# (2) scifact/queries.jsonl (format: jsonlines)
# (3) scifact/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#### Lexical Retrieval using Bm25 (Elasticsearch) ####
#### Provide a hostname (localhost) to connect to ES instance
#### Define a new index name or use an already existing one.
#### We use default ES settings for retrieval
#### https://www.elastic.co/

hostname = "your-hostname" #localhost
index_name = "your-index-name" # germanquad

#### Intialize #### 
# (1) True - Delete existing index and re-index all documents from scratch 
# (2) False - Load existing index
initialize = True # False

#### Language ####
# For languages supported by Elasticsearch by default, check here ->
# https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-lang-analyzer.html
language = "german" # Please provide full names in lowercase for eg. english, hindi ...

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname, language=language, initialize=initialize, number_of_shards=number_of_shards)

# (2) For datasets with big corpus ==> keep default configuration
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Retrieval Example ####
query_id, scores_dict = random.choice(list(results.items()))
logging.info("Query : %s\n" % queries[query_id])

scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
for rank in range(10):
    doc_id = scores[rank][0]
    logging.info("Doc %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))