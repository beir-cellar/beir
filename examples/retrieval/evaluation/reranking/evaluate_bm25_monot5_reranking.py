from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import MonoT5
from beir.reranking import Rerank

import pathlib, os
import logging
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download trec-covid.zip dataset and unzip the dataset
dataset = "trec-covid"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where trec-covid has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) trec-covid/corpus.jsonl  (format: jsonlines)
# (2) trec-covid/queries.jsonl (format: jsonlines)
# (3) trec-covid/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#########################################
#### (1) RETRIEVE Top-100 docs using BM25
#########################################

#### Provide parameters for Elasticsearch
hostname = "your-hostname" #localhost
index_name = "your-index-name" # trec-covid
initialize = True # False

model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

##############################################
#### (2) RERANK Top-100 docs using MonoT5 ####
##############################################

#### Reranking using MonoT5 model #####
# Document Ranking with a Pretrained Sequence-to-Sequence Model 
# https://aclanthology.org/2020.findings-emnlp.63/

#### Check below for reference parameters for different MonoT5 models 
#### Two tokens: token_false, token_true
# 1. 'castorini/monot5-base-msmarco':             ['▁false', '▁true']
# 2. 'castorini/monot5-base-msmarco-10k':         ['▁false', '▁true']
# 3. 'castorini/monot5-large-msmarco':            ['▁false', '▁true']
# 4. 'castorini/monot5-large-msmarco-10k':        ['▁false', '▁true']
# 5. 'castorini/monot5-base-med-msmarco':         ['▁false', '▁true']
# 6. 'castorini/monot5-3b-med-msmarco':           ['▁false', '▁true']
# 7. 'unicamp-dl/mt5-base-en-msmarco':            ['▁no'   , '▁yes']
# 8. 'unicamp-dl/ptt5-base-pt-msmarco-10k-v2':    ['▁não'  , '▁sim']
# 9. 'unicamp-dl/ptt5-base-pt-msmarco-100k-v2':   ['▁não'  , '▁sim']
# 10.'unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2':['▁não'  , '▁sim']
# 11.'unicamp-dl/mt5-base-en-pt-msmarco-v2':      ['▁no'   , '▁yes']
# 12.'unicamp-dl/mt5-base-mmarco-v2':             ['▁no'   , '▁yes']
# 13.'unicamp-dl/mt5-base-en-pt-msmarco-v1':      ['▁no'   , '▁yes']
# 14.'unicamp-dl/mt5-base-mmarco-v1':             ['▁no'   , '▁yes']
# 15.'unicamp-dl/ptt5-base-pt-msmarco-10k-v1':    ['▁não'  , '▁sim']
# 16.'unicamp-dl/ptt5-base-pt-msmarco-100k-v1':   ['▁não'  , '▁sim']
# 17.'unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1': ['▁não'  , '▁sim']

cross_encoder_model = MonoT5('castorini/monot5-base-msmarco', token_false='▁false', token_true='▁true')
reranker = Rerank(cross_encoder_model, batch_size=128)

# # Rerank top-100 results using the reranker provided
rerank_results = reranker.rerank(corpus, queries, results, top_k=100)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rerank_results, retriever.k_values)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(rerank_results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))