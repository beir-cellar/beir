from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
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

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = "trec-covid"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#### Provide parameters for elastic-search
# hostname = "your-hostname" #localhost
# index_name = "your-index-name" # nfcorpus
hostname = "desktop-158.ukp.informatik.tu-darmstadt.de:9200"
index_name = dataset
initialize = False

model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Reranking using Cross-Encoder models (list: )
cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-electra-base')
reranker = Rerank(cross_encoder_model, batch_size=128)

# Rerank top-100 results using the reranker provided
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