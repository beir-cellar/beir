from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.sparse import SparseSearch

import logging
import pathlib, os
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

dataset = "nfcorpus"

#### Download NFCorpus dataset and unzip the dataset
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))
data_path= "/home/ukp/thakur/projects/sbert_retriever/datasets-new/{}".format(dataset)
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### SPARSE Retrieval using uniCOIL ####
# uniCOIL implementes an architecture similar to COIL, SPLADE. 
# It computes a weight for each token in query and document
# Finally a dot product is used to evaluate between similar query and document tokens.

####################################################
#### 1. Loading uniCOIL model using HuggingFace ####
####################################################
# We download the publicly available uniCOIL model from the HF repository
# For more details on how the model works, please refer: (https://arxiv.org/abs/2106.14807)

model_path = "castorini/unicoil-d2q-msmarco-passage"
model = SparseSearch(models.UniCOIL(model_path=model_path), batch_size=32)
retriever = EvaluateRetrieval(model, score_function="dot")

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries, query_weights=True)

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))