from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseOfflineRetrievalExactSearch
from typing import List, Dict, Optional
import numpy as np

import logging
import pathlib, os
import random

class OfflineModel:
    def __init__(self, model_path=None, **kwargs):
        self.model = None # ---> HERE Load your custom model
        self.query_npy = np.load(f"{model_path}/queries.npy")
        self.corpus_npy = np.load(f"{model_path}/corpus0.npy")
        self.query_ids = self.load_ids(f"{model_path}/queries.ids")
        self.corpus_ids = self.load_ids(f"{model_path}/corpus0.ids")
        # self.model = SentenceTransformer(model_path)
    
    def load_ids(self, id_strs_path):
        id_strs = open(id_strs_path, 'r').read().splitlines()
        return {id.strip(): idx for idx, id in enumerate(id_strs)}
    
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    # For eg ==> return np.asarray(self.model.encode(queries, batch_size=batch_size, **kwargs))
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        idxs = [self.query_ids[q_id] for q_id in queries]
        return self.query_npy[idxs]
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    # For eg ==> sentences = [(doc["title"] + "  " + doc["text"]).strip() for doc in corpus]
    #        ==> return np.asarray(self.model.encode(sentences, batch_size=batch_size, **kwargs))
    def encode_corpus(self, corpus: List[str], batch_size: int = 8, **kwargs) -> np.ndarray:
        idxs = [self.corpus_ids[c_id] for c_id in corpus]
        return self.corpus_npy[idxs]


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

dataset = "scifact"

#### Download nfcorpus.zip dataset and unzip the dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Dense Retrieval using SBERT (Sentence-BERT) ####
#### Provide any pretrained sentence-transformers model
#### The model was fine-tuned using cosine-similarity.
#### Complete list - https://www.sbert.net/docs/pretrained_models.html

model = DenseOfflineRetrievalExactSearch(OfflineModel(model_path="./offline_model"))
retriever = EvaluateRetrieval(model, score_function="dot")

#### Retrieve dense results (format of results is identical to qrels)
start_time = time()
results = retriever.retrieve(corpus, queries)
end_time = time()
print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
