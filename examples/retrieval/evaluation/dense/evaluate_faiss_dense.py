"""
In this example, we show how to utilize different faiss indexes for evaluation in BEIR. We currently support 
IndexFlatIP, IndexPQ and IndexHNSW from faiss indexes. Faiss indexes are stored and retrieved using the CPU.

Some good notes for information on different faiss indexes can be found here:
1. https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#supported-operations
2. https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization 

For more information, please refer here: https://github.com/facebookresearch/faiss/wiki

PS: You can also save/load your corpus embeddings as a faiss index! Instead of exact search, use FlatIPFaissSearch
which implements exhaustive search using a faiss index.

Usage: python evaluate_faiss_dense.py
"""

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import PQFaissSearch, HNSWFaissSearch, FlatIPFaissSearch, HNSWSQFaissSearch    

import logging
import pathlib, os
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

dataset = "scifact"

#### Download nfcorpus.zip dataset and unzip the dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
out_dir = "/store2/scratch/n3thakur/beir-datasets/"
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Dense Retrieval using Different Faiss Indexes (Flat or ANN) ####
# Provide any Sentence-Transformer or Dense Retriever model.

model_path = "msmarco-distilbert-base-tas-b"
model = models.SentenceBERT(model_path)

########################################################
#### FLATIP: Flat Inner Product (Exhaustive Search) ####
########################################################

faiss_search = FlatIPFaissSearch(model, 
                                 batch_size=128)

######################################################
#### PQ: Product Quantization (Exhaustive Search) ####
######################################################

# faiss_search = PQFaissSearch(model, 
#                              batch_size=128, 
#                              num_of_centroids=96, 
#                              code_size=8)

#####################################################
#### HNSW: Approximate Nearest Neighbours Search ####
#####################################################

# faiss_search = HNSWFaissSearch(model, 
#                                batch_size=128, 
#                                hnsw_store_n=512, 
#                                hnsw_ef_search=128,
#                                hnsw_ef_construction=200)

###############################################################
#### HNSWSQ: Approximate Nearest Neighbours Search with SQ ####
###############################################################

# faiss_search = HNSWSQFaissSearch(model, 
#                                 batch_size=128, 
#                                 hnsw_store_n=128, 
#                                 hnsw_ef_search=128,
#                                 hnsw_ef_construction=200)

#### Load faiss index from file or disk ####
# We need two files to be present within the input_dir!
# 1. input_dir/{prefix}.{ext}.faiss => which loads the faiss index.
# 2. input_dir/{prefix}.{ext}.faiss => which loads mapping of ids i.e. (beir-doc-id \t faiss-doc-id).

prefix = "my-index"       # (default value)
ext = "flat"              # or "pq", "hnsw", "hnsw-sq"
input_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss-index")

if os.path.exists(os.path.join(input_dir, "{}.{}.faiss".format(prefix, ext))):
    faiss_search.load(input_dir=input_dir, prefix=prefix, ext=ext)

#### Retrieve dense results (format of results is identical to qrels)
retriever = EvaluateRetrieval(faiss_search, score_function="dot") # or "cos_sim"
results = retriever.retrieve(corpus, queries)

### Save faiss index into file or disk ####
# Unfortunately faiss only supports integer doc-ids, We need save two files in output_dir.
# 1. output_dir/{prefix}.{ext}.faiss => which saves the faiss index.
# 2. output_dir/{prefix}.{ext}.faiss => which saves mapping of ids i.e. (beir-doc-id \t faiss-doc-id).

prefix = "my-index"      # (default value)
ext = "flat"             # or "pq", "hnsw" 
# output_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss-index")
output_dir = "/store2/scratch/n3thakur/beir-datasets/{}/faiss-index".format(dataset)
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(os.path.join(output_dir, "{}.{}.faiss".format(prefix, ext))):
    faiss_search.save(output_dir=output_dir, prefix=prefix, ext=ext)

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