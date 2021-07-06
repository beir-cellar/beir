"""
The pre-trained models produce embeddings of size 512 - 1024. However, when storing a large
number of embeddings, this requires quite a lot of memory / storage.

In this example, we reduce the dimensionality of the embeddings to e.g. 128 dimensions. This significantly
reduces the required memory / storage while maintaining nearly the same performance.

For dimensionality reduction, we compute embeddings for a large set of (representative) sentence. Then,
we use PCA to find e.g. 128 principle components of our vector space. This allows us to maintain
us much information as possible with only 128 dimensions.

PCA gives us a matrix that down-projects vectors to 128 dimensions. We use this matrix
and extend our original SentenceTransformer model with this linear downproject. Hence,
the new SentenceTransformer model will produce directly embeddings with 128 dimensions
without further changes needed. 

Usage: python evaluate_dim_reduction.py
"""

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import PCAFaissSearch

import logging
import pathlib, os
import random
import faiss

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

# Dense Retrieval using Different Faiss Indexes (Flat or ANN) ####
# Provide any Sentence-Transformer or Dense Retriever model.

model_path = "msmarco-distilbert-base-tas-b"
model = models.SentenceBERT(model_path)

###############################################################
#### PCA: Principal Component Analysis (Exhaustive Search) ####
###############################################################
# Reduce Input Dimension (768) to output dimension of (128)

output_dimension = 128
base_index = faiss.IndexFlatIP(output_dimension)
faiss_search = PCAFaissSearch(model,
                              base_index=base_index,
                              output_dimension=output_dimension,
                              batch_size=128)

#######################################################################
#### PCA: Principal Component Analysis (with Product Quantization) ####
#######################################################################
# Reduce Input Dimension (768) to output dimension of (96)

# output_dimension = 96
# base_index = faiss.IndexPQ(output_dimension,               # output dimension
#                              96,                           # number of centroids
#                              8,                            # code size
#                              faiss.METRIC_INNER_PRODUCT)   # similarity function
                            
# faiss_search = PCAFaissSearch(model,
#                               base_index=base_index,
#                               output_dimension=output_dimension,
#                               batch_size=128)

#### Load faiss index from file or disk ####
# We need two files to be present within the input_dir!
# 1. input_dir/{prefix}.{ext}.faiss => which loads the faiss index.
# 2. input_dir/{prefix}.{ext}.faiss => which loads mapping of ids i.e. (beir-doc-id \t faiss-doc-id).

prefix = "my-index"       # (default value)
ext = "pca"               # extension

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

prefix = "my-index"     # (default value)
ext = "pca"             # extension

output_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss-index")
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