from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalBinaryCodeSearch as DRBCS

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

# Dense Retrieval with Hamming Distance with Binary-Code-SBERT (Sentence-BERT) ####
# Provide any Binary Passage Retriever Trained model.
# The model was fine-tuned using CLS Pooling and dot-product!
# Open-sourced binary code SBERT model trained on MSMARCO to be made available soon!

model_path = "distilbert-base-uncased" # To be coming soon! 
model = DRBCS(models.BinarySentenceBERT(model_path), batch_size=16)
retriever = EvaluateRetrieval(model)

# BPR first retrieves binary_k (default 1000) documents based on query hash and document hash similarity with hamming distance.
# The hamming distance similarity is constructed using IndexBinaryHash in Faiss: https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexBinaryHash.html
# BPR then reranks with similarity b/w query embedding and the documents hashes for these binary_k documents.
# Reranking is advised as its quite fast, efficient and leads to decent performances.

rerank = True                       # False would only retrieve top-k documents based on hamming distance.
binary_k = 1000                     # binary_k value denotes documents reranked for each query.
index = True                        # True would index all documents first to faiss (faiss::IndexBinaryHash).

results = retriever.retrieve(corpus, queries, rerank=rerank, binary_k=binary_k, index=index)

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