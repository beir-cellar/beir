from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import BinaryFaissSearch

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

model = models.BinarySentenceBERT("msmarco-distilbert-base-tas-b") # Proxy for now, soon coming up BPR models trained on MSMARCO!
faiss_search = BinaryFaissSearch(model, batch_size=128)

#### Load faiss index from file or disk ####
# We need two files to be present within the input_dir!
# 1. input_dir/my-index.bin.faiss ({prefix}.{ext}.faiss) which loads the faiss index.
# 2. input_dir/my-index.bin.tsv ({prefix}.{ext}.faiss) which loads mapping of ids i.e. (beir-doc-id \t faiss-doc-id).

prefix = "my-index"       # (default value)
ext = "bin"               # bin for binary (default value)
input_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss-index")

if os.path.isdir(input_dir):
    faiss_search.load(input_dir=input_dir, prefix=prefix, ext=ext)

# BPR first retrieves binary_k (default 1000) documents based on query hash and document hash similarity with hamming distance!
# The hamming distance similarity is constructed using IndexBinaryFlat in Faiss.
# BPR then reranks with dot similarity b/w query embedding and the documents hashes for these binary_k documents.
# Please Note, Reranking here is done with a bi-encoder which is quite faster compared to cross-encoders.
# Reranking is advised by the original paper as its quite fast, efficient and leads to decent performances.

score_function = "dot" # or cos_sim for cosine similarity
retriever = EvaluateRetrieval(faiss_search, score_function=score_function)

rerank = True                       # False would only retrieve top-k documents based on hamming distance.
binary_k = 1000                     # binary_k value denotes documents reranked for each query.

results = retriever.retrieve(corpus, queries, rerank=rerank, binary_k=binary_k)

### Save faiss index into file or disk ####
# Unfortunately faiss only supports integer doc-ids!
# This will mean we need save two files in your output_dir path =>
# 1. output_dir/my-index.bin.faiss ({prefix}.{ext}.faiss) which saves the faiss index.
# 2. output_dir/my-index.bin.tsv ({prefix}.{ext}.faiss) which saves mapping of ids i.e. (beir-doc-id \t faiss-doc-id).

prefix = "my-index"
ext = "bin"
output_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss-index")

os.makedirs(output_dir, exist_ok=True)
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