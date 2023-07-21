"""
The pre-trained models produce embeddings of size 512 - 1024. However, when storing a large
number of embeddings, this requires quite a lot of memory / storage.

In this example, we convert float embeddings to binary hashes using binary passage retriever (BPR).
This significantly reduces the required memory / storage while maintaining nearly the same performance.

For more information, please refer to the publication by Yamada et al. in ACL 2021 -
Efficient Passage Retrieval with Hashing for Open-domain Question Answering, (https://arxiv.org/abs/2106.00882)

For computing binary hashes, we need to train a model with bpr loss function (Margin Ranking Loss + Cross Entropy Loss).
For more details on training, check train_msmarco_v3_bpr.py on how to train a binary retriever model.

BPR model encoders vectors to 768 dimensions of binary values {1,0} of 768 dim. We pack 8 bits into bytes, this
further allows a 768 dim (bit) vector to 96 dim byte (int-8) vector. 
for more details on packing refer here: https://numpy.org/doc/stable/reference/generated/numpy.packbits.html

Hence, the new BPR model will produce directly binary hash embeddings without further changes needed. And we 
evaluate the BPR model using BinaryFlat Index in faiss, which computes hamming distance between bits to find top-k
similarity results. We also rerank top-1000 retrieved from faiss documents with the original query embedding (float)!

The Reranking step is very efficient and fast (as reranking is done by a bi-encoder), hence we advise to rerank 
with top-1000 docs retrieved by hamming distance to decrease the loss in performance!

'''
model = models.BinarySentenceBERT("msmarco-distilbert-base-tas-b")
test_corpus = [{"title": "", "text": "Python is a programming language"}]
print(model.encode_corpus(test_corpus))

>> [[195  86 160 203 135  39 155 173  89 100 107 159 112  94 144  60  57 148
  205  15 204 221 181 132 183 242 122  48 108 200  74 221  48 250  12   4
  182 165  36  72 101 169 137 227 192 109 136  18 145   5 104   5 221 195
   45 254 226 235 109   3 209 156  75 238 143  56  52 227  39   1 144 214
  142 120 181 204 166 221 179  88 142 223 110 255 105  44 108  88  47  67
  124 126 117 159  37 217]]
'''

Usage: python evaluate_bpr.py
"""

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

dataset = "msmarco"

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

model_name="income/bpr-gpl-trec-covid-base-msmarco-distilbert-tas-b"
model = models.BinarySentenceBERT(model_name) # Proxy for now, soon coming up BPR models trained on MSMARCO!
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

out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", dataset)
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, model_name.replace("/", "_") + "_results.txt"), "w+") as fOut:
  fOut.write("\nNDCG@K\n")
  fOut.write("--------\n")
  for top_k in ndcg:
    fOut.write(top_k + ":\t" + str(ndcg[top_k]) + "\n")
  
  fOut.write("\nMAP@K\n")
  fOut.write("--------\n")
  for top_k in _map:
    fOut.write(top_k + ":\t" + str(_map[top_k]) + "\n")
  
  fOut.write("\nRecall@K\n")
  fOut.write("--------\n")
  for top_k in recall:
    fOut.write(top_k + ":\t" + str(recall[top_k]) + "\n")
  
  fOut.write("\nPrecision@K\n")
  fOut.write("--------\n")
  for top_k in precision:
    fOut.write(top_k + ":\t" + str(precision[top_k]) + "\n")
  
  fOut.write("\nMRR@k\n")
  fOut.write("--------\n")
  for top_k in mrr:
    fOut.write(top_k + ":\t" + str(mrr[top_k]) + "\n")
  
  fOut.write("\nR_cap@k\n")
  fOut.write("--------\n")
  for top_k in recall_cap:
    fOut.write(top_k + ":\t" + str(recall_cap[top_k]) + "\n")

  query_id, ranking_scores = random.choice(list(results.items()))
  scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
  logging.info("Query : %s\n" % queries[query_id])
  fOut.write("\n\nQuery : %s\n" % queries[query_id])

  for rank in range(10):
      doc_id = scores_sorted[rank][0]
      # Format: Rank x: ID [Title] Body
      fOut.write("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
      logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))
