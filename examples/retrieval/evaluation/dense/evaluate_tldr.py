'''
In this example, we show how to evaluate TLDR: Twin Learning Dimensionality Reduction using the BEIR Benchmark.
TLDR is a unsupervised dimension reduction technique, which performs better in comparsion with commonly known: PCA.

In order to run and evaluate the model, it's important to first install the tldr original repository.
This can be installed conviniently using "pip install tldr". 

However, please refer here: https://github.com/naver/tldr for all requirements!
'''

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from sentence_transformers import SentenceTransformer
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os, sys
import numpy as np
import torch
import random
import importlib.util

if importlib.util.find_spec("tldr") is not None:
    from tldr import TLDR

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

dataset = "nfcorpus"

# #### Download nfcorpus.zip dataset and unzip the dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# Get all the corpus documents as a list for tldr training 
corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
corpus_list = [corpus[cid] for cid in corpus_ids]

# Dense Retrieval with Dimension Reduction with TLDR: Twin Learning Dimensionality Reduction ####
# TLDR is a dimensionality reduction technique which has been shown to perform better compared to PCA
# For more details, please refer to the publication: (https://arxiv.org/pdf/2110.09455.pdf)
# https://europe.naverlabs.com/research/publications/tldr-twin-learning-for-dimensionality-reduction/

# First load the SBERT model, which will be used to create embeddings
model_path = "sentence-transformers/msmarco-distilbert-base-tas-b"

# Create the TLDR model instance providing the SBERT model path
tldr = models.TLDR(
    encoder_model=SentenceTransformer(model_path),
    n_components=128,
    n_neighbors=5,
    encoder="linear",
    projector="mlp-1-2048",
    verbose=2,
    knn_approximation=None,
    output_folder="data/"
)

# Starting to train the TLDR model with TAS-B model on the target dataset: nfcorpus
tldr.fit(corpus=corpus_list, batch_size=128, epochs=100, warmup_epochs=10, train_batch_size=1024, print_every=100)
logging.info("TLDR model training completed\n")

# You can also save the trained model in the following path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "tldr", "inference_model.pt") 
logging.info("TLDR model saved here: %s\n" % model_save_path)
tldr.save(model_save_path)

# You can again load back the trained model using the code below:
tldr = TLDR()
tldr.load(model_save_path, init=True)  # Loads both model parameters and weights

# Now we evaluate the TLDR model using dense retrieval with dot product
retriever = EvaluateRetrieval(DRES(tldr), score_function="dot")

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

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