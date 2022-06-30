'''
This sample python shows how to evaluate BEIR dataset quickly using Mutliple GPU for evaluation (for large datasets).
To run this code, you need Python >= 3.7 (not 3.6) and need to install evaluate library separately: ``pip install evaluate``
Enabling multi-gpu evaluation has been thanks due to tremendous efforts of Noumane Tazi (https://github.com/NouamaneTazi)

IMPORTANT: The following code will not run with Python 3.6! 
1. Please install Python 3.7 using Anaconda (conda create -n myenv python=3.7)
2. Next, install Evaluate (https://github.com/huggingface/evaluate) using ``pip install evaluate``.

You are good to go!

To run this code, you preferably need access to mutliple GPUs. Faster than running on single GPU.
CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_sbert_multi_gpu.py
'''

from collections import defaultdict
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader_hf import HFDataLoader
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import time

import logging
import pathlib, os
import random

#### Just some code to print debug information to stdout
logging.basicConfig(level=logging.INFO)
#### /print debug information to stdout


#Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == "__main__":

    tick = time.time()

    dataset = "nfcorpus"
    keep_in_memory = False
    streaming = False
    corpus_chunk_size = 2048
    batch_size = 256 # sentence bert model batch size
    model_name = "msmarco-distilbert-base-tas-b"
    target_devices = None # ['cpu']*2

    corpus, queries, qrels = HFDataLoader(hf_repo=f"BeIR/{dataset}", streaming=streaming, keep_in_memory=keep_in_memory).load(split="test")

    #### Dense Retrieval using SBERT (Sentence-BERT) ####
    #### Provide any pretrained sentence-transformers model
    #### The model was fine-tuned using cosine-similarity.
    #### Complete list - https://www.sbert.net/docs/pretrained_models.html
    beir_model = models.SentenceBERT(model_name)

    #### Start with Parallel search and evaluation
    model = DRPES(beir_model, batch_size=batch_size, target_devices=target_devices, corpus_chunk_size=corpus_chunk_size)
    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time.time()
    results = retriever.retrieve(corpus, queries)
    end_time = time.time()
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))

    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
    recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
    hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

    tock = time.time()
    print("--- Total time taken: {:.2f} seconds ---".format(tock - tick))

    #### Print top-k documents retrieved ####
    top_k = 10

    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    query = queries.filter(lambda x: x['id']==query_id)[0]['text']
    logging.info("Query : %s\n" % query)

    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        doc = corpus.filter(lambda x: x['id']==doc_id)[0]
        # Format: Rank x: ID [Title] Body
        logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, doc.get("title"), doc.get("text")))