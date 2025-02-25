"""
This sample python shows how to evaluate BEIR dataset quickly using Mutliple GPU for evaluation (for large datasets).
To run this code, you need Python >= 3.7 (not 3.6)
Enabling multi-gpu evaluation has been thanks due to tremendous efforts of Noumane Tazi (https://github.com/NouamaneTazi)

IMPORTANT: The following code will not run with Python 3.6!
1. Please install Python 3.7 using Anaconda (conda create -n myenv python=3.7)

You are good to go!

To run this code, you preferably need access to mutliple GPUs. Faster than running on single GPU.
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 evaluate_sbert_multi_gpu.py
"""

import logging
import os
import pathlib
import random
import time

import torch
from torch import distributed as dist

from beir import util
from beir.datasets.data_loader_hf import HFDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalParallelExactSearch as DRPES

if __name__ == "__main__":
    # Initialize torch.distributed
    dist.init_process_group("nccl")
    device_id = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(torch.cuda.device(device_id))

    # Enable logging only first rank=0
    rank = int(os.getenv("RANK", 0))
    if rank != 0:
        logging.basicConfig(level=logging.WARN)
    else:
        logging.basicConfig(level=logging.INFO)

    tick = time.time()

    dataset = "nfcorpus"
    keep_in_memory = False
    streaming = False
    corpus_chunk_size = 2048
    batch_size = 256  # sentence bert model batch size
    ignore_identical_ids = True

    corpus, queries, qrels = HFDataLoader(
        hf_repo=f"BeIR/{dataset}", streaming=streaming, keep_in_memory=keep_in_memory
    ).load(split="test")

    #### Dense Retrieval using SBERT (Sentence-BERT) ####
    #### Provide any pretrained sentence-transformers model
    #### The model was fine-tuned using cosine-similarity.
    #### Complete list - https://www.sbert.net/docs/pretrained_models.html
    model_name_or_path = "NovaSearch/stella_en_1.5B_v5"
    max_length = 512
    query_prompt_name = "s2p_query"

    dense_model = models.SentenceBERT(
        model_name_or_path,
        max_length=max_length,
        prompt_names={"query": query_prompt_name, "passage": None},
        trust_remote_code=True,
        cache_folder="/mnt/users/n3thakur/cache",
    )

    #### Start with Parallel search and evaluation
    model = DRPES(dense_model, batch_size=batch_size, corpus_chunk_size=corpus_chunk_size)
    retriever = EvaluateRetrieval(model, score_function="dot")

    #### Retrieve dense results (format of results is identical to qrels)
    start_time = time.time()
    results = retriever.retrieve(corpus, queries, ignore_identical_ids=ignore_identical_ids)
    end_time = time.time()
    print(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

    #### Evaluate your retrieval using NDCG@k, MAP@K ...

    logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
    ndcg, _map, recall, precision = retriever.evaluate(
        qrels, results, retriever.k_values, ignore_identical_ids=ignore_identical_ids
    )
    mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

    ### If you want to save your results and runfile (useful for reranking)
    results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    os.makedirs(results_dir, exist_ok=True)

    #### Save the evaluation runfile & results
    util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
    util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)

    tock = time.time()
    print(f"--- Total time taken: {tock - tick:.2f} seconds ---")

    #### Print top-k documents retrieved ####
    top_k = 10

    query_id, ranking_scores = random.choice(list(results.items()))
    scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
    query = queries.filter(lambda x: x["id"] == query_id)[0]["text"]
    logging.info(f"Query : {query}\n" % query)

    for rank in range(top_k):
        doc_id = scores_sorted[rank][0]
        doc = corpus.filter(lambda x: x["id"] == doc_id)[0]
        # Format: Rank x: ID [Title] Body
        logging.info(f"Rank {rank + 1}: {doc_id} [{doc.get('title')}] - {doc.get('text')}\n")
