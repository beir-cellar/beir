"""
Allows for evaluating dense retrievers, i.e., in the tevatron format on BEIR datasets.
models.HuggingFace allows for multi-gpu inference with DDP.

Example usage: CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_huggingface.py (for multi-gpu inference)
"""

import logging
import os
import pathlib
import random
from time import time

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

dataset = "nfcorpus"

#### Download nfcorpus.zip dataset and unzip the dataset
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files:
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Dense Retrieval using E5 or Tevatron with Hugging Face ####
#### Provide any pretrained E5 or Tevatron fine-tuned model
#### The model was fine-tuned using normalization & cosine-similarity.

## Parameters
model_name_or_path = "intfloat/e5-mistral-7b-instruct"
max_length = 512
pooling = "eos"
normalize = True
append_eos_token = True

#### Configuration for E5-Mistral
# Check prompts: https://github.com/microsoft/unilm/blob/9c0f1ff7ca53431fe47d2637dfe253643d94185b/e5/utils.py
query_prompt = "Instruct: Given a question, retrieve relevant documents that best answer the question\nQuery: "
passage_prompt = ""
dense_model = models.HuggingFace(
    model_path=model_name_or_path,
    max_length=max_length,
    append_eos_token=append_eos_token,  # add [EOS] token to the end of the input
    pooling=pooling,
    normalize=normalize,
    prompts={"query": query_prompt, "passage": passage_prompt},
    attn_implementation="flash_attention_2",
    torch_dtype="bfloat16",
)

model = DRES(dense_model, batch_size=128)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

#### Retrieve dense results (format of results is identical to qrels)
start_time = time()
results = retriever.retrieve(corpus, queries)
end_time = time()
print(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")
#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

### If you want to save your results and runfile (useful for reranking)
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info(f"Query : {queries[query_id]}\n")

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info(f"Rank {rank + 1}: {doc_id} [{corpus[doc_id].get('title')}] - {corpus[doc_id].get('text')}\n")

### NDCG@K results should look like this:
# NDCG@1: 0.4830
# NDCG@3: 0.4287
# NDCG@5: 0.4102
# NDCG@10: 0.3845
# NDCG@100: 0.3520
# NDCG@1000: 0.4360
