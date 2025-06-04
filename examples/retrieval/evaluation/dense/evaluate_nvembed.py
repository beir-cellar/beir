"""
This script evaluates the nvidia/NV-Embed-v2 model on the TREC-COVID dataset.
Important to note that the nvidia/NV-Embed-v2 model does not work on the latest transformers versions (4.49.0): https://huggingface.co/nvidia/NV-Embed-v2/discussions/37.
So, to be able to run the script you need to install an older version of transformers, e.g.: 4.46.2.
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

dataset = "trec-covid"

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

#### Dense Retrieval using nvidia/NV-Embed-v2 ####
#### Provide nvidia/NV-Embed-v2 model: https://huggingface.co/nvidia/NV-Embed-v2
#### The model was fine-tuned using normalization & cosine-similarity.

## Parameters
model_name_or_path = "nvidia/NV-Embed-v2"
max_length = 512
pooling = "mean"
normalize = True

# Checkout prompts for NV-Embed-v2 model inside `instructions.json`.
# https://huggingface.co/nvidia/NV-Embed-v2/blob/main/instructions.json
trec_covid_prompt = "Given a query on COVID-19, retrieve documents that answer the query"

#### Load the Dense Retriever model (NVEmbed)
dense_model = models.NVEmbed(
    model_name_or_path,
    max_length=max_length,
    pooling=pooling,
    normalize=normalize,
    prompts={"query": trec_covid_prompt, "passage": ""},
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
