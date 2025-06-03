"""
In this example, we evaluate an embedding model by saving the embeddings as pickle and next dooing search with faiss.
For this you would need to install the faiss-cpu client library. You can install it using `pip install faiss-cpu`.
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
model_name_or_path = "rlhn/e5-base-rlhn-680K"
max_length = 512
pooling = "mean"
normalize = True
query_prompt = "query: "
passage_prompt = "passage: "

### BERT-base (E5-base)
dense_model = models.HuggingFace(
    model_path=model_name_or_path,
    max_length=max_length,
    pooling=pooling,
    normalize=normalize,
    prompts={"query": query_prompt, "passage": passage_prompt},
)

model = DRES(dense_model, batch_size=128)
retriever = EvaluateRetrieval(model, score_function="cos_sim")

#### Save embeddings to disk for later retrieval
embedding_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "embeddings")
os.makedirs(embedding_dir, exist_ok=True)

start_time = time()
### Encode corpus and queries, save them to disk and then retrieve results
results = retriever.encode_and_retrieve(
    corpus=corpus,
    queries=queries,
    encode_output_path=embedding_dir,
    overwrite=False,  # Set to True if you want to overwrite existing embeddings
)
end_time = time()
logging.info(f"Time taken to encode & retrieve: {end_time - start_time:.2f} seconds")

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

#### nDCG@K results should be as follows for the model:
# NDCG@1: 0.5046
# NDCG@3: 0.4467
# NDCG@5: 0.4239
# NDCG@10: 0.3896
# NDCG@100: 0.3596
# NDCG@1000: 0.4441
