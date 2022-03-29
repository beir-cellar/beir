"""
This example shows how to evaluate Anserini-BM25 in BEIR.
Since Anserini uses Java-11, we would advise you to use docker for running Pyserini. 
To be able to run the code below you must have docker locally installed in your machine.
To install docker on your local machine, please refer here: https://docs.docker.com/get-docker/

After docker installation, please follow the steps below to get docker container up and running:

1. docker pull beir/pyserini-fastapi 
2. docker build -t pyserini-fastapi .
3. docker run -p 8000:8000 -it --rm pyserini-fastapi

Once the docker container is up and running in local, now run the code below.
This code doesn't require GPU to run.

Usage: python evaluate_anserini_bm25.py
"""

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

import pathlib, os, json
import logging
import requests
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#### Convert BEIR corpus to Pyserini Format #####
pyserini_jsonl = "pyserini.jsonl"
with open(os.path.join(data_path, pyserini_jsonl), 'w', encoding="utf-8") as fOut:
    for doc_id in corpus:
        title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
        data = {"id": doc_id, "title": title, "contents": text}
        json.dump(data, fOut)
        fOut.write('\n')

#### Download Docker Image beir/pyserini-fastapi ####
#### Locally run the docker Image + FastAPI ####
docker_beir_pyserini = "http://127.0.0.1:8000"

#### Upload Multipart-encoded files ####
with open(os.path.join(data_path, "pyserini.jsonl"), "rb") as fIn:
    r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

#### Index documents to Pyserini #####
index_name = "beir/you-index-name" # beir/scifact
r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})

#### Retrieve documents from Pyserini #####
retriever = EvaluateRetrieval()
qids = list(queries)
query_texts = [queries[qid] for qid in qids]
payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}

#### Retrieve pyserini results (format of results is identical to qrels)
results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

#### Retrieve RM3 expanded pyserini results (format of results is identical to qrels)
# results = json.loads(requests.post(docker_beir_pyserini + "/lexical/rm3/batch_search/", json=payload).text)["results"]

#### Check if query_id is in results i.e. remove it from docs incase if it appears ####
#### Quite Important for ArguAna and Quora ####
for query_id in results:
    if query_id in results[query_id]:
        results[query_id].pop(query_id, None)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Retrieval Example ####
query_id, scores_dict = random.choice(list(results.items()))
logging.info("Query : %s\n" % queries[query_id])

scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
for rank in range(10):
    doc_id = scores[rank][0]
    logging.info("Doc %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))