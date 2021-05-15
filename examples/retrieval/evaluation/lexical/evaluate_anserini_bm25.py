"""
This example shows how to evaluate Anserini-BM25 in BEIR.
Since Anserini uses Java-11, we would advise you to use docker for running Pyserini. 
To be able to run the code below you must have docker locally installed in your machine.

To install docker on your local machine, please refer here: https://docs.docker.com/get-docker/
Additionally, install docker for this example using ``pip install docker``.
You need to install both!

After docker installation, please follow the steps below to get docker container up and running:

1. Download beir-pyserini docker image (link: TODO)
2. cd beir-pyserini
3. docker build -t beir-pyserini .
4. docker run -p 8000:8000 -it --name docker-beir-pyserini --rm beir-pyserini

Once the docker container is up and running in local, now run the code below.

Important thing to remember docker cannot access local files! 
So we copy (similar to docker cp) local file onto the mounted docker filesystem.
For convinience, we are using docker-py client here below. One can also work with Terminal.

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
import tarfile
import docker

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Load docker using docker-client
client = docker.from_env()

#### Mount local files (src) to docker (dst) #### 
def copy_to(src, dst):
    name, dst = dst.split(':')
    container = client.containers.get(name)

    os.chdir(os.path.dirname(src))
    srcname = os.path.basename(src)
    tar = tarfile.open(src + '.tar', mode='w')
    try:
        tar.add(srcname)
    finally:
        tar.close()

    data = open(src + '.tar', 'rb').read()
    container.put_archive(os.path.dirname(dst), data)

#### Download scifact.zip dataset and unzip the dataset
dataset = "scifact"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

#### Convert BEIR corpus to Pyserini #####
pyserini_output = "pyserini.jsonl"
with open(os.path.join(data_path, pyserini_output),'w', encoding="utf-8") as fOut:
    for doc_id in corpus:
        title, text = corpus[doc_id].get("title", ""), corpus[doc_id].get("text", "")
        data = {"id": doc_id, "contents": title + " " + text}
        json.dump(data, fOut)
        fOut.write('\n')

#### Mount Pyserini Data to Docker #####
container_name_or_id = "docker-beir-pyserini" # or provide id, for eg. "64c9fbfbd9e4"
container_datapath = "/home/datasets"
# local path ----> docker path (my-container-name:/home/datasets/pyserini.jsonl)
# make sure destination directory is already present in container 
copy_to(os.path.join(data_path, pyserini_output), "{}:{}/{}".format(container_name_or_id, container_datapath, pyserini_output))

#### Index documents to Pyserini #####
index_name = "beir/your-index-name" # beir/scifact
index_endpoint = "http://127.0.0.1:8000/index/" # for indexing
# provide data folder path where "pyserini.jsonl" is mounted in docker
params = {"name": index_name, "data_folder": container_datapath}
requests.get(index_endpoint, params=params)

#### Retrieve documents from Pyserini #####
retriever = EvaluateRetrieval()
search_endpoint = "http://127.0.0.1:8000/batch_search/"
qids = list(queries)
query_texts = [queries[qid] for qid in qids]
payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values)}

#### Retrieve pyserini results (format of results is identical to qrels)
results = json.loads(requests.post(search_endpoint, json=payload).text)["results"]

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