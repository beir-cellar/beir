"""
This example shows how to evaluate DeepCT (using Anserini) in BEIR.
For more details on DeepCT, refer here: https://arxiv.org/abs/1910.10687

The original DeepCT repository is not modularised and only works with Tensorflow 1.x (1.15).
We modified the DeepCT repository to work with Tensorflow latest (2.x).
We do not change the core-prediction code, only few input/output file format and structure to adapt to BEIR formats.
For more details on changes, check: https://github.com/NThakur20/DeepCT and compare it with original repo!

Please follow the steps below to install DeepCT:

1. git clone https://github.com/NThakur20/DeepCT.git

Since Anserini uses Java-11, we would advise you to use docker for running Pyserini. 
To be able to run the code below you must have docker locally installed in your machine.
To install docker on your local machine, please refer here: https://docs.docker.com/get-docker/

After docker installation, please follow the steps below to get docker container up and running:

1. docker pull docker pull beir/pyserini-fastapi
2. docker build -t pyserini-fastapi .
3. docker run -p 8000:8000 -it --rm pyserini-fastapi 

Usage: python evaluate_deepct.py
"""
from DeepCT.deepct import run_deepct                            # git clone https://github.com/NThakur20/DeepCT.git

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.generation.models import QGenModel
from tqdm.autonotebook import trange

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

#### 1. Download Google BERT-BASE, Uncased model ####
# Ref: https://github.com/google-research/bert

base_model_url = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "models")
bert_base_dir = util.download_and_unzip(base_model_url, out_dir)

#### 2. Download DeepCT MSMARCO Trained BERT checkpoint ####
# Credits to DeepCT authors: Zhuyun Dai, Jamie Callan, (https://github.com/AdeDZY/DeepCT)

model_url = "http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/outputs/marco.zip" 
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "models")
checkpoint_dir = util.download_and_unzip(model_url, out_dir)

##################################################
#### 3. Configure Params for DeepCT inference ####
##################################################
# We cannot use the original Repo (https://github.com/AdeDZY/DeepCT) as it only runs with TF 1.15.
# We reformatted the code (https://github.com/NThakur20/DeepCT) and made it working with latest TF 2.X!

if not os.path.isfile(os.path.join(data_path, "deepct.jsonl")):
    ################################
    #### Command-Line Arugments ####
    ################################
    run_deepct.FLAGS.task_name = "beir"                                                     # Defined a seperate BEIR task in DeepCT. Check out run_deepct.
    run_deepct.FLAGS.do_train = False                                                       # We only want to use the code for inference.
    run_deepct.FLAGS.do_eval = False                                                        # No evaluation.
    run_deepct.FLAGS.do_predict = True                                                      # True, as we would use DeepCT model for only prediction.
    run_deepct.FLAGS.data_dir = os.path.join(data_path, "corpus.jsonl")                     # Provide original path to corpus data, follow beir format.
    run_deepct.FLAGS.vocab_file = os.path.join(bert_base_dir, "vocab.txt")                  # Provide bert-base-uncased model vocabulary.
    run_deepct.FLAGS.bert_config_file = os.path.join(bert_base_dir, "bert_config.json")     # Provide bert-base-uncased config.json file.
    run_deepct.FLAGS.init_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-65816")     # Provide DeepCT MSMARCO model (bert-base-uncased) checkpoint file.
    run_deepct.FLAGS.max_seq_length = 350                                                   # Provide Max Sequence Length used for consideration. (Max: 512)
    run_deepct.FLAGS.train_batch_size = 128                                                 # Inference batch size, Larger more Memory but faster!
    run_deepct.FLAGS.output_dir = data_path                                                 # Output directory, this will contain two files: deepct.jsonl (output-file) and predict.tf_record
    run_deepct.FLAGS.output_file = "deepct.jsonl"                                           # Output file for storing final DeepCT produced corpus.
    run_deepct.FLAGS.m = 100                                                                # Scaling parameter for DeepCT weights: scaling parameter > 0, recommend 100
    run_deepct.FLAGS.smoothing = "sqrt"                                                     # Use sqrt to smooth weights. DeepCT Paper uses None.
    run_deepct.FLAGS.keep_all_terms = True                                                  # Do not allow DeepCT to delete terms.

    # Runs DeepCT model on the corpus.jsonl
    run_deepct.main()

#### Download Docker Image beir/pyserini-fastapi ####
#### Locally run the docker Image + FastAPI ####
docker_beir_pyserini = "http://127.0.0.1:8000"

#### Upload Multipart-encoded files ####
with open(os.path.join(data_path, "deepct.jsonl"), "rb") as fIn:
    r = requests.post(docker_beir_pyserini + "/upload/", files={"file": fIn}, verify=False)

#### Index documents to Pyserini #####
index_name = "beir/you-index-name" # beir/scifact
r = requests.get(docker_beir_pyserini + "/index/", params={"index_name": index_name})

######################################
#### 2. Pyserini-Retrieval (BM25) ####
######################################

#### Retrieve documents from Pyserini #####
retriever = EvaluateRetrieval()
qids = list(queries)
query_texts = [queries[qid] for qid in qids]
payload = {"queries": query_texts, "qids": qids, "k": max(retriever.k_values), 
           "fields": {"contents": 1.0}, "bm25": {"k1": 18, "b": 0.7}}

#### Retrieve pyserini results (format of results is identical to qrels)
results = json.loads(requests.post(docker_beir_pyserini + "/lexical/batch_search/", json=payload).text)["results"]

#### Retrieve RM3 expanded pyserini results (format of results is identical to qrels)
# results = json.loads(requests.post(docker_beir_pyserini + "/lexical/rm3/batch_search/", json=payload).text)["results"]

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
