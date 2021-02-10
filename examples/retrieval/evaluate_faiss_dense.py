from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalFaissSearch as DRFS

import pathlib, os
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = "nfcorpus.zip"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")


#### Defining our FAISS index
#### Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N)
#### https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
n_clusters = 1024

#### Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.
nprobe = 3

model = DRFS(model=models.SentenceBERT("distilroberta-base-msmarco-v2"), n_clusters=n_clusters, nprobe=nprobe)
retriever = EvaluateRetrieval(model)

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Retrieval Example ####
query_id, scores_dict = random.choice(list(results.items()))
print("Query : %s\n" % queries[query_id])

scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
for rank in range(10):
    print("Doc %d: [%s] - %s\n" % (rank+1, corpus[scores[rank][0]].get("title"), corpus[scores[rank][0]].get("text")))