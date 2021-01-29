import pathlib, os
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models
from beir import util

dataset = "nfcorpus.zip"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")

data_path = util.download_and_unzip(url, out_dir)
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

retriever = EvaluateRetrieval(models.SentenceBERT("distilroberta-base-msmarco-v2"))

results = retriever.retrieve(corpus, queries, qrels)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
print(ndcg, _map, recall, precision)