from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval


data_path = "../datasets/nfcorpus"

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
retriever = EvaluateRetrieval(model="sbert", model_name="distilroberta-base-paraphrase-v1")
ndcg, _map, recall = retriever.evaluate(corpus, queries, qrels)
print(ndcg, _map, recall)