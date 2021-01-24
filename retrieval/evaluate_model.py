from loaders.data_loader import GenericDataLoader
from evaluation.evaluation import EvaluateRetrieval

data_path = "/home/thakur/seir/sbert_retriever/datasets-new/nfcorpus"

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
retriever = EvaluateRetrieval(model="sbert", model_name="distilroberta-base-paraphrase-v1")

print(retriever.evaluate(corpus, queries, qrels))