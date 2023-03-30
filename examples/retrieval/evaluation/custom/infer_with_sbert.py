from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import random
import json

# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger...")
# debugpy.wait_for_client()
# print("Debugger Attached.")

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
data_path = "/data/ashok-4983/zlabsnlp/znlp_vectorizer/data/support_data/benchmark_format"

#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files: 
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Dense Retrieval using SBERT (Sentence-BERT) ####
#### Provide any pretrained sentence-transformers model
#### The model was fine-tuned using cosine-similarity.
#### Complete list - https://www.sbert.net/docs/pretrained_models.html

model = DRES(models.SentenceBERT("msmarco-distilbert-base-tas-b"))
retriever = EvaluateRetrieval(model, score_function="cos_sim",k_values=[1])

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K, Recall@K and P@K

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

#### Evaluate your retreival using MRR@K, Recall_cap@K, Hole@K
mrr = retriever.evaluate_custom(qrels,results,k_values=[1],metric="mrr")

result_map = {}
for q, docs in results.items():
    max_score = 0
    chosen_doc = None
    for doc, score in docs.items():
        if score > max_score:
            max_score = score
            chosen_doc = doc
    result_map[q] = chosen_doc


metrics = {"ndcg":ndcg["NDCG@1"],
           "MAP":_map["MAP@1"],
           "Recall":recall["Recall@1"],
           "Precision":precision["P@1"],
           "MRR":mrr["MRR@1"]}
 
with open("/data/ashok-4983/zlabsnlp/znlp_vectorizer/experiments/fine_tuning_on_support_data/benchmark/distil_bert_cos_sim_result_map.json", "w") as outfile:
    json.dump(result_map, outfile)

# with open("/data/ashok-4983/zlabsnlp/znlp_vectorizer/experiments/fine_tuning_on_support_data/benchmark/fine_tuned_distil_bert_cos_sim_result.tsv", "w") as f:
#     for metric in metrics:
#         f.write(metric)
#         f.write("\t")
#     f.write("\n")
#     for val in metrics.values():
#         f.write(str(val))
#         f.write("\t")

