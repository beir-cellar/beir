from beir import LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import pathlib, os
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### METHOD 2 ####

# Provide the path to your CORPUS file, it should be jsonlines format (ref: https://jsonlines.org/)
# Saved corpus file must have .jsonl extension (for eg: your_corpus_file.jsonl)
# Corpus file structure: 
# [
#   {"_id": "doc1", "title": "Albert Einstein", "text": "Albert Einstein was a German-born...."},
#   {"_id": "doc2", "title": "", "text": "Wheat beer is a top-fermented beer...."}},
#   ....
# ]
corpus_path = "/home/thakur/your-custom-dataset/your_corpus_file.jsonl"

# Provide the path to your QUERY file, it should be jsonlines format (ref: https://jsonlines.org/)
# Saved query file must have .jsonl extension (for eg: your_query_file.jsonl)
# Query file structure: 
# [
#   {"_id": "q1", "text": "Who developed the mass-energy equivalence formula?"},
#   {"_id": "q2", "text": "Which beer is brewed with a large proportion of wheat?"},
#   ....
# ]
query_path = "/home/thakur/your-custom-dataset/your_query_file.jsonl"

# Provide the path to your QRELS file, it should be tsv or tab-seperated format.
# Saved qrels file must have .tsv extension (for eg: your_qrels_file.tsv)
# Qrels file structure: (Keep 1st row as header)
# query-id  corpus-id   score
# q1    doc1    1
# q2    doc2    1
# ....
qrels_path = "/home/thakur/your-custom-dataset/your_qrels_file.tsv"

# Load using load_custom function in GenericDataLoader
corpus, queries, qrels = GenericDataLoader(
    corpus_file=corpus_path,
    query_file=query_path,
    qrels_file=qrels_path).load_custom()

#### Sentence-Transformer ####
#### Provide any pretrained sentence-transformers model path
#### Complete list - https://www.sbert.net/docs/pretrained_models.html
model = DRES(models.SentenceBERT("msmarco-distilbert-base-v3"))

retriever = EvaluateRetrieval(model, score_function="cos_sim")

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)