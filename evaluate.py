from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer

import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "MSMARCO"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
# out_dir = os.path.join('/data/richard/taggerv2/test/test6/beir/outputs', "datasets")
data_path = '/data/richard/taggerv2/test/test6/beir/outputs/datasets/msmarco'

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

#### Load the SBERT model and retrieve using cosine-similarity
# model = DRES(models.SentenceBERT("Alibaba-NLP/gte-modernbert-base"), batch_size=16)
model = DRES(models.SentenceBERT("BAAI/bge-large-en-v1.5"))
# model = DRES(models.SentenceBERT('sentence-transformers/gtr-t5-xl'))

#### Or load models directly from HuggingFace
# model = DRES(models.HuggingFace(
#     "intfloat/e5-large-unsupervised",
#     max_length=512,
#     pooling="mean",
#     normalize=True,
#     prompts={"query": "query: ", "passage": "passage: "}), batch_size=16)

# model = SentenceTransformer('sentence-transformers/gtr-t5-xl')      # gtr-t5-xl

retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "dot" for dot product
results = retriever.retrieve(corpus, queries)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000]
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
print(f'ndcg, _map, recall, precision: {ndcg, _map, recall, precision}')
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

### If you want to save your results and runfile (useful for reranking)
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)