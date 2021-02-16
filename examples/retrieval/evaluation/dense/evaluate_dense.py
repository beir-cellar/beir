from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os
import random

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = "nq.zip"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where nfcorpus has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Sentence-Transformer ####
#### Provide any pretrained sentence-transformers model path
#### Complete list - https://www.sbert.net/docs/pretrained_models.html
model = DRES(models.SentenceBERT("distilroberta-base-msmarco-v2"))

#### DPR ####
#### Use the DPR NQ trained question and context encoder
#### For more details - https://huggingface.co/transformers/model_doc/dpr.html
# model = DRES(models.DPR(
#     'facebook/dpr-question_encoder-single-nq-base',
#     'facebook/dpr-ctx_encoder-single-nq-base'
#     ))

#### USE-QA ####
#### We use the English USE-QA v3 and provide the tf hub url
#### Link: https://tfhub.dev/google/universal-sentence-encoder-qa/3
# model = DRES(models.UseQA("https://tfhub.dev/google/universal-sentence-encoder-qa/3"))

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