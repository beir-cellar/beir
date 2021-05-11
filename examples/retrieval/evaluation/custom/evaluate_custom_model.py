from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import List, Dict

import logging
import numpy as np
import pathlib, os
import random

class YourCustomModel:
    def __init__(self, model_path=None, **kwargs):
        self.model = None # ---> HERE Load your custom model
        # self.model = SentenceTransformer(model_path)
    
    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    # For eg ==> return np.asarray(self.model.encode(queries, batch_size=batch_size, **kwargs))
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        pass
    
    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    # For eg ==> sentences = [(doc["title"] + "  " + doc["text"]).strip() for doc in corpus]
    #        ==> return np.asarray(self.model.encode(sentences, batch_size=batch_size, **kwargs))
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        pass


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

#### Provide your custom model class name --> HERE
model = DRES(YourCustomModel(model_path="your-custom-model-path"))

retriever = EvaluateRetrieval(model, score_function="cos_sim") # or "dot" if you wish dot-product

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
