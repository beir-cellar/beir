from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from sentence_transformers import SentenceTransformer
import torch
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

def msmarco_collate_fn(batch):
    # batch is a list of tuples (query, paragraphs, scores)
    queries, paras, scores = zip(*batch)
    return list(queries), list(paras), list(scores)

class MSMARCO_dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        if split == 'train':
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")
        elif split == 'test':
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        self.corpus = corpus

        self.corpus_keys = list(self.corpus.keys())
        self.corpus_values = list(self.corpus.values())
        self.qrels_keys = list(qrels.keys())
        self.qrels_values = list(qrels.values())

    def __getitem__(self, idx):

        query = self.qrels_keys[idx]
        doc2score = self.qrels_values[idx]            # a dict {doc_id: score}
        doc_ids, scores = zip(*doc2score.items())     # two tuples
        paragraphs     = [ self.corpus[d] for d in doc_ids ]
        return query, list(paragraphs), list(scores)

    def __len__(self):
        return len(self.qrels_keys)