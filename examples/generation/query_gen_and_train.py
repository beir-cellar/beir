from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
from beir.retrieval.train import TrainRetriever

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
corpus = GenericDataLoader(data_path).load_corpus()

#### Model Loading 
model_path = "/home/ukp/srivastava/projects/generation-train/output/msmarco/t5-small-1-epoch/checkpoint-66500"
generator = QGen(model=QGenModel(model_path))

#### Query-Generation ####
#### Prefix is required to seperate out synthetic queries and qrels from original
prefix = "gen"

#### Generating 3 questions per passage. 
#### Reminder the higher value might produce lots of duplicates
ques_per_passage = 3

#### Generate queries per passage from docs in corpus and save them in data_path
generator.generate(corpus, data_path, ques_per_passage, prefix=prefix)

#### Training on Generated Queries ####
corpus, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=prefix).load(split="train")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

#### Provide any sentence-transformers model path
model_name = "distilroberta-base"
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-genQ-nfcorpus".format(model_name))
os.makedirs(model_save_path, exist_ok=True)

retriever = TrainRetriever(model_name=model_name, model_save_path=model_save_path)
train_samples = retriever.load_train(corpus, gen_queries, gen_qrels)
ir_evaluator = retriever.load_dev(dev_corpus, dev_queries, dev_qrels)

#### Train the model
results = retriever.train(train_samples, evaluator=ir_evaluator, num_epochs=1, evaluation_steps=50)
