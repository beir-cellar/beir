from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
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
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

#### Provide any sentence-transformers model path
model_name = "distilroberta-base"
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-nfcorpus".format(model_name))
os.makedirs(model_save_path, exist_ok=True)

retriever = TrainRetriever(model_name=model_name, model_save_path=model_save_path)
train_samples = retriever.load_train(corpus, queries, qrels)

#### Train the model
results = retriever.train(train_samples, num_epochs=1)
