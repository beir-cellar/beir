from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
from beir.filtration import QueryFilter as QFilter
from beir.filtration.models import QFilterModel
from beir.retrieval.train import TrainRetriever
from sentence_transformers import losses

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


##############################
#### 1. Query-Generation  ####
##############################


#### Model Loading 
model_path = "BeIR/query-gen-msmarco-t5-base"
generator = QGen(model=QGenModel(model_path))

#### Prefix is required to seperate out synthetic queries and qrels from original
prefix = "gen"

#### Generating 3 questions per passage. 
#### Reminder the higher value might produce lots of duplicates
ques_per_passage = 3

#### Generate queries per passage from docs in corpus and save them in data_path
generator.generate(corpus, output_dir=data_path, ques_per_passage=ques_per_passage, prefix=prefix)


##############################
#### 2. Query-Filtration  ####
##############################


#### Load the Generated datasets, by providing the unique prefix  
corpus, gen_queries, gen_qrels = GenericDataLoader(data_path, prefix=prefix).load(split="train")

#### Mention any Cross-Encoder model in Sentence-Transformers package ####
cross_encoder_name = "cross-encoder/ms-marco-TinyBERT-L-4"
query_filter = QFilter(model=QFilterModel(cross_encoder_name))

#### Filter queries whose cosine similarity b/w query and passage is greather than threshold = 0.5
#### cosine similarity scores vary from [0,1] where 0 is dissimilar and 1 is identical.
filter_prefix = "filter"
query_filter.filter(corpus, gen_queries, gen_qrels, output_dir=data_path, threshold=0.5, prefix=filter_prefix)


################################
#### 3. Train Dense-Encoder ####
################################


#### Load the Filter datasets, by providing the unique filter prefix  
corpus, filter_queries, filter_qrels = GenericDataLoader(data_path, prefix=filter_prefix).load(split="train")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

#### Provide any sentence-transformers model path
model_name = "distilroberta-base"
retriever = TrainRetriever(model_name=model_name, batch_size=64)

#### Prepare training samples
train_samples = retriever.load_train(corpus, filter_queries, filter_qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

#### Prepare dev evaluator
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
#### If no dev set is present
# ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-GenQ-filter-nfcorpus".format(model_name))
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = 1
evaluation_steps = 5000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)
