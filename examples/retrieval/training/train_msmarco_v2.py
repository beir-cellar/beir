'''
"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The model is trained with BM25 (only lexical) sampled hard negatives provided by the SentenceTransformers Repo. 

This example has been taken from here with few modifications to train SBERT (MSMARCO-v2) models: 
(https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder-v2.py) 

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that where retrieved by lexical search. We use the negative
passages (the triplets) that are provided by the MS MARCO dataset.

Running this script:
python train_msmarco_v2.py
'''

from sentence_transformers import SentenceTransformer, models, losses
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os, gzip
import logging
import json

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download msmarco.zip dataset and unzip the dataset
dataset = "msmarco"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Please Note not all datasets contain a dev split, comment out the line if such the case
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

########################################
#### Download MSMARCO Triplets File ####
########################################

train_batch_size = 75           # Increasing the train batch size improves the model performance, but requires more GPU memory (O(n^2))
max_seq_length = 350            # Max length for passages. Increasing it, requires more GPU memory (O(n^4))

# The triplets file contains 5,028,051 sentence pairs (ref: https://sbert.net/datasets/paraphrases)
triplets_url = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/paraphrases/msmarco-query_passage_negative.jsonl.gz"
msmarco_triplets_filepath = os.path.join(data_path, "msmarco-triplets.jsonl.gz")

if not os.path.isfile(msmarco_triplets_filepath):
    util.download_url(triplets_url, msmarco_triplets_filepath)

#### The triplets file contains tab seperated triplets in each line =>
# 1. train query (text), 2. positive doc (text), 3. hard negative doc (text) 
triplets = []
with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        triplet = json.loads(line)
        triplets.append(triplet)
        
#### Provide any sentence-transformers or HF model
model_name = "distilbert-base-uncased" 
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Provide a high batch-size to train better with triplets!
retriever = TrainRetriever(model=model, batch_size=train_batch_size)

#### Prepare triplets samples
train_samples = retriever.load_train_triplets(triplets=triplets)
train_dataloader = retriever.prepare_train_triplets(train_samples)

#### Training SBERT with cosine-product
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
# #### training SBERT with dot-product
# # train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

#### Prepare dev evaluator
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

#### If no dev set is present from above use dummy evaluator
# ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v2-{}".format(model_name, dataset))
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = 1
evaluation_steps = 10000
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)
