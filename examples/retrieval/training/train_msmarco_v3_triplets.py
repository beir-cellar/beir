'''
This example shows how to train a SOTA Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The model is trained using hard negatives which were specially mined with different dense and lexical search methods for MSMARCO. 

This example has been taken from here with few modifications to train SBERT (MSMARCO-v3) models: 
(https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder-v3.py)

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

Running this script:
python train_msmarco_v3_triplets.py
'''

from sentence_transformers import SentenceTransformer, models, losses
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, json, os, gzip
import tqdm
import logging
import random

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

### Load BEIR MSMARCO training dataset, this will be used for query and corpus for reference.
corpus, queries, _ = GenericDataLoader(data_path).load(split="train")

#### Please Note not all datasets contain a dev split, comment out the line if such the case
# dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

#################################
#### Parameters for Training ####
#################################

train_batch_size = 75           # Increasing the train batch size improves the model performance, but requires more GPU memory (O(n^2))
max_seq_length = 350            # Max length for passages. Increasing it, requires more GPU memory (O(n^4))
ce_score_margin = 3             # Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = 5         # We used different systems to mine hard negatives. Number of hard negatives to add from each system

##################################################
#### Download MSMARCO Hard Negs Triplets File ####
##################################################

triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
msmarco_triplets_filepath = os.path.join(data_path, "msmarco-hard-negatives.jsonl.gz")
util.download_url(triplets_url, msmarco_triplets_filepath)

#### Load the hard negative MSMARCO jsonl triplets from SBERT 
#### These contain a ce-score which denotes the cross-encoder score for the query and passage.
#### We chose a margin between positive and negative passage scores => above which consider negative as hard negative. 
#### Finally to limit the number of negatives per passage, we define num_negs_per_system across all different systems.

train_queries = {}
with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
    for line in tqdm.tqdm(fIn, total=502939):
        data = json.loads(line)
        
        #Get the positive passage ids
        pos_pids = [item['pid'] for item in data['pos']]
        pos_min_ce_score = min([item['ce-score'] for item in data['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin
        
        #Get the hard negatives
        neg_pids = set()
        for system_negs in data['neg'].values():
            negs_added = 0
            for item in system_negs:
                if item['ce-score'] > ce_score_threshold:
                    continue

                pid = item['pid']
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break
        
        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[data['qid']] = {'pos': pos_pids, 'hard_neg': list(neg_pids)}

logging.info("Train queries: {}".format(len(train_queries)))

#### Finally we train on each possible combination of positive passage and hard negative!
#### Add all of these together and form your triplets for training MSMARCO.

triplets = []
for query_id in train_queries:
    query_text = queries[query_id]
    hard_negs = train_queries[query_id]['hard_neg']
    random.shuffle(hard_negs)
    for pos_id in train_queries[query_id]["pos"]:
        for neg_id in hard_negs:
            pos_text = corpus.get(pos_id).get("text")
            neg_text = corpus.get(neg_id).get("text")
            triplets.append([query_text, pos_text, neg_text])

logging.info("All Triplets: {}".format(len(triplets)))

#### For training the SentenceTransformer model 
#### we need a dataset, a dataloader, and a loss used for training.
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
# ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

#### If no dev set is present from above use dummy evaluator
ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v3-{}".format(model_name, dataset))
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