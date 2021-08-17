'''
This example shows how to train a SOTA Bi-Encoder with Margin-MSE loss for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).

In this example we use a knowledge distillation setup. Sebastian Hofstätter et al. trained in https://arxiv.org/abs/2010.02666 an
an ensemble of large Transformer models for the MS MARCO datasets and combines the scores from a BERT-base, BERT-large, and ALBERT-large model.

We use the MSMARCO Hard Negatives File (Provided by Nils Reimers): https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz
Negative passage are hard negative examples, that were mined using different dense embedding, cross-encoder methods and lexical search methods.
Contains upto 50 negatives for each of the four retrieval systems: [bm25, msmarco-distilbert-base-tas-b, msmarco-MiniLM-L-6-v3, msmarco-distilbert-base-v3]
Each positive and negative passage comes with a score from a Cross-Encoder (msmarco-MiniLM-L-6-v3). This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

This example has been taken from here with few modifications to train SBERT (MSMARCO-v3) models: 
(https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder-v3.py)

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using dot-product to find matching passages for a given query.

For training, we use Margin MSE Loss. There, we pass triplets in the format:
triplets: (query, positive_passage, negative_passage)
label: positive_ce_score - negative_ce_score => (ce-score b/w query and positive or negative_passage)

PS: Using Margin MSE Loss doesn't require a threshold, or to set maximum negatives per system (required for Multiple Ranking Negative Loss)!
This is often a cumbersome process to find the optimal threshold which is dependent for Multiple Negative Ranking Loss.

Running this script:
python train_msmarco_v3_margin_MSE.py
'''

from sentence_transformers import SentenceTransformer, models, InputExample
from beir import util, LoggingHandler, losses
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
import pathlib, os, gzip, json
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

#################################
#### Parameters for Training ####
#################################

train_batch_size = 75           # Increasing the train batch size improves the model performance, but requires more GPU memory (O(n))
max_seq_length = 350            # Max length for passages. Increasing it, requires more GPU memory (O(n^2))

##################################################
#### Download MSMARCO Hard Negs Triplets File ####
##################################################

triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
msmarco_triplets_filepath = os.path.join(data_path, "msmarco-hard-negatives.jsonl.gz")
if not os.path.isfile(msmarco_triplets_filepath):
    util.download_url(triplets_url, msmarco_triplets_filepath)

#### Load the hard negative MSMARCO jsonl triplets from SBERT 
#### These contain a ce-score which denotes the cross-encoder score for the query and passage.

logging.info("Loading MSMARCO hard-negatives...")

train_queries = {}
with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
    for line in tqdm(fIn, total=502939):
        data = json.loads(line)
        
        #Get the positive passage ids
        pos_pids = [item['pid'] for item in data['pos']]
        pos_scores = dict(zip(pos_pids, [item['ce-score'] for item in data['pos']]))
        
        #Get all the hard negatives
        neg_pids = set()
        neg_scores = {}
        for system_negs in data['neg'].values():
            for item in system_negs:
                pid = item['pid']
                score = item['ce-score']
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    neg_scores[pid] = score
        
        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[data['qid']] = {'query': queries[data['qid']], 'pos': pos_pids, 'pos_scores': pos_scores, 
                                          'hard_neg': neg_pids, 'hard_neg_scores': neg_scores}
        
logging.info("Train queries: {}".format(len(train_queries)))

# We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.

class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['hard_neg'] = list(self.queries[qid]['hard_neg'])
            random.shuffle(self.queries[qid]['hard_neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]["text"]
        query['pos'].append(pos_id)
        pos_score = float(query['pos_scores'][pos_id])

        neg_id = query['hard_neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]["text"]
        query['hard_neg'].append(neg_id)
        neg_score = float(query['hard_neg_scores'][neg_id])

        return InputExample(texts=[query_text, pos_text, neg_text], label=(pos_score - neg_score))

    def __len__(self):
        return len(self.queries)

# We construct the SentenceTransformer bi-encoder from scratch with mean-pooling
model_name = "distilbert-base-uncased" 
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Provide a high batch-size to train better with triplets!
retriever = TrainRetriever(model=model, batch_size=train_batch_size)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = retriever.prepare_train(train_dataset, shuffle=True, dataset_present=True)

#### Training SBERT with dot-product (default) using Margin MSE Loss
train_loss = losses.MarginMSELoss(model=retriever.model)

#### If no dev set is present from above use dummy evaluator
ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v3-margin-MSE-loss-{}".format(model_name, dataset))
os.makedirs(model_save_path, exist_ok=True)

#### Configure Train params
num_epochs = 11
evaluation_steps = 10000
warmup_steps = 1000

retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True)