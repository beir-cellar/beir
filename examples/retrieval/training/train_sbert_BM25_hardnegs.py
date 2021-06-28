'''
This examples show how to train a Bi-Encoder for any BEIR dataset.

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that where retrieved by lexical search. We use Elasticsearch
to get (max=10) hard negative examples given a positive passage. 

Running this script:
python train_sbert_BM25_hardnegs.py
'''

from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.train import TrainRetriever
import pathlib, os, tqdm
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download nfcorpus.zip dataset and unzip the dataset
dataset = "scifact"

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

# #### Please Note not all datasets contain a dev split, comment out the line if such the case
# dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

#### Lexical Retrieval using Bm25 (Elasticsearch) ####
#### Provide a hostname (localhost) to connect to ES instance
#### Define a new index name or use an already existing one.
#### We use default ES settings for retrieval
#### https://www.elastic.co/

hostname = "your-hostname" #localhost
index_name = "your-index-name" # scifact

#### Intialize #### 
# (1) True - Delete existing index and re-index all documents from scratch 
# (2) False - Load existing index
initialize = True # False

#### Sharding ####
# (1) For datasets with small corpus (datasets ~ < 5k docs) => limit shards = 1 
# SciFact is a relatively small dataset! (limit shards to 1)
number_of_shards = 1
model = BM25(index_name=index_name, hostname=hostname, initialize=initialize, number_of_shards=number_of_shards)

# (2) For datasets with big corpus ==> keep default configuration
# model = BM25(index_name=index_name, hostname=hostname, initialize=initialize)
bm25 = EvaluateRetrieval(model)

#### Index passages into the index (seperately)
bm25.retriever.index(corpus)

triplets = []
qids = list(qrels) 
hard_negatives_max = 10

#### Retrieve BM25 hard negatives => Given a positive document, find most similar lexical documents
for idx in tqdm.tqdm(range(len(qids)), desc="Retrieve Hard Negatives using BM25"):
    query_id, query_text = qids[idx], queries[qids[idx]]
    pos_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
    pos_doc_texts = [corpus[doc_id]["title"] + " " + corpus[doc_id]["text"] for doc_id in pos_docs]
    hits = bm25.retriever.es.lexical_multisearch(texts=pos_doc_texts, top_hits=hard_negatives_max+1)
    for (pos_text, hit) in zip(pos_doc_texts, hits):
        for (neg_id, _) in hit.get("hits"):
            if neg_id not in pos_docs:
                neg_text = corpus[neg_id]["title"] + " " + corpus[neg_id]["text"]
                triplets.append([query_text, pos_text, neg_text])

#### Provide any sentence-transformers or HF model
model_name = "distilbert-base-uncased" 
word_embedding_model = models.Transformer(model_name, max_seq_length=300)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

#### Provide a high batch-size to train better with triplets!
retriever = TrainRetriever(model=model, batch_size=32)

#### Prepare triplets samples
train_samples = retriever.load_train_triplets(triplets=triplets)
train_dataloader = retriever.prepare_train_triplets(train_samples)

#### Training SBERT with cosine-product
train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

#### training SBERT with dot-product
# train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

#### Prepare dev evaluator
# ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)

#### If no dev set is present from above use dummy evaluator
ir_evaluator = retriever.load_dummy_evaluator()

#### Provide model save path
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", "{}-v2-{}-bm25-hard-negs".format(model_name, dataset))
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