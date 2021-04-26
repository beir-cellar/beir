from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.reranking.models import CrossEncoder
from operator import itemgetter

import pathlib, os
import datetime
import logging
import random

random.seed(42)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

#### /print debug information to stdout

#### Download dbpedia-entity.zip dataset and unzip the dataset
dataset = "dbpedia-entity"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Loading test queries and corpus in DBPedia
corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
corpus_ids, query_ids = list(corpus), list(queries)
corpus_texts = {corpus_id: corpus[corpus_id]["title"] + " " + corpus[corpus_id]["text"] for corpus_id in corpus}

#### Randomly sample 1M pairs from Original Corpus (4.63M pairs) 
#### First include all relevant documents (i.e. present in qrels)
corpus_set = set()
for query_id in qrels:
    corpus_set.update(list(qrels[query_id].keys()))
corpus_new = {corpus_id: corpus[corpus_id] for corpus_id in corpus_set}

#### Remove already seen k relevant documents and sample (1M - k) docs randomly
remaining_corpus = list(set(corpus_ids) - corpus_set)
sample = 1000000 - len(corpus_set)

for corpus_id in random.sample(remaining_corpus, sample):
    corpus_new[corpus_id] = corpus[corpus_id]

#### Provide parameters for Elasticsearch
hostname = "desktop-158.ukp.informatik.tu-darmstadt.de:9200"
index_name = dataset
model = BM25(index_name=index_name, hostname=hostname)
bm25 = EvaluateRetrieval(model)
 
#### Index 1M passages into the index (seperately)
bm25.retriever.index(corpus_new)

#### Reranking using Cross-Encoder model
reranker = CrossEncoder('cross-encoder/ms-marco-electra-base')

#### Saving benchmark times
time_taken_all = {}

for query_id in query_ids:
    query = queries[query_id]
    
    #### Measure time to retrieve top-100 BM25 documents using single query latency
    start = datetime.datetime.now()
    results = bm25.retriever.es.lexical_search(text=query, top_hits=100) 
    
    #### Measure time to rerank top-100 BM25 documents using CE
    sentence_pairs = [[queries[query_id], corpus_texts[hit[0]]] for hit in results["hits"]]
    scores = reranker.predict(sentence_pairs, batch_size=100, show_progress_bar=False)
    hits = {results["hits"][idx][0]: scores[idx] for idx in range(len(scores))}            
    sorted_results = {k: v for k,v in sorted(hits.items(), key=itemgetter(1), reverse=True)} 
    end = datetime.datetime.now()
    
    #### Measuring time taken in ms (milliseconds)
    time_taken = (end - start)
    time_taken = time_taken.total_seconds() * 1000
    time_taken_all[query_id] = time_taken
    logging.info("{}: {} {:.2f}ms".format(query_id, query, time_taken))

time_taken = list(time_taken_all.values())
logging.info("Average time taken: {:.2f}ms".format(sum(time_taken)/len(time_taken_all)))