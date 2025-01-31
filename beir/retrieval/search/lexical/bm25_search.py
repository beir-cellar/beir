from .. import BaseSearch
from .elastic_search import ElasticSearch
import tqdm
import time
from typing import List, Dict

def sleep(seconds):
    if seconds: time.sleep(seconds) 

class BM25Search(BaseSearch):
    def __init__(self, index_name: str, hostname: str = "localhost", keys: Dict[str, str] = {"title": "title", "body": "txt"}, language: str = "english",
                 batch_size: int = 128, timeout: int = 100, retry_on_timeout: bool = True, maxsize: int = 24, number_of_shards: int = "default", 
                 initialize: bool = True, sleep_for: int = 2):
        self.results = {}
        self.batch_size = batch_size
        self.initialize = initialize
        self.sleep_for = sleep_for
        self.config = {
            "hostname": hostname, 
            "index_name": index_name,
            "keys": keys,
            "timeout": timeout,
            "retry_on_timeout": retry_on_timeout,
            "maxsize": maxsize,
            "number_of_shards": number_of_shards,
            "language": language
        }
        self.es = ElasticSearch(self.config)
        if self.initialize:
            self.initialise()
    
    def initialise(self):
        self.es.delete_index()
        sleep(self.sleep_for)
        self.es.create_index()
    
    def search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        
        # Index the corpus within elastic-search
        # False, if the corpus has been already indexed
        if self.initialize:
            self.index(corpus)
            # Sleep for few seconds so that elastic-search indexes the docs properly
            sleep(self.sleep_for)
        
        #retrieve results from BM25 
        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in query_ids]
        
        for start_idx in tqdm.trange(0, len(queries), self.batch_size, desc='que'):
            query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
            results = self.es.lexical_multisearch(
                texts=queries[start_idx:start_idx+self.batch_size], 
                top_hits=top_k + 1) # Add 1 extra if query is present with documents
            
            for (query_id, hit) in zip(query_ids_batch, results):
                scores = {}
                for corpus_id, score in hit['hits']:
                    if corpus_id != query_id: # query doesnt return in results
                        scores[corpus_id] = score
                    self.results[query_id] = scores
        
        return self.results
        
    
    def index(self, corpus: Dict[str, Dict[str, str]]):
        progress = tqdm.tqdm(unit="docs", total=len(corpus))
        # dictionary structure = {_id: {title_key: title, text_key: text}}
        dictionary = {idx: {
            self.config["keys"]["title"]: corpus[idx].get("title", None), 
            self.config["keys"]["body"]: corpus[idx].get("text", None)
            } for idx in list(corpus.keys())
        }
        self.es.bulk_add_to_index(
                generate_actions=self.es.generate_actions(
                dictionary=dictionary, update=False),
                progress=progress
                )
