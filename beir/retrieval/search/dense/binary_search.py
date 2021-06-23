from .util import cos_sim, dot_score
from .faiss_index import FaissBinaryIndex
import logging
import sys
import torch
import faiss
from typing import Dict, List

logger = logging.getLogger(__name__)

#Parent class for any dense model
class DenseRetrievalBinaryCodeSearch:
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True
        self.faiss_index = None
        self.results = {}
        self.mapping = {}
        self.rev_mapping = {}
    
    def _create_mapping_ids(self, corpus_ids):
        if not all(isinstance(doc_id, int) for doc_id in corpus_ids):
            for idx in range(len(corpus_ids)):
                self.rev_mapping[idx] = corpus_ids[idx]
                self.mapping[corpus_ids[idx]] = idx
    
    def index(self, corpus: Dict[str, Dict[str, str]], hash_num_bits: int = 768):
        
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        corpus_ids = list(corpus.keys())
        self._create_mapping_ids(corpus_ids)
        corpus = [corpus[cid] for cid in corpus_ids]
        
        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))
            sub_corpus_ids = [self.mapping[idx] for idx in corpus_ids[corpus_start_idx:corpus_end_idx]]
            
            #Encode chunk of corpus    
            sub_corpus_binary_codes = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                show_progress_bar=self.show_progress_bar, 
                batch_size=self.batch_size)
            
            #Index chunk of corpus into faiss
            logger.info("Indexing Batch {}/{} into Faiss...".format(batch_num+1, len(itr)))
            dim_size = sub_corpus_binary_codes.shape[1]
            base_index = faiss.IndexBinaryHash(dim_size * 8, hash_num_bits)
            self.faiss_index = FaissBinaryIndex.build(sub_corpus_ids, sub_corpus_binary_codes, base_index)
        
        del sub_corpus_ids, sub_corpus_binary_codes
        
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str], 
               top_k: int,
               score_function = None,
               rerank: bool = True,
               binary_k: int = 1000, 
               index: bool = True, **kwargs) -> Dict[str, Dict[str, float]]:
        
        ## Used for Indexing
        if index: self.index(corpus)

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size)

        scores, doc_ids = self.faiss_index.search(query_embeddings, top_k, binary_k=binary_k, rerank=rerank)
        
        for idx in range(len(query_ids)):
            _scores = [float(score) for score in scores[idx]]
            if len(self.rev_mapping) != 0:
                _doc_ids = [self.rev_mapping[doc_id] for doc_id in doc_ids[idx]]
            else:
                _doc_ids = [str(doc_id) for doc_id in doc_ids[idx]]
            self.results[query_ids[idx]] = dict(zip(_doc_ids, _scores))
        
        return self.results