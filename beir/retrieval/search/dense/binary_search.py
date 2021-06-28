from .util import cos_sim, dot_score
from .faiss_index import FaissBinaryIndex
import logging
import sys
import torch
import faiss
import numpy as np
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
                self.mapping[corpus_ids[idx]] = idx
                self.rev_mapping[idx] = corpus_ids[idx]
    
    def index(self, corpus: Dict[str, Dict[str, str]], hash_num_bits: int = 768, output_dir: str = None):
        
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        corpus_ids = list(corpus.keys())
        self._create_mapping_ids(corpus_ids)

        corpus = [corpus[cid] for cid in corpus_ids]

        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))
            
            #Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                show_progress_bar=self.show_progress_bar, 
                batch_size=self.batch_size)
            
            if not batch_num: 
                corpus_embeddings = sub_corpus_embeddings
            else:
                corpus_embeddings = np.vstack([corpus_embeddings, sub_corpus_embeddings])
        
        #Index chunk of corpus into faiss
        logger.info("Indexing Passages into Faiss...") 
        
        base_index = faiss.IndexBinaryHash(corpus_embeddings.shape[1] * 8, hash_num_bits)
        faiss_ids = [self.mapping.get(corpus_id) for corpus_id in corpus_ids]
        self.faiss_index = FaissBinaryIndex.build(faiss_ids, corpus_embeddings, base_index)
        
        logger.info("Faiss indexing completed! {} Documents Indexed...".format(len(self.faiss_index._passage_ids)))
        del sub_corpus_embeddings, corpus_embeddings
        
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str], 
               top_k: int,
               score_function = None,
               rerank: bool = True,
               binary_k: int = 1000, 
               index: bool = True, **kwargs) -> Dict[str, Dict[str, float]]:
        
        ## Used for Indexing
        if index: self.index(corpus, **kwargs)

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, show_progress_bar=self.show_progress_bar, batch_size=self.batch_size)

        faiss_scores, faiss_doc_ids = self.faiss_index.search(query_embeddings, top_k, binary_k=binary_k, rerank=rerank)
        
        for idx in range(len(query_ids)):
            scores = [float(score) for score in faiss_scores[idx]]
            if len(self.rev_mapping) != 0:
                doc_ids = [self.rev_mapping[doc_id] for doc_id in faiss_doc_ids[idx]]
            else:
                doc_ids = [str(doc_id) for doc_id in faiss_doc_ids[idx]]
            self.results[query_ids[idx]] = dict(zip(doc_ids, scores))
        
        return self.results