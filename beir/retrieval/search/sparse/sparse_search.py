from tqdm.autonotebook import trange
from typing import List, Dict, Union, Tuple
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

class SparseSearch:
    
    def __init__(self, model, batch_size: int = 16, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.sparse_matrix = None
        self.results = {}
    
    def search(self, 
        corpus: Dict[str, Dict[str, str]], 
        queries: Dict[str, str], 
        top_k: int, *args, **kwargs
    ) -> Dict[str, Dict[str, float]]:
        
        doc_ids = list(corpus.keys())
        query_ids = list(queries.keys())
        documents = [corpus[doc_id] for doc_id in doc_ids]
        logging.info("Computing document embeddings and creating sparse matrix")
        self.sparse_matrix_doc = self.model.encode_corpus(documents, batch_size=self.batch_size)  # [n_doc, n_voc]
        logging.info("Starting to Retrieve...")
        for start_idx in trange(0, len(queries), self.batch_size, desc='query'):
            local_query_ids = query_ids[start_idx:start_idx+self.batch_size]
            local_queries = [queries[qid] for qid in local_query_ids]
            qry_matrix = self.model.encode_query(local_queries)
            scores = self.sparse_matrix_doc.dot(qry_matrix.transpose()).todense() # [n_doc, vocab]x[vocab, n_qry] -> [n_doc, n_qry]
            scores = torch.from_numpy(scores) # [n_qry, n_doc]
            top_k_values, top_k_indices = torch.topk(scores, top_k, dim=0, sorted=False)
            top_k_values = top_k_values.transpose(0, 1).tolist() # [n_qry, top_k]
            top_k_indices = top_k_indices.transpose(0, 1).tolist() # [n_qry, top_k]
            for i, qid in enumerate(local_query_ids):
                k_ind = top_k_indices[i]
                k_val = top_k_values[i]
                self.results[qid] = {doc_ids[pid]: score for pid, score in zip(k_ind, k_val) if doc_ids[pid] != qid}
        return self.results

