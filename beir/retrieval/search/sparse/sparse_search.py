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
        top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
        
        doc_ids = list(corpus.keys())
        query_ids = list(queries.keys())
        documents = [corpus[doc_id] for doc_id in doc_ids]
        logging.info("Computing document embeddings and creating sparse matrix")
        self.sparse_matrix = self.model.encode_corpus(documents, batch_size=self.batch_size)  # [n_doc, n_voc]
        logging.info("Starting to Retrieve...")
        for start_idx in trange(0, len(queries), desc='query'):
            qid = query_ids[start_idx]
            query_vector = self.model.encode_query(queries[qid]) # [1, n_voc]
            scores = self.sparse_matrix.dot(query_vector.transpose()).todense()
            scores = torch.from_numpy(scores).squeeze()
            top_k_values, top_k_indices = torch.topk(scores, top_k, sorted=False)
            top_k_values = top_k_values.squeeze().tolist()
            top_k_indices = top_k_indices.squeeze().tolist()
            self.results[qid] = {doc_ids[pid]: score for pid, score in zip(top_k_indices, top_k_values) if doc_ids[pid] != qid}
        
        return self.results

