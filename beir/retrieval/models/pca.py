from sentence_transformers import SentenceTransformer
from torch import Tensor
from typing import List, Dict, Union, Tuple
import numpy as np
import faiss
import logging

logger = logging.getLogger(__name__)

class PCA:
    def __init__(self, model, output_dim, **kwargs):
        self.model = model
        self.output_dim = output_dim
        self.pca_matrix = None
        self.input_dim = None
    
    def train(self, corpus: Dict[str, Dict[str, str]], batch_size: int = 128, corpus_chunk_size: int = 50000):
        
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        corpus_ids = list(corpus.keys())
        corpus = [corpus[cid] for cid in corpus_ids]

        itr = range(0, len(corpus), corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus))
            
            #Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=batch_size,
                show_progress_bar=True, 
                )
            if not batch_num:
                corpus_embeddings = sub_corpus_embeddings
            else:
                corpus_embeddings = np.vstack([corpus_embeddings, sub_corpus_embeddings])
        
        self.input_dim = corpus_embeddings.shape[1]
        self.pca_matrix = faiss.PCAMatrix(self.input_dim, self.output_dim)
        logger.info("Creating PCA Matrix...")
        logger.info("Input Dimension: {}, Output Dimension: {}".format(self.input_dim, self.output_dim))
        logger.info("Starting to train the PCA Matrix...")
        self.pca_matrix.train(corpus_embeddings)
    
    def save(self, output_path):
        logger.info("Saving PCA Matrix to path: {}".format(output_path))
        faiss.write_VectorTransform(self.pca_matrix, output_path)
    
    def load(self, input_path):
        logger.info("Loading PCA Matrix from path: {}".format(input_path))
        self.pca_matrix = faiss.read_VectorTransform(input_path)
        assert self.pca_matrix.is_trained
    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        query_embeddings = self.model.encode_queries(queries=queries, batch_size=batch_size)
        return self.pca_matrix.apply_py(query_embeddings)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        corpus_embeddings = self.model.encode_corpus(corpus=corpus, batch_size=batch_size)
        return self.pca_matrix.apply_py(corpus_embeddings)