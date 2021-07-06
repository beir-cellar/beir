from sentence_transformers import SentenceTransformer
from torch import Tensor
from typing import List, Dict, Union, Tuple
import numpy as np

class SentenceBERT:
    def __init__(self, model_path: Union[str, Tuple] = None, sep: str = " ", **kwargs):
        self.sep = sep
        
        if isinstance(model_path, str):
            self.q_model = SentenceTransformer(model_path)
            self.doc_model = self.q_model
        
        elif isinstance(model_path, tuple):
            self.q_model = SentenceTransformer(model_path[0])
            self.doc_model = SentenceTransformer(model_path[1])
    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)
