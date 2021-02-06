from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

class SentenceBERT:
    def __init__(self, model_path=None, **kwargs):
        self.model = SentenceTransformer(model_path)
    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        return np.asarray(self.model.encode(queries, batch_size=batch_size, **kwargs))
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + "  " + doc["text"]).strip() for doc in corpus]
        return np.asarray(self.model.encode(sentences, batch_size=batch_size, **kwargs))