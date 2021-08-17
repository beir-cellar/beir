from sentence_transformers import SentenceTransformer
from torch import Tensor
from typing import List, Dict, Union, Tuple
import numpy as np

class BinarySentenceBERT:
    def __init__(self, model_path: Union[str, Tuple] = None, sep: str = " ", threshold: Union[float, Tensor] = 0, **kwargs):
        self.sep = sep
        self.threshold = threshold
        
        if isinstance(model_path, str):
            self.q_model = SentenceTransformer(model_path)
            self.doc_model = self.q_model
        
        elif isinstance(model_path, tuple):
            self.q_model = SentenceTransformer(model_path[0])
            self.doc_model = SentenceTransformer(model_path[1])
    
    def _convert_embedding_to_binary_code(self, embeddings: List[Tensor]) -> List[Tensor]:
        return embeddings.new_ones(embeddings.size()).masked_fill_(embeddings < self.threshold, -1.0)
    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() for doc in corpus]
        embs = self.doc_model.encode(sentences, batch_size=batch_size, convert_to_tensor=True, **kwargs)
        embs = self._convert_embedding_to_binary_code(embs).cpu().numpy()
        embs = np.where(embs == -1, 0, embs).astype(np.bool)
        embs = np.packbits(embs).reshape(embs.shape[0], -1)
        return np.vstack(embs)