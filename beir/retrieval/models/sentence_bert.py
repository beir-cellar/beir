from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceBERT:
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
    
    def encode_queries(self, queries, **kwargs):
        return np.asarray(self.model.encode(queries, **kwargs))
    
    def encode_corpus(self, corpus, **kwargs):
        sentences = [(doc["title"] + "  " + doc["text"]).strip() for doc in corpus]   
        return np.asarray(self.model.encode(sentences, **kwargs))