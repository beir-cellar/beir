from sentence_transformers import SentenceTransformer    
import torch
from torch import Tensor
from typing import List, Dict, Union, Tuple
import numpy as np
import importlib.util

if importlib.util.find_spec("tldr") is not None:
    from tldr import TLDR as NaverTLDR

class TLDR:
    def __init__(self, encoder_model: SentenceTransformer, model_path: Union[str, Tuple] = None, sep: str = " ", n_components: int = 128, n_neighbors: int = 5,
                encoder: str = "linear", projector: str = "mlp-2-2048", verbose: int = 2, knn_approximation: str = None, output_folder: str = "data/", **kwargs):
        self.encoder_model = encoder_model
        self.sep = sep
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_folder = output_folder
        
        if model_path: self.load(model_path)
        
        else: 
            self.model = NaverTLDR(
                n_components=n_components,
                n_neighbors=n_neighbors,
                encoder=encoder,
                projector=projector,
                device=self.device,
                verbose=verbose,
                knn_approximation=knn_approximation,
            )
    
    def fit(self, corpus: List[Dict[str, str]], batch_size: int = 8, epochs: int = 100, warmup_epochs: int = 10, 
            train_batch_size: int = 1024, print_every: int = 100, **kwargs):
        
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        self.model.fit(self.encoder_model.encode(sentences, batch_size=batch_size, **kwargs), 
                 epochs=epochs, 
                 warmup_epochs=warmup_epochs, 
                 batch_size=batch_size, 
                 output_folder=self.output_folder, 
                 print_every=print_every)
    
    def save(self, model_path: str, knn_path: str = None):
        self.model.save(model_path)
        if knn_path: self.model.save_knn(knn_path)
    
    def load(self, model_path: str):
        self.model = NaverTLDR()
        self.model.load(model_path, init=True)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        return self.model.transform(self.encoder_model.encode(queries, batch_size=batch_size, **kwargs), l2_norm=True)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.model.transform(self.encoder_model.encode(sentences, batch_size=batch_size, **kwargs), l2_norm=True)