from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
from typing import List, Dict

class QFilterModel:
    def __init__(self, model_path: str = None, max_length: int = 512, **kwargs):
        self.model = CrossEncoder(model_path, max_length=max_length)
    
    def predict(self, sentences: List[List[str,str]], batch_size: int = 32, show_progress_bar: bool = False) -> List[float]:
        return self.model.predict(
            sentences=sentences, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar)