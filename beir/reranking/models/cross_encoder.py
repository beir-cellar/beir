from sentence_transformers.cross_encoder import CrossEncoder as CE
import numpy as np
from typing import List, Dict, Tuple

class CrossEncoder:
    def __init__(self, model_path: str = None, max_length: int = None, **kwargs):
        if max_length != None:
            self.model = CE(model_path, max_length=max_length, **kwargs)
        self.model = CE(model_path, **kwargs)
    
    def predict(self, sentences: List[Tuple[str,str]], batch_size: int = 32, show_progress_bar: bool = True) -> List[float]:
        return self.model.predict(
            sentences=sentences, 
            batch_size=batch_size, 
            show_progress_bar=show_progress_bar)