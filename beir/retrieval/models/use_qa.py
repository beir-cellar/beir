import numpy as np
import importlib.util
from typing import List, Dict
from tqdm.autonotebook import trange

if importlib.util.find_spec("tensorflow") is not None \
    and importlib.util.find_spec("tensorflow_hub") is not None \
    and importlib.util.find_spec("tensorflow_text") is not None:
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text

class UseQA:
    def __init__(self, hub_url=None, **kwargs):
        self.initialisation()
        self.model = hub.load(hub_url)
        
    @staticmethod
    def initialisation():
        # limiting tensorflow gpu-memory if used
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        output = []
        for start_idx in trange(0, len(queries), batch_size, desc='que'):
            embeddings_q = self.model.signatures['question_encoder'](
                tf.constant(queries[start_idx:start_idx+batch_size]))
            for emb in embeddings_q["outputs"]:
                output.append(emb)

        return np.asarray(output)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> np.ndarray:
        output = []
        for start_idx in trange(0, len(corpus), batch_size, desc='pas'):
            titles = [row.get('title', '') for row in corpus[start_idx:start_idx+batch_size]]
            texts = [row.get('text', '') if row.get('text', '') != None else "" for row in corpus[start_idx:start_idx+batch_size]]
            
            if all(title == "" for title in titles): # Check is title is not present in the dataset
                titles = texts # title becomes the context as well

            embeddings_c = self.model.signatures['response_encoder'](
                input=tf.constant(titles),
                context=tf.constant(texts))
            for emb in embeddings_c["outputs"]:
                output.append(emb)
                    
        return np.asarray(output)
