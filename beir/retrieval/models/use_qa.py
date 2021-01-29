import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
from tqdm.autonotebook import trange


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
        
    def encode_queries(self, queries, batch_size=16, **kwargs):
        output = []
        for start_idx in trange(0, len(queries), batch_size, desc='que'):
            embeddings_q = self.model.signatures['question_encoder'](
                tf.constant(queries[start_idx:start_idx+batch_size]))
            for emb in embeddings_q["outputs"]:
                output.append(emb)

        return np.asarray(output)
    
    def encode_corpus(self, corpus, batch_size=8, **kwargs):
        output = []
        for start_idx in trange(0, len(corpus), batch_size, desc='pas'):
            titles = [row['title'] for row in corpus[start_idx:start_idx+batch_size]]
            texts = [row['text']  for row in corpus[start_idx:start_idx+batch_size]]
            embeddings_c = self.model.signatures['response_encoder'](
                input=tf.constant(titles),
                context=tf.constant(texts))
            for emb in embeddings_c["outputs"]:
                output.append(emb)
                    
        return np.asarray(output)