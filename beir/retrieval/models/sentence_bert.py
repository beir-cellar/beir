from sentence_transformers import SentenceTransformer
from torch import Tensor
from typing import List, Dict, Union, Tuple
import numpy as np
import torch
import math
import logging

logger = logging.getLogger(__name__)
class SentenceBERT:
    def __init__(self, model_path: Union[str, Tuple] = None, sep: str = " ", **kwargs):
        self.sep = sep
        
        if isinstance(model_path, str):
            self.q_model = SentenceTransformer(model_path)
            self.doc_model = self.q_model
        
        elif isinstance(model_path, tuple):
            self.q_model = SentenceTransformer(model_path[0])
            self.doc_model = SentenceTransformer(model_path[1])
    
    def start_multi_process_pool(self, target_devices: List[str] = None) -> Dict[str, object]:
        return self.doc_model.start_multi_process_pool(target_devices=target_devices)

    def stop_multi_process_pool(self, pool: Dict[str, object]):
        return self.doc_model.stop_multi_process_pool(pool)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        sentences = [(corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip() for i in range(len(corpus['text']))]
        return self.doc_model.encode(sentences, batch_size=batch_size, **kwargs)

    ## Encoding corpus in parallel
    def encode_corpus_parallel(self, corpus: List[Dict[str, str]], pool: Dict[str, str], batch_size: int = 8, chunk_size: int= None, **kwargs) -> Union[List[Tensor], np.ndarray, Tensor]:
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        return self.encode_multi_process(sentences=sentences, pool=pool, batch_size=batch_size, chunk_size=chunk_size)

    @staticmethod
    def encode_multi_process(sentences: List[str], pool: Dict[str, object], batch_size: int = 32, chunk_size: int = None):
        """
        (taken from UKPLab/sentence-transformers/sentence_transformers/SentenceTransformer.py)
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences

        Note: Updated the output of the processes to only return the similarity scores of the top k sentences.

        :param sentences: List of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param batch_size: Encode sentences with batch size
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        :return: Tuple of tensors with top k sentences indexes and their similarity scores
        """
        if chunk_size is None:
            chunk_size = min(math.ceil(len(sentences) / len(pool["processes"]) / 10), 5000)

        logger.info(f"Chunk data into {math.ceil(len(sentences)/chunk_size)} packages of size {chunk_size}")

        input_queue = pool['input']
        last_chunk_id = 0
        chunk = []

        for sentence in sentences:
            chunk.append(sentence)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, batch_size, chunk])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, batch_size, chunk])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in range(last_chunk_id)], key=lambda x: x[0])
        cos_scores_top_k_values = torch.cat([result[1].cpu() for result in results_list], dim=1) # (num_queries, (top_k + 1) * num_sentences / chunk_size) = (num_queries, top_k * num_batches)
        cos_scores_top_k_idx = torch.cat([result[2].cpu() for result in results_list], dim=1)
        return cos_scores_top_k_values, cos_scores_top_k_idx