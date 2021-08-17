from .util import normalize
from typing import List, Optional, Tuple, Union
from tqdm.autonotebook import trange
import numpy as np

import faiss
import logging
import time

logger = logging.getLogger(__name__)


class FaissIndex:
    def __init__(self, index: faiss.Index, passage_ids: List[int] = None):
        self.index = index
        self._passage_ids = None
        if passage_ids is not None:
            self._passage_ids = np.array(passage_ids, dtype=np.int64)

    def search(self, query_embeddings: np.ndarray, k: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        scores_arr, ids_arr = self.index.search(query_embeddings, k)
        if self._passage_ids is not None:
            ids_arr = self._passage_ids[ids_arr.reshape(-1)].reshape(query_embeddings.shape[0], -1)
        logger.info("Total search time: %.3f", time.time() - start_time)
        return scores_arr, ids_arr
    
    def save(self, fname: str):
        faiss.write_index(self.index, fname)

    @classmethod
    def build(
        cls,
        passage_ids: List[int],
        passage_embeddings: np.ndarray,
        index: Optional[faiss.Index] = None,
        buffer_size: int = 50000,
    ):
        if index is None:
            index = faiss.IndexFlatIP(passage_embeddings.shape[1])
        for start in trange(0, len(passage_ids), buffer_size):
            index.add(passage_embeddings[start : start + buffer_size])

        return cls(index, passage_ids)

    def to_gpu(self):
        if faiss.get_num_gpus() == 1:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        else:
            cloner_options = faiss.GpuMultipleClonerOptions()
            cloner_options.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=cloner_options)

        return self.index


class FaissHNSWIndex(FaissIndex):
    def search(self, query_embeddings: np.ndarray, k: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        query_embeddings = np.hstack((query_embeddings, np.zeros((query_embeddings.shape[0], 1), dtype=np.float32)))
        return super().search(query_embeddings, k)
    
    def save(self, output_path: str):
        super().save(output_path)

    @classmethod
    def build(
        cls,
        passage_ids: List[int],
        passage_embeddings: np.ndarray,
        index: Optional[faiss.Index] = None,
        buffer_size: int = 50000,
    ):
        sq_norms = (passage_embeddings ** 2).sum(1)
        max_sq_norm = float(sq_norms.max())
        aux_dims = np.sqrt(max_sq_norm - sq_norms)
        passage_embeddings = np.hstack((passage_embeddings, aux_dims.reshape(-1, 1)))
        return super().build(passage_ids, passage_embeddings, index, buffer_size)

class FaissTrainIndex(FaissIndex):
    def search(self, query_embeddings: np.ndarray, k: int, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return super().search(query_embeddings, k)
    
    def save(self, output_path: str):
        super().save(output_path)

    @classmethod
    def build(
        cls,
        passage_ids: List[int],
        passage_embeddings: np.ndarray,
        index: Optional[faiss.Index] = None,
        buffer_size: int = 50000,
    ):
        index.train(passage_embeddings)
        return super().build(passage_ids, passage_embeddings, index, buffer_size)

class FaissBinaryIndex(FaissIndex):
    def __init__(self, index: faiss.Index, passage_ids: List[int] = None, passage_embeddings: np.ndarray = None):
        self.index = index
        self._passage_ids = None
        if passage_ids is not None:
            self._passage_ids = np.array(passage_ids, dtype=np.int64)

        self._passage_embeddings = None
        if passage_embeddings is not None:
            self._passage_embeddings = passage_embeddings

    def search(self, query_embeddings: np.ndarray, k: int, binary_k: int = 1000, rerank: bool = True, 
               score_function: str = "dot", threshold: Union[int, np.ndarray] = 0, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()
        num_queries = query_embeddings.shape[0]
        bin_query_embeddings = np.packbits(np.where(query_embeddings > threshold, 1, 0)).reshape(num_queries, -1)

        if not rerank:
            scores_arr, ids_arr = self.index.search(bin_query_embeddings, k)
            if self._passage_ids is not None:
                ids_arr = self._passage_ids[ids_arr.reshape(-1)].reshape(num_queries, -1)
            return scores_arr, ids_arr

        if self._passage_ids is not None:
            _, ids_arr = self.index.search(bin_query_embeddings, binary_k)
            logger.info("Initial search time: %.3f", time.time() - start_time)
            passage_embeddings = np.unpackbits(self._passage_embeddings[ids_arr.reshape(-1)])
            passage_embeddings = passage_embeddings.reshape(num_queries, binary_k, -1).astype(np.float32)
        else:
            raw_index = self.index.index
            _, ids_arr = raw_index.search(bin_query_embeddings, binary_k)
            logger.info("Initial search time: %.3f", time.time() - start_time)
            passage_embeddings = np.vstack(
                [np.unpackbits(raw_index.reconstruct(int(id_))) for id_ in ids_arr.reshape(-1)]
            )
            passage_embeddings = passage_embeddings.reshape(
                query_embeddings.shape[0], binary_k, query_embeddings.shape[1]
            )
            passage_embeddings = passage_embeddings.astype(np.float32)

        passage_embeddings = passage_embeddings * 2 - 1
        
        if score_function == "cos_sim": 
            passage_embeddings, query_embeddings = normalize(passage_embeddings), normalize(query_embeddings)

        scores_arr = np.einsum("ijk,ik->ij", passage_embeddings, query_embeddings)
        sorted_indices = np.argsort(-scores_arr, axis=1)

        ids_arr = ids_arr[np.arange(num_queries)[:, None], sorted_indices]
        if self._passage_ids is not None:
            ids_arr = self._passage_ids[ids_arr.reshape(-1)].reshape(num_queries, -1)
        else:
            ids_arr = np.array([self.index.id_map.at(int(id_)) for id_ in ids_arr.reshape(-1)], dtype=np.int)
            ids_arr = ids_arr.reshape(num_queries, -1)

        scores_arr = scores_arr[np.arange(num_queries)[:, None], sorted_indices]
        logger.info("Total search time: %.3f", time.time() - start_time)

        return scores_arr[:, :k], ids_arr[:, :k]
    
    def save(self, fname: str):
        faiss.write_index_binary(self.index, fname)

    @classmethod
    def build(
        cls,
        passage_ids: List[int],
        passage_embeddings: np.ndarray,
        index: Optional[faiss.Index] = None,
        buffer_size: int = 50000,
    ):
        if index is None:
            index = faiss.IndexBinaryFlat(passage_embeddings.shape[1] * 8)
        for start in trange(0, len(passage_ids), buffer_size):
            index.add(passage_embeddings[start : start + buffer_size])

        return cls(index, passage_ids, passage_embeddings)