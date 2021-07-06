from .util import cos_sim, dot_score, normalize, save_dict_to_tsv, load_tsv_to_dict
from .faiss_index import FaissBinaryIndex, FaissPQIndex, FaissHNSWIndex, FaissPCAIndex, FaissIndex
import logging
import sys
import torch
import faiss
import numpy as np
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

#Parent class for any faiss search
class DenseRetrievalFaissSearch:
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.score_functions = ['cos_sim','dot']
        self.mapping_tsv_keys = ["beir-docid", "faiss-docid"]
        self.faiss_index = None
        self.dim_size = 0
        self.results = {}
        self.mapping = {}
        self.rev_mapping = {}
    
    def _create_mapping_ids(self, corpus_ids):
        if not all(isinstance(doc_id, int) for doc_id in corpus_ids):
            for idx in range(len(corpus_ids)):
                self.mapping[corpus_ids[idx]] = idx
                self.rev_mapping[idx] = corpus_ids[idx]
    
    def _load(self, input_dir: str, prefix: str, ext: str):

        # Load ID mappings from file
        input_mappings_path = os.path.join(input_dir, "{}.{}.tsv".format(prefix, ext))
        logger.info("Loading Faiss ID-mappings from path: {}".format(input_mappings_path))
        self.mapping = load_tsv_to_dict(input_mappings_path, header=True)
        self.rev_mapping = {v: k for k, v in self.mapping.items()}
        passage_ids = sorted(list(self.rev_mapping))
        
        # Load Faiss Index from disk
        input_faiss_path = os.path.join(input_dir, "{}.{}.faiss".format(prefix, ext))
        logger.info("Loading Faiss Index from path: {}".format(input_faiss_path))
        
        return input_faiss_path, passage_ids

    def save(self, output_dir: str, prefix: str, ext: str):
        
        # Save BEIR -> Faiss ids mappings
        save_mappings_path = os.path.join(output_dir, "{}.{}.tsv".format(prefix, ext))
        logger.info("Saving Faiss ID-mappings to path: {}".format(save_mappings_path))
        save_dict_to_tsv(self.mapping, save_mappings_path, keys=self.mapping_tsv_keys)

        # Save Faiss Index to disk
        save_faiss_path = os.path.join(output_dir, "{}.{}.faiss".format(prefix, ext))
        logger.info("Saving Faiss Index to path: {}".format(save_faiss_path))
        self.faiss_index.save(save_faiss_path)
        logger.info("Index size: {:.2f}MB".format(os.path.getsize(save_faiss_path)*0.000001))
    
    def _index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None):
        
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        corpus_ids = list(corpus.keys())
        self._create_mapping_ids(corpus_ids)
        corpus = [corpus[cid] for cid in corpus_ids]
        normalize_embeddings = True if score_function == "cos_sim" else False

        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))
            
            #Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=True, 
                normalize_embeddings=normalize_embeddings)
            
            if not batch_num: 
                corpus_embeddings = sub_corpus_embeddings
            else:
                corpus_embeddings = np.vstack([corpus_embeddings, sub_corpus_embeddings])
        
        #Index chunk of corpus into faiss index
        logger.info("Indexing Passages into Faiss...") 
        
        faiss_ids = [self.mapping.get(corpus_id) for corpus_id in corpus_ids]
        self.dim_size = corpus_embeddings.shape[1]

        del sub_corpus_embeddings

        return faiss_ids, corpus_embeddings
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str], 
               top_k: int,
               score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        assert score_function in self.score_functions

        if not self.faiss_index: self.index(corpus, score_function)

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, show_progress_bar=True, batch_size=self.batch_size)

        faiss_scores, faiss_doc_ids = self.faiss_index.search(query_embeddings, top_k, **kwargs)
        
        for idx in range(len(query_ids)):
            scores = [float(score) for score in faiss_scores[idx]]
            if len(self.rev_mapping) != 0:
                doc_ids = [self.rev_mapping[doc_id] for doc_id in faiss_doc_ids[idx]]
            else:
                doc_ids = [str(doc_id) for doc_id in faiss_doc_ids[idx]]
            self.results[query_ids[idx]] = dict(zip(doc_ids, scores))
        
        return self.results


class BinaryFaissSearch(DenseRetrievalFaissSearch):

    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "bin"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index_binary(input_faiss_path)
        passage_embeddings = base_index.reconstruct(0)
        for idx in range(1, len(passage_ids)):
            passage_embeddings = np.vstack((passage_embeddings, base_index.reconstruct(idx)))            
        self.faiss_index = FaissBinaryIndex(base_index, passage_ids, passage_embeddings)

    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function)
        logger.info("Using Binary Hashing in Flat Mode!")
        base_index = faiss.IndexBinaryFlat(self.dim_size * 8)
        self.faiss_index = FaissBinaryIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "bin"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:

        return super().search(corpus, queries, top_k, score_function, **kwargs)
    

class PQFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, num_of_centroids: int = 96, 
                 code_size: int = 8, similarity_metric=faiss.METRIC_INNER_PRODUCT, **kwargs):
        super(PQFaissSearch, self).__init__(model, batch_size, corpus_chunk_size, **kwargs)
        self.num_of_centroids = num_of_centroids
        self.code_size = code_size
        self.similarity_metric = similarity_metric
    
    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "pq"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        self.faiss_index = FaissPQIndex(base_index, passage_ids)

    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None, **kwargs):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)            
        logger.info("Using Product Quantization (PQ) in Flat mode!")
        logger.info("Parameters Used: num_of_centroids: {} ".format(self.num_of_centroids))
        logger.info("Parameters Used: code_size: {}".format(self.code_size))

        base_index = faiss.IndexPQ(self.dim_size, self.num_of_centroids, self.code_size, self.similarity_metric)
        self.faiss_index = FaissPQIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "pq"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        return super().search(corpus, queries, top_k, score_function, **kwargs)


class HNSWFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, hnsw_store_n: int = 512, 
                 hnsw_ef_search: int = 128, hnsw_ef_construction: int = 200, similarity_metric=faiss.METRIC_INNER_PRODUCT, **kwargs):
        super(HNSWFaissSearch, self).__init__(model, batch_size, corpus_chunk_size, **kwargs)
        self.hnsw_store_n = hnsw_store_n
        self.hnsw_ef_search = hnsw_ef_search
        self.hnsw_ef_construction = hnsw_ef_construction
        self.similarity_metric = similarity_metric
    
    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "hnsw"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        self.faiss_index = FaissHNSWIndex(base_index, passage_ids)
    
    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None, **kwargs):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)

        logger.info("Using Approximate Nearest Neighbours (HNSW) in Flat Mode!")
        logger.info("Parameters Required: hnsw_store_n: {}".format(self.hnsw_store_n))
        logger.info("Parameters Required: hnsw_ef_search: {}".format(self.hnsw_ef_search))
        logger.info("Parameters Required: hnsw_ef_construction: {}".format(self.hnsw_ef_construction))
        
        base_index = faiss.IndexHNSWFlat(self.dim_size + 1, self.hnsw_store_n, self.similarity_metric)
        base_index.hnsw.efSearch = self.hnsw_ef_search
        base_index.hnsw.efConstruction = self.hnsw_ef_construction
        self.faiss_index = FaissHNSWIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "hnsw"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        return super().search(corpus, queries, top_k, score_function, **kwargs)

class FlatIPFaissSearch(DenseRetrievalFaissSearch):
    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "flat"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        self.faiss_index = FaissIndex(base_index, passage_ids)

    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None, **kwargs):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)
        base_index = faiss.IndexFlatIP(self.dim_size)
        self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, base_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "flat"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        return super().search(corpus, queries, top_k, score_function, **kwargs)

class PCAFaissSearch(DenseRetrievalFaissSearch):
    def __init__(self, model, base_index: faiss.Index, output_dimension: int, batch_size: int = 128, 
                corpus_chunk_size: int = 50000, **kwargs):
        super(PCAFaissSearch, self).__init__(model, batch_size, corpus_chunk_size, **kwargs)
        self.base_index = base_index
        self.output_dim = output_dimension

    def load(self, input_dir: str, prefix: str = "my-index", ext: str = "pca"):
        input_faiss_path, passage_ids = super()._load(input_dir, prefix, ext)
        base_index = faiss.read_index(input_faiss_path)
        self.faiss_index = FaissPCAIndex(base_index, passage_ids)

    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None, **kwargs):
        faiss_ids, corpus_embeddings = super()._index(corpus, score_function, **kwargs)
        logger.info("Creating PCA Matrix...")
        logger.info("Input Dimension: {}, Output Dimension: {}".format(self.dim_size, self.output_dim))
        pca_matrix = faiss.PCAMatrix(self.dim_size, self.output_dim, 0, True)
        final_index = faiss.IndexPreTransform(pca_matrix, self.base_index)
        self.faiss_index = FaissPCAIndex.build(faiss_ids, corpus_embeddings, final_index)

    def save(self, output_dir: str, prefix: str = "my-index", ext: str = "pca"):
        super().save(output_dir, prefix, ext)
    
    def search(self, 
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str], 
            top_k: int,
            score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        return super().search(corpus, queries, top_k, score_function, **kwargs)