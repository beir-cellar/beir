from .util import cos_sim, dot_score, normalize, save_dict_to_tsv, load_tsv_to_dict
from .faiss_index import FaissBinaryIndex, FaissPQIndex, FaissHNSWIndex, FaissIndex
import logging
import sys
import torch
import faiss
import numpy as np
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

#Parent class for any dense model
class DenseRetrievalFaissSearch:
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, index_save_path: str = None, 
                 use_binary_hash: bool = False, use_quantization: bool = False, use_hnsw: bool = False, faiss_params: dict = {}, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.score_functions = ['cos_sim','dot']
        self.mapping_tsv_keys = ["beir-docid", "faiss-docid"]
        self.faiss_index = None
        self.index_save_path = index_save_path
        self.use_binary_hash = use_binary_hash
        self.use_quantization = use_quantization
        self.use_hnsw = use_hnsw
        self.faiss_params = faiss_params
        self.results = {}
        self.mapping = {}
        self.rev_mapping = {}
    
    def _create_mapping_ids(self, corpus_ids):
        if not all(isinstance(doc_id, int) for doc_id in corpus_ids):
            for idx in range(len(corpus_ids)):
                self.mapping[corpus_ids[idx]] = idx
                self.rev_mapping[idx] = corpus_ids[idx]
    
    def load(self, input_dir: str, prefix: str = "index"):
        
        logger.info("Loading Faiss Index from path: {}".format(input_dir))

        # Load ID mappings from file
        input_mappings_path = os.path.join(input_dir, "mappings.tsv")
        self.mapping = load_tsv_to_dict(input_mappings_path, header=True)
        self.rev_mapping = {v: k for k, v in self.mapping.items()}
        passage_ids = sorted(list(self.rev_mapping))
        
        # Load Faiss Index from disk
        input_faiss_path = os.path.join(input_dir, "{}.faiss".format(prefix))
        
        if self.use_binary_hash:
            base_index = faiss.read_index_binary(input_faiss_path) 
            with open(os.path.join(input_dir, "{}.npy".format(prefix)), 'rb') as f:
                passage_embeddings = np.load(f)
            self.faiss_index = FaissBinaryIndex(base_index, passage_ids, passage_embeddings)
        
        elif self.use_quantization:
            base_index = faiss.read_index(input_faiss_path) 
            self.faiss_index = FaissPQIndex(base_index, passage_ids)
        
        elif self.use_hnsw:
            base_index = faiss.read_index(input_faiss_path) 
            self.faiss_index = FaissHNSWIndex(base_index, passage_ids)
        
        else:
            base_index = faiss.read_index(input_faiss_path) 
            self.faiss_index = FaissIndex(base_index, passage_ids)

    def save(self, output_dir: str, prefix: str = "index"):
        
        logger.info("Saving Faiss Index to path: {}".format(output_dir))

        # Save BEIR -> Faiss ids mappings
        save_mappings_path = os.path.join(output_dir, "mappings.tsv")
        save_dict_to_tsv(self.mapping, save_mappings_path, keys=self.mapping_tsv_keys)

        # Save Faiss Index to disk
        save_faiss_path = os.path.join(output_dir, "{}.faiss".format(prefix))
        self.faiss_index.save(save_faiss_path)
        logger.info("Index size: {:.2f}MB".format(os.path.getsize(save_faiss_path)*0.000001))
    
    def index(self, corpus: Dict[str, Dict[str, str]], score_function: str = None, **kwargs):
        
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        corpus_ids = list(corpus.keys())
        self._create_mapping_ids(corpus_ids)
        corpus = [corpus[cid] for cid in corpus_ids]

        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info("Encoding Batch {}/{}...".format(batch_num+1, len(itr)))
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))
            
            #Encode chunk of corpus    
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                show_progress_bar=True, 
                batch_size=self.batch_size)
            
            if not batch_num: 
                corpus_embeddings = sub_corpus_embeddings
            else:
                corpus_embeddings = np.vstack([corpus_embeddings, sub_corpus_embeddings])
        
        #Index chunk of corpus into faiss index
        logger.info("Indexing Passages into Faiss...") 

        if score_function == "cos_sim": corpus_embeddings = normalize(corpus_embeddings)
        
        faiss_ids = [self.mapping.get(corpus_id) for corpus_id in corpus_ids]
        dim_size = corpus_embeddings.shape[1]

        if self.use_binary_hash:
            hash_num_bits = self.faiss_params.get("hash_num_bits", 768)
            
            logger.info("Using Binary Hashing with Faiss!")
            logger.info("Parameters Required: hash_num_bits: {}".format(hash_num_bits))  
            
            base_index = faiss.IndexBinaryHash(dim_size * 8, hash_num_bits)
            self.faiss_index = FaissBinaryIndex.build(faiss_ids, corpus_embeddings, base_index)
        
        elif self.use_quantization:
            num_of_centroids = self.faiss_params.get("num_of_centroids", 96)
            code_size = self.faiss_params.get("code_size", 8)
            
            logger.info("Using Product Quantization (PQ) in Flat mode!")
            logger.info("Parameters Required: num_of_centroids: {} ".format(num_of_centroids))
            logger.info("Parameters Required: code_size: {}".format(code_size))

            base_index = faiss.IndexPQ(dim_size, num_of_centroids, code_size, faiss.METRIC_INNER_PRODUCT)
            self.faiss_index = FaissPQIndex.build(faiss_ids, corpus_embeddings, base_index)
        
        elif self.use_hnsw:
            hnsw_store_n = self.faiss_params.get("hnsw_store_n", 512)
            hnsw_ef_search = self.faiss_params.get("hnsw_ef_search", 128)
            hnsw_ef_construction = self.faiss_params.get("hnsw_ef_construction", 200)

            logger.info("Using Approximate Nearest Neighbours (HNSW) in Flat Mode!")
            logger.info("Parameters Required: hnsw_store_n: {}".format(hnsw_store_n))
            logger.info("Parameters Required: hnsw_ef_search: {}".format(hnsw_ef_search))
            logger.info("Parameters Required: hnsw_ef_construction: {}".format(hnsw_ef_construction))
            
            base_index = faiss.IndexHNSWFlat(dim_size + 1, hnsw_store_n, faiss.METRIC_INNER_PRODUCT)
            base_index.hnsw.efSearch = hnsw_ef_search
            base_index.hnsw.efConstruction = hnsw_ef_construction
            self.faiss_index = FaissHNSWIndex.build(faiss_ids, corpus_embeddings, base_index)
        
        else:
            logger.info("Using Exact Search for Inner Product in Flat mode!")
            base_index = faiss.IndexFlatIP(dim_size)
            self.faiss_index = FaissIndex.build(faiss_ids, corpus_embeddings, base_index)
        
        logger.info("Faiss indexing completed! {} Documents Indexed...".format(len(self.faiss_index._passage_ids)))
        
        del sub_corpus_embeddings, corpus_embeddings
        
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]],
               queries: Dict[str, str], 
               top_k: int,
               score_function = str, **kwargs) -> Dict[str, Dict[str, float]]:
        
        assert score_function in self.score_functions
        
        ## Used for Indexing
        if not self.faiss_index: self.index(corpus, score_function, **kwargs)

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