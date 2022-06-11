from datasets import Dataset
from .util import cos_sim, dot_score
import logging
import torch
from typing import Dict, List
import math
import queue
from sentence_transformers import SentenceTransformer
from multiprocess import set_start_method
from accelerate import Accelerator, DistributedType
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

#Parent class for any dense model

class DenseRetrievalParallelExactSearch:
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = None, target_devices: List[str] = None, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.target_devices = target_devices  # PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used, or 4 CPU processes
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True #TODO: implement no progress bar if false
        self.convert_to_tensor = True
        self.results = {}

        self.query_embeddings = {}
        self.top_k = None
        self.score_function = None
    
    def search(self, 
               corpus: Dataset, 
               queries: Dataset, 
               top_k: List[int], 
               score_function: str,
               **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries['id'])
        self.results = {qid: {} for qid in query_ids}

        # initialize accelerator
        accelerator = Accelerator(cpu=not torch.cuda.is_available(), mixed_precision=None)

        # Instantiate dataloader
        queries_dl = DataLoader(queries, batch_size=self.batch_size)
        corpus_dl = DataLoader(corpus, batch_size=self.batch_size)



        self.model, queries_dl, corpus_dl = accelerator.prepare(self.model, queries_dl, corpus_dl)


        # query_embeddings = self.model.encode_queries(
        #     queries, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)



        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        if self.target_devices is None:
            if torch.cuda.is_available():
                self.target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                self.target_devices = ['cpu']*4
        # copy the query embeddings to all target devices
        self.query_embeddings = {target_device: query_embeddings.to(target_device) for target_device in self.target_devices}
        self.top_k = top_k
        self.score_function = score_function    
        
        SentenceTransformer._encode_multi_process_worker = self._encode_multi_process_worker
        pool = self.model.start_multi_process_pool(self.target_devices)
        self.corpus_chunk_size = min(math.ceil(len(corpus) / len(pool["processes"]) / 10), 5000) if self.corpus_chunk_size is None else self.corpus_chunk_size
    
        #Encode chunk of corpus (Send them to device:0 once computed for dot-product)    
        cos_scores_top_k_values, cos_scores_top_k_idx = self.model.encode_corpus_parallel(
            corpus,
            pool=pool,
            batch_size=self.batch_size,
            chunk_size=self.corpus_chunk_size,
            )
        cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
        cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

        # Stop the proccesses in the pool and free memory
        self.model.stop_multi_process_pool(pool)

        for query_itr in range(len(query_embeddings)):
            query_id = query_ids[query_itr]
            for i in range(len(cos_scores_top_k_values[query_itr])):
                batch_num = i // (self.top_k+1)
                sub_corpus_id = cos_scores_top_k_idx[query_itr][i] + batch_num * self.corpus_chunk_size
                score = cos_scores_top_k_values[query_itr][i]
                corpus_id = corpus_ids[sub_corpus_id]
                if corpus_id != query_id:
                    self.results[query_id][corpus_id] = score

        return self.results 

    def _encode_multi_process_worker(self, target_device: str, model, input_queue, results_queue):
        """
        (taken from UKPLab/sentence-transformers/sentence_transformers/SentenceTransformer.py)
        Internal working process to encode sentences in multi-process setup.
        Note: Added distributed similarity computing and finding top k similar docs.
        """
        while True:
            try:
                id, batch_size, sentences = input_queue.get()
                embeddings = model.encode(
                    sentences, device=target_device, show_progress_bar=False, convert_to_tensor=True, batch_size=batch_size
                )
                cos_scores = self.score_functions[self.score_function](self.query_embeddings[target_device], embeddings)
                cos_scores[torch.isnan(cos_scores)] = -1

                #Get top-k values
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(self.top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=False)

                results_queue.put([id, cos_scores_top_k_values, cos_scores_top_k_idx])
            except queue.Empty:
                break
