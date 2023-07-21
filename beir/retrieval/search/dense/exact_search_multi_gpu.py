from .. import BaseSearch
from .util import cos_sim, dot_score
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from datasets import Features, Value
from datasets.utils.filelock import FileLock
from datasets import Array2D, Dataset
from tqdm.autonotebook import tqdm
from typing import Dict, List

import logging
import torch
import math
import queue
import os
import time
import numpy as np

logger = logging.getLogger(__name__)

import importlib.util

### HuggingFace Evaluate library (pip install evaluate) only available with Python >= 3.7.
### Hence for no import issues with Python 3.6, we move DummyMetric if ``evaluate`` library is found.
if importlib.util.find_spec("evaluate") is not None:
    from evaluate.module import EvaluationModule, EvaluationModuleInfo
    
    class DummyMetric(EvaluationModule):
        len_queries = None
        
        def _info(self):
            return EvaluationModuleInfo(
                description="dummy metric to handle storing middle results",
                citation="",
                features=Features(
                    {"cos_scores_top_k_values": Array2D((None, self.len_queries), "float32"), "cos_scores_top_k_idx": Array2D((None, self.len_queries), "int32"), "batch_index": Value("int32")},
                ),
            )

        def _compute(self, cos_scores_top_k_values, cos_scores_top_k_idx, batch_index):
            for i in range(len(batch_index) - 1, -1, -1):
                if batch_index[i] == -1:
                    del cos_scores_top_k_values[i]
                    del cos_scores_top_k_idx[i]
            cos_scores_top_k_values = np.concatenate(cos_scores_top_k_values, axis=0)
            cos_scores_top_k_idx = np.concatenate(cos_scores_top_k_idx, axis=0)
            return cos_scores_top_k_values, cos_scores_top_k_idx

        def warmup(self):
            """
            Add dummy batch to acquire filelocks for all processes and avoid getting errors
            """
            self.add_batch(cos_scores_top_k_values=torch.ones((1, 1, self.len_queries), dtype=torch.float32), cos_scores_top_k_idx=torch.ones((1, 1, self.len_queries), dtype=torch.int32), batch_index=-torch.ones(1, dtype=torch.int32))

#Parent class for any dense model
class DenseRetrievalParallelExactSearch(BaseSearch):
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = None, target_devices: List[str] = None, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu']*1 # 4
        self.target_devices = target_devices  # PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used, or 4 CPU processes
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.score_function_desc = {'cos_sim': "Cosine Similarity", 'dot': "Dot Product"}
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}

        self.query_embeddings = {}
        self.top_k = None
        self.score_function = None
        self.sort_corpus = True
        self.experiment_id = "exact_search_multi_gpu" # f"test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    def search(self, 
               corpus: Dataset, 
               queries: Dataset, 
               top_k: int, 
               score_function: str,
               **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (cos_sim) for cosine similarity or (dot) for dot product".format(score_function))
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        if importlib.util.find_spec("evaluate") is None:
            raise ImportError("evaluate library not available. Please do ``pip install evaluate`` library with Python>=3.7 (not available with Python 3.6) to use distributed and multigpu evaluation.")
            
        self.corpus_chunk_size = min(math.ceil(len(corpus) / len(self.target_devices) / 10), 5000) if self.corpus_chunk_size is None else self.corpus_chunk_size
        self.corpus_chunk_size = min(self.corpus_chunk_size, len(corpus)-1) # to avoid getting error in metric.compute()
        
        if self.sort_corpus:
            logger.info("Sorting Corpus by document length (Longest first)...")
            corpus = corpus.map(lambda x: {'len': len(x.get("title", "") + x.get("text", ""))}, num_proc=4)
            corpus = corpus.sort('len', reverse=True)

        # Initiate dataloader
        queries_dl = DataLoader(queries, batch_size=self.corpus_chunk_size)
        corpus_dl = DataLoader(corpus, batch_size=self.corpus_chunk_size)

        # Encode queries
        logger.info("Encoding Queries in batches...")
        query_embeddings = []
        for step, queries_batch in enumerate(queries_dl):
            with torch.no_grad():
                q_embeds = self.model.encode_queries(
                    queries_batch['text'], batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
            query_embeddings.append(q_embeds)
        query_embeddings = torch.cat(query_embeddings, dim=0)

        # copy the query embeddings to all target devices
        self.query_embeddings = query_embeddings
        self.top_k = top_k
        self.score_function = score_function

        # Start the multi-process pool on all target devices
        SentenceTransformer._encode_multi_process_worker = self._encode_multi_process_worker
        pool = self.model.start_multi_process_pool(self.target_devices)

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        start_time = time.time()
        for chunk_id, corpus_batch in tqdm(enumerate(corpus_dl), total=len(corpus) // self.corpus_chunk_size):
            with torch.no_grad():
                self.model.encode_corpus_parallel(
                    corpus_batch, pool=pool, batch_size=self.batch_size, chunk_id=chunk_id)

        # Stop the proccesses in the pool and free memory
        self.model.stop_multi_process_pool(pool)

        end_time = time.time()
        logger.info("Encoded all batches in {:.2f} seconds".format(end_time - start_time))

        # Gather all results
        DummyMetric.len_queries = len(queries)
        metric = DummyMetric(experiment_id=self.experiment_id, num_process=len(self.target_devices), process_id=0)
        metric.filelock = FileLock(os.path.join(metric.data_dir, f"{metric.experiment_id}-{metric.num_process}-{metric.process_id}.arrow.lock"))
        metric.cache_file_name = os.path.join(metric.data_dir, f"{metric.experiment_id}-{metric.num_process}-{metric.process_id}.arrow")

        cos_scores_top_k_values, cos_scores_top_k_idx = metric.compute()

        # sort similar docs for each query by cosine similarity and keep only top_k
        sorted_idx = np.argsort(cos_scores_top_k_values, axis=0)[::-1]
        sorted_idx = sorted_idx[:self.top_k+1]
        cos_scores_top_k_values = np.take_along_axis(cos_scores_top_k_values, sorted_idx, axis=0)
        cos_scores_top_k_idx = np.take_along_axis(cos_scores_top_k_idx, sorted_idx, axis=0)

        logger.info("Formatting results...")
        # Load corpus ids in memory
        query_ids = queries['id']
        corpus_ids = corpus['id']
        self.results = {qid: {} for qid in query_ids}
        for query_itr in tqdm(range(len(query_embeddings))):
            query_id = query_ids[query_itr]
            for i in range(len(cos_scores_top_k_values)):
                sub_corpus_id = cos_scores_top_k_idx[i][query_itr]
                score = cos_scores_top_k_values[i][query_itr].item() # convert np.float to float
                corpus_id = corpus_ids[sub_corpus_id]
                if corpus_id != query_id:
                    self.results[query_id][corpus_id] = score
        return self.results 

    def _encode_multi_process_worker(self, process_id, device, model, input_queue, results_queue):
        """
        (taken from UKPLab/sentence-transformers/sentence_transformers/SentenceTransformer.py)
        Internal working process to encode sentences in multi-process setup.
        Note: Added distributed similarity computing and finding top k similar docs.
        """
        DummyMetric.len_queries = len(self.query_embeddings)
        metric = DummyMetric(experiment_id=self.experiment_id, num_process=len(self.target_devices), process_id=process_id)
        metric.warmup()
        with torch.no_grad():
            while True:
                try:
                    id, batch_size, sentences = input_queue.get()
                    corpus_embeds = model.encode(
                        sentences, device=device, show_progress_bar=False, convert_to_tensor=True, batch_size=batch_size
                    ).detach()

                    cos_scores = self.score_functions[self.score_function](self.query_embeddings.to(corpus_embeds.device), corpus_embeds).detach()
                    cos_scores[torch.isnan(cos_scores)] = -1

                    #Get top-k values
                    cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(self.top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=False)
                    cos_scores_top_k_values = cos_scores_top_k_values.T.unsqueeze(0).detach()
                    cos_scores_top_k_idx = cos_scores_top_k_idx.T.unsqueeze(0).detach()

                    # correct sentence ids
                    cos_scores_top_k_idx += id * self.corpus_chunk_size

                    # Store results in an Apache Arrow table
                    metric.add_batch(cos_scores_top_k_values=cos_scores_top_k_values, cos_scores_top_k_idx=cos_scores_top_k_idx, batch_index=[id]*len(cos_scores_top_k_values))

                    # Alarm that process finished processing a batch
                    results_queue.put(None)
                except queue.Empty:
                    break
