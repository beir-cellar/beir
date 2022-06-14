import os
from datasets import Dataset
from .util import cos_sim, dot_score
import logging
import torch
from typing import Dict, List
import math
import queue
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from evaluate.module import EvaluationModule, EvaluationModuleInfo
from datasets import Features, Value, Sequence
from datasets.utils.filelock import FileLock
from tqdm import tqdm
import time
logger = logging.getLogger(__name__)


class DummyMetric(EvaluationModule):
    def _info(self):
        return EvaluationModuleInfo(
            features=Features(
                {"cos_scores_top_k_values": Sequence(Value("float")), "cos_scores_top_k_idx": Sequence(Value("int64")), "batch_index": Value("int64")},
            ),
        )

    def _compute(self, cos_scores_top_k_values, cos_scores_top_k_idx, batch_index):
        return cos_scores_top_k_values, cos_scores_top_k_idx, batch_index


#Parent class for any dense model
class DenseRetrievalParallelExactSearch:
    
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = None, target_devices: List[str] = None, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu']*4
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
            
        query_ids = list(queries['id'])
        self.results = {qid: {} for qid in query_ids}
        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus = corpus.map(lambda x: {'len': len(x.get("title", "") + x.get("text", ""))}, num_proc=4)
        corpus = corpus.sort('len', reverse=True)

        # Initiate dataloader
        self.corpus_chunk_size = min(math.ceil(len(corpus) / len(self.target_devices) / 10), 5000) if self.corpus_chunk_size is None else self.corpus_chunk_size
        self.corpus_chunk_size = min(self.corpus_chunk_size, len(corpus)-1) # to avoid getting error in metric.compute()
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

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))

        # copy the query embeddings to all target devices
        self.query_embeddings = query_embeddings
        self.top_k = top_k
        self.score_function = score_function

        # Start the multi-process pool on all target devices
        SentenceTransformer._encode_multi_process_worker = self._encode_multi_process_worker
        pool = self.model.start_multi_process_pool(self.target_devices)

        start_time = time.time()
        for chunk_id, corpus_batch in enumerate(corpus_dl):
            with torch.no_grad():
                self.model.encode_corpus_parallel(
                    corpus_batch, pool=pool, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor, chunk_id=chunk_id)

        # Stop the proccesses in the pool and free memory
        self.model.stop_multi_process_pool(pool, len_queue=chunk_id+1)

        end_time = time.time()
        logger.info("Encoded all batches in {:.2f} seconds".format(end_time - start_time))

        # Gather all results
        metric = DummyMetric(experiment_id="test_experiment", num_process=len(self.target_devices), process_id=0)
        metric.filelock = FileLock(os.path.join(metric.data_dir, f"{metric.experiment_id}-{metric.num_process}-{metric.process_id}.arrow.lock"))

        cos_scores_top_k_values, cos_scores_top_k_idx, chunk_ids = metric.compute()

        logger.info("Formatting results...")
        # Load corpus ids in memory
        corpus_ids = corpus['id']
        for query_itr in tqdm(range(len(query_embeddings))):
            query_id = query_ids[query_itr]
            for i in range(len(cos_scores_top_k_values)):
                batch_num = chunk_ids[i]
                sub_corpus_id = cos_scores_top_k_idx[i][query_itr] + batch_num * self.corpus_chunk_size
                score = cos_scores_top_k_values[i][query_itr]
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
        process_id = self.target_devices.index(target_device)
        metric = DummyMetric(experiment_id="test_experiment", num_process=len(self.target_devices), process_id=process_id)
        while True:
            try:
                id, batch_size, sentences = input_queue.get()
                corpus_embeds = model.encode(
                    sentences, device=target_device, show_progress_bar=False, convert_to_tensor=True, batch_size=batch_size
                )

                cos_scores = self.score_functions[self.score_function](self.query_embeddings.to(corpus_embeds.device), corpus_embeds)
                cos_scores[torch.isnan(cos_scores)] = -1

                #Get top-k values
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(self.top_k+1, len(cos_scores[1])), dim=1, largest=True, sorted=False)
                cos_scores_top_k_values = cos_scores_top_k_values.T
                cos_scores_top_k_idx = cos_scores_top_k_idx.T

                # Store results in an Apache Arrow table
                metric.add_batch(cos_scores_top_k_values=cos_scores_top_k_values, cos_scores_top_k_idx=cos_scores_top_k_idx, batch_index=[id]*len(cos_scores_top_k_values))

                # Alarm that process finished processing a batch
                results_queue.put(None)
            except queue.Empty:
                break
