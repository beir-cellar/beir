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
import heapq
from datasets.distributed import split_dataset_by_node
import torch.distributed as dist

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
        self.encoding_batch_size = batch_size
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
    
    @torch.no_grad()
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

        world_size = int(os.getenv("WORLD_SIZE", 1))

        # WARNING: We remove the query from results if it exists in corpus
        corpus = corpus.filter(lambda x: x["id"] not in queries["id"])

        logger.warning(f"corpus_chunk_size wasn't specified. Setting it to {min(math.ceil(len(corpus) / len(self.target_devices) / 10), 5000)}")
        self.corpus_chunk_size = min(math.ceil(len(corpus) / len(self.target_devices) / 10), 5000) if self.corpus_chunk_size is None else self.corpus_chunk_size

        # limit corpus
        # import joblib
        # query_ids = joblib.load("query_ids.pkl")
        # corpus_ids = joblib.load("corpus_ids.pkl")
        # # check if corpus_id in queries
        # assert len(set(corpus_ids).intersection(set(query_ids))) == 0, "corpus_ids and query_ids must be disjoint"
        # # keep only queries with `id` column in query_ids
        # queries = queries.filter(lambda x: x["id"] in query_ids)
        # corpus = corpus.filter(lambda x: x["id"] in corpus_ids)

        # # check all corpus_ids exist in corpus
        # for corpus_id in corpus_ids:
        #     assert corpus_id in corpus["id"], f"corpus_id: {corpus_id} not found in corpus"
        # queries = queries.select(range(min(len(queries), 10)))
        # corpus = corpus.select(range(min(len(corpus), 100)))
        # top_k = 10

        # add index column
        corpus = corpus.add_column("mteb_index", range(len(corpus)))

        # Split corpus across devices to parallelize encoding
        assert isinstance(corpus, Dataset), f"Corpus must be a Dataset object, but got {type(corpus)}"
        local_corpus = split_dataset_by_node(corpus, rank=int(os.getenv("RANK", 0)), world_size=world_size)

        all_ranks_corpus_start_idx = local_corpus["mteb_index"][0] # I'm assuming `split_dataset_by_node` splits local_corpus evenly while keeping same order

        # if self.sort_corpus:
        #     # SentenceTransformer.encode sorts sentences anyway
        #     logger.info("Sorting Corpus by document length (Longest first)...")
        #     local_corpus = local_corpus.map(lambda x: {'len': len(x.get("title", "") + x.get("text", ""))}, num_proc=4)
        #     local_corpus = local_corpus.sort('len', reverse=True)

        # Initiate dataloader
        queries_dl = DataLoader(queries, batch_size=self.corpus_chunk_size)
        corpus_dl = DataLoader(local_corpus, batch_size=self.corpus_chunk_size)

        # Encode queries (all ranks do this)
        logger.info("Encoding Queries in batches...")
        query_embeddings = []
        for step, queries_batch in enumerate(queries_dl):
            q_embeds = self.model.encode_queries(
                queries_batch['text'], batch_size=self.encoding_batch_size, show_progress_bar=self.show_progress_bar, convert_to_tensor=self.convert_to_tensor)
            query_embeddings.append(q_embeds)
        query_embeddings = torch.cat(query_embeddings, dim=0)

        self.query_embeddings = query_embeddings
        self.top_k = top_k
        self.score_function = score_function

        # Encode corpus (each rank encodes `local_corpus`)
        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        start_time = time.time()
        query_ids = queries["id"]
        self.results = {qid: {} for qid in query_ids}
        all_chunks_cos_scores_top_k_values = []
        all_chunks_cos_scores_top_k_idx = []
        for chunk_id, corpus_batch in tqdm(enumerate(corpus_dl), total=len(local_corpus) // self.corpus_chunk_size):
            corpus_start_idx = chunk_id * self.corpus_chunk_size
            
            # Encode chunk of local_corpus 
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus_batch, 
                batch_size=self.encoding_batch_size,
                show_progress_bar=self.show_progress_bar, 
                convert_to_tensor = self.convert_to_tensor
            )

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings) # (num_queries, self.corpus_chunk_size)
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, cos_scores.shape[1]), dim=1, largest=True) # (num_queries, top_k)

            # Displace index by corpus_start_idx
            cos_scores_top_k_idx += corpus_start_idx

            all_chunks_cos_scores_top_k_values.append(cos_scores_top_k_values)
            all_chunks_cos_scores_top_k_idx.append(cos_scores_top_k_idx)


        end_time = time.time()
        logger.info("Encoded all batches in {:.2f} seconds".format(end_time - start_time))

        # TODO: have shapes (top_k, num_queries) instead?
        # keep only top_k top scoring docs from this rank
        all_chunks_cos_scores_top_k_values = torch.cat(all_chunks_cos_scores_top_k_values, dim=1) # (num_queries, (top_k)*n_chunks)
        all_chunks_cos_scores_top_k_idx = torch.cat(all_chunks_cos_scores_top_k_idx, dim=1) # (num_queries, (top_k)*n_chunks) 
        cos_scores_top_k_values, temp_cos_scores_top_k_idx = torch.topk(all_chunks_cos_scores_top_k_values, min(top_k, all_chunks_cos_scores_top_k_values.shape[1]), dim=1, largest=True) # (num_queries, top_k)

        # `all_chunks_cos_scores_top_k_idx` should index `corpus_ids` and not `all_chunks_cos_scores_top_k_values`
        all_chunks_cos_scores_top_k_idx = torch.gather(all_chunks_cos_scores_top_k_idx, dim=1, index=temp_cos_scores_top_k_idx) # (num_queries, top_k) // indexes between (0, len(local_corpus))

        # Displace index by all_ranks_corpus_start_idx so that we index `corpus` and not `local_corpus`
        all_chunks_cos_scores_top_k_idx += all_ranks_corpus_start_idx

        # If local_corpus doesn't have top_k samples we pad scores to `pad_gathered_tensor_to`
        if len(local_corpus) < top_k:
            pad_gathered_tensor_to = math.ceil(len(corpus) / world_size)
            cos_scores_top_k_values = torch.cat(
                [cos_scores_top_k_values, torch.ones((cos_scores_top_k_values.shape[0], pad_gathered_tensor_to - cos_scores_top_k_values.shape[1]), device=cos_scores_top_k_values.device) * -1], 
                dim=1
            )
            all_chunks_cos_scores_top_k_idx = torch.cat(
                [all_chunks_cos_scores_top_k_idx, torch.ones((all_chunks_cos_scores_top_k_idx.shape[0], pad_gathered_tensor_to - all_chunks_cos_scores_top_k_idx.shape[1]), device=all_chunks_cos_scores_top_k_idx.device, dtype=all_chunks_cos_scores_top_k_idx.dtype) * -1], 
                dim=1
            )
        else:
            pad_gathered_tensor_to = top_k

        # all gather top_k results from all ranks
        n_queries = len(query_ids)
        all_ranks_top_k_values = torch.empty((world_size, n_queries, pad_gathered_tensor_to), dtype=cos_scores_top_k_values.dtype, device="cuda")
        all_ranks_top_k_idx = torch.empty((world_size, n_queries, pad_gathered_tensor_to), dtype=all_chunks_cos_scores_top_k_idx.dtype, device="cuda")
        dist.barrier()
        logger.info(f"All gather top_k values from all ranks...")
        dist.all_gather_into_tensor(all_ranks_top_k_values, cos_scores_top_k_values)
        logger.info(f"All gather top_k idx from all ranks...")
        dist.all_gather_into_tensor(all_ranks_top_k_idx, all_chunks_cos_scores_top_k_idx)
        logger.info(f"All gather ... Done!")

        all_ranks_top_k_values = all_ranks_top_k_values.permute(1, 0, 2).reshape(n_queries, -1) # (n_queries, world_size*(pad_gathered_tensor_to))
        all_ranks_top_k_idx = all_ranks_top_k_idx.permute(1, 0, 2).reshape(n_queries, -1) # (n_queries, world_size*(pad_gathered_tensor_to))

        # keep only top_k top scoring docs from all ranks
        cos_scores_top_k_values, temp_cos_scores_top_k_idx = torch.topk(all_ranks_top_k_values, min(top_k, all_ranks_top_k_values.shape[1]), dim=1, largest=True) # (num_queries, top_k)

        # `all_ranks_top_k_idx` should index `corpus_ids` and not `all_ranks_top_k_values`
        all_ranks_top_k_idx = torch.gather(all_ranks_top_k_idx, dim=1, index=temp_cos_scores_top_k_idx) # (num_queries, top_k) // indexes between (0, len(corpus))

        # fill in results
        for qid, top_k_values, top_k_idx in tqdm(zip(query_ids, cos_scores_top_k_values, all_ranks_top_k_idx), desc="Formatting results..."):
            for score, corpus_id in zip(top_k_values, top_k_idx):
                if corpus_id != qid: # WARNING: We remove the query from results if it exists in corpus
                    corpus_idx = corpus[corpus_id.item()]["id"]
                    self.results[qid][corpus_idx] = score.item()

        # import joblib
        # joblib.dump(self.results, f"results.pkl")
        # ref_results = joblib.load(f"results.pkl")

        # compare results
        # sort self.results and ref_results
        # self.results = {k: {k2: v2 for k2, v2 in sorted(v.items(), key=lambda item: item[1], reverse=True)} for k, v in self.results.items()}
        # ref_results = {k: {k2: v2 for k2, v2 in sorted(v.items(), key=lambda item: item[1], reverse=True)} for k, v in ref_results.items()}
        # for qid in self.results:
        #     for corpus_idx in self.results[qid]:
        #         assert self.results[qid][corpus_idx] == ref_results[qid][corpus_idx]
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
                    cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(self.top_k, len(cos_scores[1])), dim=1, largest=True, sorted=False)
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
