from __future__ import annotations

import heapq
import importlib
import logging
import os

if importlib.util.find_spec("faiss") is not None:
    import faiss

from itertools import chain

import torch
from tqdm import tqdm

from .. import BaseSearch
from .faiss_index import FaissFlatSearcher
from .util import cos_sim, dot_score, pickle_load, save_embeddings

logger = logging.getLogger(__name__)


# DenseRetrievalExactSearch is parent class for any dense model that can be used for retrieval
# Abstract class is BaseSearch
class DenseRetrievalExactSearch(BaseSearch):
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int = 50000, **kwargs):
        # model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        self.score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_function_desc = {
            "cos_sim": "Cosine Similarity",
            "dot": "Dot Product",
        }
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True)
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        score_function: str,
        return_sorted: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError(
                f"score function: {score_function} must be either (cos_sim) for cosine similarity or (dot) for dot product"
            )

        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=self.convert_to_tensor,
        )

        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")
        logger.info(f"Scoring Function: {self.score_function_desc[score_function]} ({score_function})")

        itr = range(0, len(corpus), self.corpus_chunk_size)

        result_heaps = {qid: [] for qid in query_ids}  # Keep only the top-k docs for each query
        for batch_num, corpus_start_idx in enumerate(itr):
            logger.info(f"Encoding Batch {batch_num + 1}/{len(itr)}...")
            corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

            # Encode chunk of corpus
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus[corpus_start_idx:corpus_end_idx],
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor,
            )

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](query_embeddings, sub_corpus_embeddings)
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores,
                min(top_k + 1, len(cos_scores[1])),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_embeddings)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr], cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_ids[corpus_start_idx + sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            # Push item on the heap
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            # If item is larger than the smallest in the heap, push it on the heap then pop the smallest element
                            heapq.heappushpop(result_heaps[query_id], (score, corpus_id))

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results

    def encode(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        encode_output_path: str = "./embeddings/",
        overwrite: bool = False,
        query_filename: str = "queries.pkl",
        corpus_filename: str = "corpus.*.pkl",
        **kwargs,
    ):
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings_file = os.path.join(encode_output_path, query_filename)

        if not os.path.exists(query_embeddings_file) or overwrite:
            query_embeddings = self.model.encode_queries(
                queries,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor,
            )
            logger.info(f"Saving query embeddings to {encode_output_path}")
            save_embeddings(query_embeddings, query_ids, query_embeddings_file)
        else:
            logger.info(f"Query embeddings already exist at {query_embeddings_file}, skipping encoding.")

        logger.info("Sorting Corpus by document length (Longest first)...")

        corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )
        corpus = [corpus[cid] for cid in corpus_ids]

        logger.info("Encoding Corpus in batches... Warning: This might take a while!")

        itr = range(0, len(corpus), self.corpus_chunk_size)

        for batch_num, corpus_start_idx in enumerate(itr):
            batch_corpus_filename = corpus_filename.replace("*", str(batch_num))
            corpus_embeddings_file = os.path.join(encode_output_path, batch_corpus_filename)
            if not os.path.exists(corpus_embeddings_file) or overwrite:
                logger.info(f"Encoding Batch {batch_num + 1}/{len(itr)}...")
                corpus_end_idx = min(corpus_start_idx + self.corpus_chunk_size, len(corpus))

                # Encode chunk of corpus
                sub_corpus_embeddings = self.model.encode_corpus(
                    corpus[corpus_start_idx:corpus_end_idx],
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_tensor=self.convert_to_tensor,
                )

                # Save embeddings for the sub-corpus if encode_output_path is provided
                sub_corpus_ids = corpus_ids[corpus_start_idx:corpus_end_idx]
                corpus_embeddings_file = os.path.join(encode_output_path, f"corpus.{batch_num}.pkl")

                logger.info(f"Saving Corpus embeddings to {encode_output_path}")
                save_embeddings(
                    sub_corpus_embeddings,
                    sub_corpus_ids,
                    corpus_embeddings_file,
                )
            else:
                logger.info(f"Corpus embeddings already exist at {corpus_embeddings_file}, skipping encoding.")

    # Credits to tevatron repo & requires faiss-cpu to be installed!
    # https://github.com/texttron/tevatron/blob/main/src/tevatron/retriever/driver/search.py
    def search_from_files(
        self,
        query_embeddings_file: str,
        corpus_embeddings_files: str,
        top_k: int,
        score_function: str = "dot",
    ):
        """
        Search using precomputed embeddings.
        :param query_embeddings: Path to the query embeddings file.
        :param corpus_embeddings: Path to the corpus embeddings file.
        :param top_k: Number of top results to return.
        :return: Dictionary with query ids as keys and ranked corpus ids with scores as values.
        """
        logger.info("Loading precomputed embeddings...")
        corpus_embeddings_0, corpus_ids_0 = pickle_load(corpus_embeddings_files[0])
        faiss_flat = FaissFlatSearcher(corpus_embeddings_0)

        shards = chain([(corpus_embeddings_0, corpus_ids_0)], map(pickle_load, corpus_embeddings_files[1:]))
        if len(corpus_embeddings_files) > 1:
            shards = tqdm(shards, desc="Loading shards into index", total=len(corpus_embeddings_files))
        all_corpus_ids = []
        for corpus_embeddings, corpus_ids in shards:
            faiss_flat.add(corpus_embeddings)
            all_corpus_ids += corpus_ids

        query_embeddings, query_ids = pickle_load(query_embeddings_file)
        num_gpus = faiss.get_num_gpus()
        if num_gpus == 0:
            logger.info("No GPU found or using faiss-cpu. Back to CPU.")
        else:
            logger.info(f"Using {num_gpus} GPU")
            if num_gpus == 1:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                res = faiss.StandardGpuResources()
                faiss_flat.index = faiss.index_cpu_to_gpu(res, 0, faiss_flat.index, co)
            else:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                faiss_flat.index = faiss.index_cpu_to_all_gpus(faiss_flat.index, co, ngpu=num_gpus)

        scores, retrieved_query_ids = faiss_flat.batch_search(
            query_embeddings, top_k, batch_size=self.batch_size, quiet=not self.show_progress_bar
        )
        retrieved_corpus_ids = [[str(all_corpus_ids[x]) for x in q_dd] for q_dd in retrieved_query_ids]

        self.results = {}
        for qid, corpus_ids, score in zip(query_ids, retrieved_corpus_ids, scores):
            self.results[qid] = {corpus_id: score for corpus_id, score in zip(corpus_ids, score)}

        return self.results
