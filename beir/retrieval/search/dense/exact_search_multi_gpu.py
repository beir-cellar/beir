from __future__ import annotations

import logging
import math
import os
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

from .. import BaseSearch
from .util import cos_sim, dot_score

logger = logging.getLogger(__name__)


@contextmanager
def main_rank_first(group: dist.ProcessGroup):
    is_main_rank = dist.get_rank(group) == 0
    if is_main_rank:
        yield

    dist.barrier(group)

    if not is_main_rank:
        yield


# parent class for any dense model that can be used for retrieval
# Abstract class is BaseSearch
class DenseRetrievalParallelExactSearch(BaseSearch):
    def __init__(self, model, batch_size: int = 128, corpus_chunk_size: int | None = None, **kwargs):
        # model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.encoding_batch_size = batch_size
        self.score_functions = {"cos_sim": cos_sim, "dot": dot_score}
        self.score_function_desc = {"cos_sim": "Cosine Similarity", "dot": "Dot Product"}
        self.corpus_chunk_size = batch_size * 100 if corpus_chunk_size is None else corpus_chunk_size
        self.show_progress_bar = kwargs.get("show_progress_bar", True) if int(os.getenv("RANK", 0)) == 0 else False
        self.convert_to_tensor = kwargs.get("convert_to_tensor", True)
        self.results = {}

        self.query_embeddings = {}
        self.top_k = None
        self.score_function = None

    @torch.no_grad()
    def search(
        self,
        corpus: Dataset,
        queries: Dataset,
        top_k: int,
        score_function: str,
        ignore_identical_ids: bool = True,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        # Create embeddings for all queries using model.encode_queries()
        # Runs semantic search against the corpus embeddings
        # Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError(
                f"score function: {score_function} must be either (cos_sim) for cosine similarity or (dot) for dot product"
            )
        logger.info(f"Scoring Function: {self.score_function_desc[score_function]} ({score_function})")

        world_size = int(os.getenv("WORLD_SIZE", 1))
        rank = int(os.getenv("RANK", 0))

        if ignore_identical_ids:
            with main_rank_first(dist.group.WORLD):
                # We remove the query from results if it exists in corpus
                logger.info("Ignoring identical ids between queries and corpus...")
                start_time = time.time()
                # batched filter must return list of bools
                queries_ids = queries["id"]
                corpus = corpus.filter(
                    lambda batch: [cid not in queries_ids for cid in batch["id"]],
                    batched=True,
                    num_proc=12,
                    load_from_cache_file=True,
                )
                logger.info(f"Ignore identical ids took {time.time() - start_time:.2f} seconds")

        # add index column
        corpus = corpus.add_column("mteb_index", range(len(corpus)))

        # Split corpus across devices to parallelize encoding
        assert isinstance(corpus, Dataset), f"Corpus must be a Dataset object, but got {type(corpus)}"
        local_corpus = split_dataset_by_node(corpus, rank=rank, world_size=world_size)
        logger.info(
            f"Splitted corpus into {world_size} chunks. Rank {int(os.getenv('RANK', 0))} has {len(local_corpus)} docs."
        )

        all_ranks_corpus_start_idx = local_corpus["mteb_index"][
            0
        ]  # I'm assuming `split_dataset_by_node` splits local_corpus evenly while keeping same order

        # Initiate dataloader
        queries_dl = DataLoader(queries, batch_size=self.corpus_chunk_size)
        corpus_dl = DataLoader(local_corpus, batch_size=self.corpus_chunk_size)

        # Encode queries (all ranks do this)
        logger.info(
            f"Encoding Queries ({len(queries)} queries) in {len(queries) // self.corpus_chunk_size + 1} batches... Warning: This might take a while!"
        )
        query_embeddings = []
        for step, queries_batch in tqdm(
            enumerate(queries_dl),
            total=len(queries) // self.corpus_chunk_size + 1,
            desc="Encode Queries",
            disable=rank != 0,
        ):
            q_embeds = self.model.encode_queries(
                queries_batch["text"],
                batch_size=self.encoding_batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor,
            )
            query_embeddings.append(q_embeds)
        query_embeddings = torch.cat(query_embeddings, dim=0)

        self.query_embeddings = query_embeddings
        self.top_k = top_k
        self.score_function = score_function

        # Encode corpus (each rank encodes `local_corpus`)
        logger.info(
            f"Encoding Corpus ({len(local_corpus)} docs) in {len(local_corpus) // self.corpus_chunk_size + 1} batches... Warning: This might take a while!"
        )
        start_time = time.time()
        query_ids = queries["id"]
        self.results = {qid: {} for qid in query_ids}
        all_chunks_cos_scores_top_k_values = []
        all_chunks_cos_scores_top_k_idx = []
        for chunk_id, corpus_batch in tqdm(
            enumerate(corpus_dl),
            total=len(local_corpus) // self.corpus_chunk_size + 1,
            desc="Encode Corpus",
            disable=rank != 0,
        ):
            corpus_start_idx = chunk_id * self.corpus_chunk_size

            # Encode chunk of local_corpus
            sub_corpus_embeddings = self.model.encode_corpus(
                corpus_batch,
                batch_size=self.encoding_batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=self.convert_to_tensor,
            )

            # Compute similarites using either cosine-similarity or dot product
            cos_scores = self.score_functions[score_function](
                query_embeddings, sub_corpus_embeddings
            )  # (num_queries, self.corpus_chunk_size)
            cos_scores[torch.isnan(cos_scores)] = -1

            # Get top-k values
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(
                cos_scores, min(top_k, cos_scores.shape[1]), dim=1, largest=True
            )  # (num_queries, top_k)

            # Displace index by corpus_start_idx
            cos_scores_top_k_idx += corpus_start_idx

            all_chunks_cos_scores_top_k_values.append(cos_scores_top_k_values)
            all_chunks_cos_scores_top_k_idx.append(cos_scores_top_k_idx)

        end_time = time.time()
        logger.info(f"Encoded all batches in {end_time - start_time:.2f} seconds")

        # TODO: have shapes (top_k, num_queries) instead?
        # keep only top_k top scoring docs from this rank
        all_chunks_cos_scores_top_k_values = torch.cat(
            all_chunks_cos_scores_top_k_values, dim=1
        )  # (num_queries, (top_k)*n_chunks)
        all_chunks_cos_scores_top_k_idx = torch.cat(
            all_chunks_cos_scores_top_k_idx, dim=1
        )  # (num_queries, (top_k)*n_chunks)
        cos_scores_top_k_values, temp_cos_scores_top_k_idx = torch.topk(
            all_chunks_cos_scores_top_k_values,
            min(top_k, all_chunks_cos_scores_top_k_values.shape[1]),
            dim=1,
            largest=True,
        )  # (num_queries, top_k)

        # `all_chunks_cos_scores_top_k_idx` should index `corpus_ids` and not `all_chunks_cos_scores_top_k_values`
        all_chunks_cos_scores_top_k_idx = torch.gather(
            all_chunks_cos_scores_top_k_idx, dim=1, index=temp_cos_scores_top_k_idx
        )  # (num_queries, top_k) // indexes between (0, len(local_corpus))

        # Displace index by all_ranks_corpus_start_idx so that we index `corpus` and not `local_corpus`
        all_chunks_cos_scores_top_k_idx += all_ranks_corpus_start_idx

        # If local_corpus doesn't have top_k samples we pad scores to `pad_gathered_tensor_to`
        if len(local_corpus) < top_k:
            pad_gathered_tensor_to = math.ceil(len(corpus) / world_size)
            cos_scores_top_k_values = torch.cat(
                [
                    cos_scores_top_k_values,
                    torch.ones(
                        (cos_scores_top_k_values.shape[0], pad_gathered_tensor_to - cos_scores_top_k_values.shape[1]),
                        device=cos_scores_top_k_values.device,
                    )
                    * -1,
                ],
                dim=1,
            )
            all_chunks_cos_scores_top_k_idx = torch.cat(
                [
                    all_chunks_cos_scores_top_k_idx,
                    torch.ones(
                        (
                            all_chunks_cos_scores_top_k_idx.shape[0],
                            pad_gathered_tensor_to - all_chunks_cos_scores_top_k_idx.shape[1],
                        ),
                        device=all_chunks_cos_scores_top_k_idx.device,
                        dtype=all_chunks_cos_scores_top_k_idx.dtype,
                    )
                    * -1,
                ],
                dim=1,
            )
        else:
            pad_gathered_tensor_to = top_k

        # all gather top_k results from all ranks
        n_queries = len(query_ids)
        all_ranks_top_k_values = torch.empty(
            (world_size, n_queries, pad_gathered_tensor_to), dtype=cos_scores_top_k_values.dtype, device="cuda"
        )
        all_ranks_top_k_idx = torch.empty(
            (world_size, n_queries, pad_gathered_tensor_to), dtype=all_chunks_cos_scores_top_k_idx.dtype, device="cuda"
        )
        dist.barrier()
        logger.info("All gather top_k values from all ranks...")
        dist.all_gather_into_tensor(all_ranks_top_k_values, cos_scores_top_k_values)
        logger.info("All gather top_k idx from all ranks...")
        dist.all_gather_into_tensor(all_ranks_top_k_idx, all_chunks_cos_scores_top_k_idx)
        logger.info("All gather ... Done!")

        all_ranks_top_k_values = all_ranks_top_k_values.permute(1, 0, 2).reshape(
            n_queries, -1
        )  # (n_queries, world_size*(pad_gathered_tensor_to))
        all_ranks_top_k_idx = all_ranks_top_k_idx.permute(1, 0, 2).reshape(
            n_queries, -1
        )  # (n_queries, world_size*(pad_gathered_tensor_to))

        # keep only top_k top scoring docs from all ranks
        cos_scores_top_k_values, temp_cos_scores_top_k_idx = torch.topk(
            all_ranks_top_k_values, min(top_k, all_ranks_top_k_values.shape[1]), dim=1, largest=True
        )  # (num_queries, top_k)

        # `all_ranks_top_k_idx` should index `corpus_ids` and not `all_ranks_top_k_values`
        all_ranks_top_k_idx = torch.gather(
            all_ranks_top_k_idx, dim=1, index=temp_cos_scores_top_k_idx
        )  # (num_queries, top_k) // indexes between (0, len(corpus))

        # fill in results
        cos_scores_top_k_values = cos_scores_top_k_values.tolist()
        all_ranks_top_k_idx = all_ranks_top_k_idx.tolist()
        corpus_i_to_idx = corpus["id"]
        for qid, top_k_values, top_k_idx in zip(query_ids, cos_scores_top_k_values, all_ranks_top_k_idx):
            for score, corpus_i in zip(top_k_values, top_k_idx):
                corpus_idx = corpus_i_to_idx[corpus_i]
                assert ignore_identical_ids is False or corpus_idx != qid, (
                    "Query id and corpus id should not be the same if ignore_identical_ids is set to True."
                )
                self.results[qid][corpus_idx] = score
        return self.results
