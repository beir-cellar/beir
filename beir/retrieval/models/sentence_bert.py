from __future__ import annotations

import logging

import numpy as np
import torch.multiprocessing as mp
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from torch import Tensor

logger = logging.getLogger(__name__)


class SentenceBERT:
    def __init__(
        self,
        model_path: str | tuple = None,
        sep: str = " ",
        prompts: dict[str, str] = None,
        **kwargs,
    ):
        self.sep = sep

        if isinstance(model_path, str):
            self.q_model = SentenceTransformer(model_path, kwargs)
            self.doc_model = self.q_model

        elif isinstance(model_path, tuple):
            self.q_model = SentenceTransformer(model_path[0], kwargs)
            self.doc_model = SentenceTransformer(model_path[1], kwargs)

        self.query_prefix = ""
        self.doc_prefix = ""

        # Checks if prompts are not set in Sentence Transformers but required during inference
        if prompts and (len(self.q_model.prompts) or len(self.doc_model.prompts) == 0):
            self.query_prefix = prompts["query"]
            self.doc_prefix = prompts["passage"]

    def get_similarity(self):
        return self.q_model.similarity

    def start_multi_process_pool(self, target_devices: list[str] = None) -> dict[str, object]:
        logger.info("Start multi-process pool on devices: {}".format(", ".join(map(str, target_devices))))

        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for process_id, device_name in enumerate(target_devices):
            p = ctx.Process(
                target=SentenceTransformer._encode_multi_process_worker,
                args=(
                    process_id,
                    device_name,
                    self.doc_model,
                    input_queue,
                    output_queue,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    def stop_multi_process_pool(self, pool: dict[str, object]):
        output_queue = pool["output"]
        [output_queue.get() for _ in range(len(pool["processes"]))]
        return self.doc_model.stop_multi_process_pool(pool)

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        return self.q_model.encode(
            [self.query_prefix + query for query in queries],
            batch_size=batch_size,
            **kwargs,
        )

    def encode_corpus(
        self,
        corpus: list[dict[str, str]] | dict[str, list] | list[str],
        batch_size: int = 8,
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor:
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        elif isinstance(corpus, list):
            if isinstance(corpus[0], str):  # if corpus is a list of strings
                sentences = corpus
            else:
                sentences = [
                    (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                    for doc in corpus
                ]
        return self.doc_model.encode(
            [self.doc_prefix + sentence for sentence in sentences],
            batch_size=batch_size,
            **kwargs,
        )

    ## Encoding corpus in parallel
    def encode_corpus_parallel(
        self,
        corpus: list[dict[str, str]] | Dataset,
        pool: dict[str, str],
        batch_size: int = 8,
        chunk_id: int = None,
        **kwargs,
    ):
        if isinstance(corpus, dict):
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            if isinstance(corpus[0], str):
                sentences = corpus
            else:
                sentences = [
                    (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                    for doc in corpus
                ]

        if chunk_id is not None and chunk_id >= len(pool["processes"]):
            output_queue = pool["output"]
            output_queue.get()

        input_queue = pool["input"]
        input_queue.put(
            [
                chunk_id,
                batch_size,
                [self.doc_prefix + sentence for sentence in sentences],
            ]
        )
