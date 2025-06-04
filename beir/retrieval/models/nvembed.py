from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.autonotebook import trange
from transformers import AutoModel

from .util import extract_corpus_sentences

logger = logging.getLogger(__name__)


class NVEmbed:
    def __init__(
        self,
        model_path: str | tuple = None,
        max_length: int = None,
        sep: str = " ",
        normalize: bool = False,
        prompts: dict[str, str] = None,
        **kwargs,
    ):
        self.sep = sep
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, **kwargs)
        self.max_length = max_length if max_length else self.tokenizer.model_max_length
        self.normalize = normalize  # Normalize the embeddings

        if prompts:
            self.query_prefix = prompts.get("query", "")
            self.doc_prefix = prompts.get("passage", "")

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        query_embeddings = []

        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                sub_queries = [self.query_prefix + query for query in queries[start_idx : start_idx + batch_size]]
                query_embeddings += self.model.encode(
                    sub_queries, instruction=self.query_prefix, max_length=self.max_length
                )

        query_embeddings = torch.stack(query_embeddings)

        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        return query_embeddings

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list] | list[str], batch_size: int = 8, **kwargs
    ) -> list[Tensor] | np.ndarray | Tensor:
        corpus_embeddings = []
        sentences = extract_corpus_sentences(corpus=corpus, sep=self.sep)

        with torch.no_grad():
            for start_idx in trange(0, len(sentences), batch_size):
                sub_sentences = [
                    self.doc_prefix + sentence for sentence in sentences[start_idx : start_idx + batch_size]
                ]
                corpus_embeddings += self.model.encode(
                    sub_sentences, instruction=self.doc_prefix, max_length=self.max_length
                )

            corpus_embeddings = torch.stack(corpus_embeddings)

            if self.normalize:
                corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=1)

            return corpus_embeddings
