from __future__ import annotations

import importlib.util
import logging

if importlib.util.find_spec("llm2vec") is not None:
    from llm2vec import LLM2Vec as LLM2VecOriginal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.autonotebook import trange

from .util import extract_corpus_sentences

logger = logging.getLogger(__name__)

POOLING_MODES = {
    "mean": "mean",
    "weighted_mean": "weighted_mean",
    "eos": "eos_token",
    "bos_token": "bos_token",
    "last_token": "last_token",
}


class LLM2Vec:
    def __init__(
        self,
        model_path: str | tuple = None,
        max_length: int = None,
        sep: str = " ",
        pooling: str = "mean",
        normalize: bool = True,
        prompts: dict[str, str] = None,
        peft_model_path: str = None,
        **kwargs,
    ):
        self.sep = sep
        self.normalize = normalize
        if pooling not in POOLING_MODES:
            raise ValueError(f"Pooling mode {pooling} not supported. Choose from {list(POOLING_MODES.keys())}")

        self.model = LLM2VecOriginal.from_pretrained(
            base_model_name_or_path=model_path,
            peft_model_name_or_path=peft_model_path,
            pooling_mode=POOLING_MODES[pooling],
            max_length=max_length,
            **kwargs,
        )

        if prompts:
            self.query_prefix = prompts.get("query", "")
            self.doc_prefix = prompts.get("passage", "")

    def _append_eos_token(self, texts, pad_to_multiple_of: int = 16):
        """Tokenizes the input texts and pads the tokenized input to the max_length with the eos token"""
        collated_texts = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.max_length - 1 if self.append_eos_token else self.max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        collated_texts["input_ids"] = [x + [self.tokenizer.eos_token_id] for x in collated_texts["input_ids"]]
        collated_texts = self.tokenizer.pad(
            collated_texts,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return collated_texts

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        query_embeddings = []

        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                sub_queries = [[self.query_prefix, query] for query in queries[start_idx : start_idx + batch_size]]
                query_embeddings += self.model.encode(sub_queries, batch_size=batch_size, show_progress_bar=False)

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
                if self.doc_prefix:
                    sub_sentences = [
                        [self.doc_prefix, sentence] for sentence in sentences[start_idx : start_idx + batch_size]
                    ]
                else:
                    sub_sentences = sentences[start_idx : start_idx + batch_size]
                corpus_embeddings += self.model.encode(sub_sentences, batch_size=batch_size, show_progress_bar=False)

            corpus_embeddings = torch.stack(corpus_embeddings)

            if self.normalize:
                corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=1)

            return corpus_embeddings
