from __future__ import annotations

import importlib.util
import logging

if importlib.util.find_spec("peft") is not None:
    from peft import PeftConfig, PeftModel

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.autonotebook import trange
from transformers import AutoModel, AutoTokenizer

from .pooling import cls_pooling, eos_pooling, mean_pooling

logger = logging.getLogger(__name__)

POOL_FUNC = {"cls": cls_pooling, "mean": mean_pooling, "eos": eos_pooling}


def get_peft_model(peft_model_name: str) -> PeftModel:
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    return model


class HuggingFace:
    def __init__(
        self,
        model_path: str | tuple = None,
        max_length: int = None,
        sep: str = " ",
        pooling: str = "mean",
        normalize: bool = False,
        prompts: dict[str, str] = None,
        append_eos_token: bool = False,
        peft_model_path: str = None,
        **kwargs,
    ):
        self.sep = sep
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        if peft_model_path:
            self.model = get_peft_model(peft_model_path)
        else:
            self.model = AutoModel.from_pretrained(
                model_path, device_map="auto", torch_dtype=kwargs.get("torch_dtype", "auto"), trust_remote_code=True
            )
        self.model.eval()
        self.max_length = max_length if max_length else self.tokenizer.model_max_length
        self.normalize = normalize  # Normalize the embeddings
        self.append_eos_token = append_eos_token  # Add eos token to the input

        if pooling not in ["cls", "mean", "eos"]:
            raise ValueError("Supported Pooling techniques should be either 'cls', 'mean' or 'eos'")
        self.pooling_func = POOL_FUNC[pooling]

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
                sub_queries = [self.query_prefix + query for query in queries[start_idx : start_idx + batch_size]]
                if self.append_eos_token:
                    query_input = self._append_eos_token(sub_queries)
                else:
                    query_input = self.tokenizer(sub_queries, truncation=True, padding=True, return_tensors="pt")

                # Move the input to the device
                query_input = query_input.to(self.model.device)
                query_output = self.model(**query_input)
                query_embeddings += self.pooling_func(query_output, query_input["attention_mask"])

        query_embeddings = torch.stack(query_embeddings)

        if self.normalize:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        return query_embeddings

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list] | list[str], batch_size: int = 8, **kwargs
    ) -> list[Tensor] | np.ndarray | Tensor:
        corpus_embeddings = []

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

        with torch.no_grad():
            for start_idx in trange(0, len(sentences), batch_size):
                sub_sentences = [
                    self.doc_prefix + sentence for sentence in sentences[start_idx : start_idx + batch_size]
                ]
                if self.append_eos_token:
                    ctx_input = self._append_eos_token(sub_sentences)
                else:
                    ctx_input = self.tokenizer(sub_sentences, truncation=True, padding=True, return_tensors="pt")

                # Move the input to the device
                ctx_input = ctx_input.to(self.model.device)
                ctx_output = self.model(**ctx_input)
                corpus_embeddings += self.pooling_func(ctx_output, ctx_input["attention_mask"])

            corpus_embeddings = torch.stack(corpus_embeddings)

            if self.normalize:
                corpus_embeddings = F.normalize(corpus_embeddings, p=2, dim=1)

            return corpus_embeddings
