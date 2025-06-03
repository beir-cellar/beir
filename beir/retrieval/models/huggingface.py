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
from .util import extract_corpus_sentences, move_to_cuda

logger = logging.getLogger(__name__)

POOL_FUNC = {"cls": cls_pooling, "mean": mean_pooling, "eos": eos_pooling}


def get_peft_model(peft_model_name: str, **kwargs) -> tuple[PeftModel, str]:
    config = PeftConfig.from_pretrained(peft_model_name)
    logger.info(f"Loading Auto Model {config.base_model_name_or_path} with kwargs: {kwargs}")
    base_model = AutoModel.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        **kwargs,
    )
    # Taken from tevatron repository: https://github.com/texttron/tevatron
    logger.info(f"Loading PEFT model from {peft_model_name}")
    model = PeftModel.from_pretrained(base_model, peft_model_name, use_cache=False)
    model = model.merge_and_unload()
    return model, config.base_model_name_or_path


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
        convert_to_numpy: bool = False,
        **kwargs,
    ):
        self.sep = sep
        if peft_model_path:
            self.model, base_model_path = get_peft_model(peft_model_path, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)

        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Padding side should be right for LLM-based models such as LLAMA-2 to avoid warnings
        if self.tokenizer.padding_side != "right":
            logging.info(f"Default padding side is {self.tokenizer.padding_side}, making padding side right.")
            self.tokenizer.padding_side = "right"

        self.max_length = max_length if max_length else self.tokenizer.model_max_length
        self.normalize = normalize  # Normalize the embeddings
        self.append_eos_token = append_eos_token  # Add eos token to the input
        self.convert_to_numpy = convert_to_numpy  # Convert the embeddings to numpy array

        # To enable model parallelism, by replicating the model on multiple GPUs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            logging.info(f"Using data-parallel GPUs: {self.num_gpus}")
            self.model = torch.nn.DataParallel(self.model)

        if self.device == "cuda":
            self.model.cuda()
        self.model.eval()

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

    @torch.no_grad()
    def encode(
        self, texts: list[str], batch_size: int = 16, prefix: str = "", show_progress_bar: bool = True, **kwargs
    ) -> list[Tensor] | np.ndarray | Tensor:
        if self.num_gpus > 1:
            batch_size *= self.num_gpus

        embeddings = []
        for start_idx in trange(0, len(texts), batch_size, disable=not show_progress_bar):
            sub_texts = [prefix + text for text in texts[start_idx : start_idx + batch_size]]
            if self.append_eos_token:
                input_texts = self._append_eos_token(sub_texts)
            else:
                input_texts = self.tokenizer(
                    sub_texts,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=self.max_length,  # max_length to trnucate the input
                )

            # Credit to https://github.com/microsoft/unilm/blob/master/e5/mteb_beir_eval.py
            with torch.amp.autocast(device_type=self.device):
                if self.device == "cuda":
                    input_texts = move_to_cuda(input_texts)
                output = self.model(**input_texts)
                embeddings += self.pooling_func(output, input_texts["attention_mask"])

        embeddings = torch.stack(embeddings)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        if self.convert_to_numpy:
            embeddings = embeddings.cpu().numpy()

        return embeddings

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        return self.encode(queries, batch_size=batch_size, prefix=self.query_prefix, **kwargs)

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list] | list[str], batch_size: int = 8, **kwargs
    ) -> list[Tensor] | np.ndarray | Tensor:
        sentences = extract_corpus_sentences(corpus=corpus, sep=self.sep)
        return self.encode(sentences, batch_size=batch_size, prefix=self.doc_prefix, **kwargs)
