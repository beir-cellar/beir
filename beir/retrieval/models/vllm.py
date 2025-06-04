from __future__ import annotations

import importlib.util
import logging

if importlib.util.find_spec("vllm") is not None:
    from vllm import LLM
    from vllm.config import PoolerConfig
    from vllm.inputs import token_inputs
    from vllm.lora.request import LoRARequest

import numpy as np
import torch
from torch import Tensor
from tqdm.autonotebook import trange
from transformers import AutoTokenizer

from .util import extract_corpus_sentences

logger = logging.getLogger(__name__)

TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

POOL_TYPES_VLLM = {
    "cls": "cls",
    "mean": "mean",
    "eos": "last",
    "last": "last",
}


class VLLMEmbed:
    def __init__(
        self,
        model_path: str | tuple,
        lora_name_or_path: str = None,
        lora_r: int = None,
        max_length: int = None,
        sep: str = " ",
        pooling: str = "eos",
        normalize: bool = False,
        prompts: dict[str, str] = None,
        append_eos_token: bool = False,
        torch_dtype: str = "bfloat16",
        cache_dir: str = None,
        convert_to_numpy: bool = False,
        **kwargs,
    ):
        self.sep = sep
        self.lora_name_or_path = lora_name_or_path
        self.convert_to_numpy = convert_to_numpy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.append_eos_token = append_eos_token
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=cache_dir if cache_dir is not None else None
        )
        # Set the tokenizer's pad token ID if it is not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        # Set the Pooling configuration in VLLM
        self.pooler_config = PoolerConfig(pooling_type=POOL_TYPES_VLLM[pooling].upper(), normalize=normalize)

        # Create an LLM.
        self.model = LLM(
            model=model_path,
            task="embed",
            enforce_eager=True,
            override_pooler_config=self.pooler_config,
            dtype=torch_dtype,
            enable_lora=True if self.lora_name_or_path else False,
            max_lora_rank=lora_r,
        )  # skip graph capturing for faster cold starts)
        self.torch_dtype = torch_dtype

        if prompts:
            self.query_prefix = prompts.get("query", "")
            self.doc_prefix = prompts.get("passage", "")

    def _append_eos_token(self, texts):
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
        if self.append_eos_token:
            collated_texts["input_ids"] = [x + [self.tokenizer.eos_token_id] for x in collated_texts["input_ids"]]
        return collated_texts

    @torch.no_grad()
    def encode(
        self, texts: list[str], batch_size: int = 16, prefix: str = "", show_progress_bar: bool = True, **kwargs
    ) -> list[Tensor] | np.ndarray | Tensor:
        embeddings, vllm_inputs = [], []

        for start_idx in trange(0, len(texts), batch_size, disable=not show_progress_bar):
            sub_texts = [prefix + text for text in texts[start_idx : start_idx + batch_size]]

            if self.append_eos_token:
                ctx_input = self._append_eos_token(sub_texts)
            else:
                ctx_input = self.tokenizer(sub_texts, truncation=True, padding=True, return_tensors="pt")

            # Move the input to the device
            vllm_inputs.extend([token_inputs(prompt_token_ids=token_ids) for token_ids in ctx_input["input_ids"]])

        logger.info(f"Encoding {len(vllm_inputs)} texts...")

        outputs = self.model.embed(
            vllm_inputs,
            lora_request=LoRARequest("emb_adapter", 1, self.lora_name_or_path) if self.lora_name_or_path else None,
        )
        for output in outputs:
            embeddings.append(output.outputs.embedding)

        if self.convert_to_numpy:
            embeddings = np.stack(embeddings, dtype=np.float16)

        else:
            # Convert the list of embeddings to a tensor
            embeddings = torch.as_tensor(embeddings, dtype=TORCH_DTYPES["float16"], device=self.device)

        return embeddings

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        return self.encode(queries, batch_size=batch_size, prefix=self.query_prefix, **kwargs)

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list] | list[str], batch_size: int = 8, **kwargs
    ) -> list[Tensor] | np.ndarray | Tensor:
        sentences = extract_corpus_sentences(corpus=corpus, sep=self.sep)
        return self.encode(sentences, batch_size=batch_size, prefix=self.doc_prefix, **kwargs)
