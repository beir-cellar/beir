from __future__ import annotations

import importlib.util
import logging

if importlib.util.find_spec("voyageai") is not None:
    import voyageai

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.autonotebook import trange

from .util import extract_corpus_sentences

logger = logging.getLogger(__name__)

TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class VoyageAPI:
    def __init__(
        self,
        model_path: str | tuple = None,
        api_key: str = None,
        sep: str = " ",
        torch_dtype: str = "float32",
        prompts: dict[str, str] = None,
        normalize: bool = False,
        convert_to_numpy: bool = False,
        **kwargs,
    ):
        self.sep = sep
        self.model_path = model_path
        self.normalize = normalize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vo = voyageai.Client()
        self.torch_dtype = torch_dtype
        self.convert_to_numpy = convert_to_numpy
        self.query_prefix, self.doc_prefix = "", ""
        if prompts:
            self.query_prefix = prompts.get("query", "")
            self.doc_prefix = prompts.get("passage", "")

    def encode(
        self,
        texts: list[str],
        input_type: Literal["query", "document"],
        batch_size: int = 16,
        prefix: str = "",
        **kwargs,
    ) -> list[Tensor] | np.ndarray | Tensor:
        embeddings = []

        for start_idx in trange(0, len(texts), batch_size):
            sub_texts = [prefix + text for text in texts[start_idx : start_idx + batch_size]]
            try:
                response = self.vo.embed(
                    texts=sub_texts, model=self.model_path, input_type=input_type, output_type="float"
                )
                embeddings += response.embeddings
            except Exception as e:
                logger.error(f"Error while encoding texts: {e}")

        embeddings = torch.as_tensor(embeddings, dtype=TORCH_DTYPES[self.torch_dtype], device=self.device)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        if self.convert_to_numpy:
            embeddings = embeddings.cpu().numpy()

        return embeddings

    def encode_queries(self, queries: list[str], batch_size: int = 16, **kwargs) -> list[Tensor] | np.ndarray | Tensor:
        return self.encode(queries, input_type="query", batch_size=batch_size, prefix=self.query_prefix, **kwargs)

    def encode_corpus(
        self, corpus: list[dict[str, str]] | dict[str, list] | list[str], batch_size: int = 8, **kwargs
    ) -> list[Tensor] | np.ndarray | Tensor:
        sentences = extract_corpus_sentences(corpus=corpus, sep=self.sep)
        return self.encode(sentences, input_type="document", batch_size=batch_size, prefix=self.doc_prefix, **kwargs)
