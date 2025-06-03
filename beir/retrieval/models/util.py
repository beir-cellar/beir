from __future__ import annotations

from collections.abc import Mapping

import torch


def extract_corpus_sentences(corpus: list[dict[str, str]] | dict[str, list] | list[str], sep: str) -> list[str]:
    """Extracts sentences from the corpus"""
    if isinstance(corpus, dict):
        sentences = [
            (corpus["title"][i] + sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][i].strip()
            for i in range(len(corpus["text"]))
        ]

    elif isinstance(corpus, list):
        if isinstance(corpus[0], str):  # if corpus is a list of strings
            sentences = corpus
        else:
            sentences = [
                (doc["title"] + sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus
            ]
    return sentences


# Taken from https://github.com/microsoft/unilm/blob/master/e5/utils.py#L24
def move_to_cuda(sample: dict | list | tuple | torch.Tensor) -> dict | list | tuple | torch.Tensor:
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple([_move_to_cuda(x) for x in maybe_tensor])
        elif isinstance(maybe_tensor, Mapping):
            return type(maybe_tensor)({k: _move_to_cuda(v) for k, v in maybe_tensor.items()})
        else:
            return maybe_tensor

    return _move_to_cuda(sample)
