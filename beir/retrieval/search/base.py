from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSearch(ABC):
    @abstractmethod
    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        pass

    @abstractmethod
    def encode(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        encode_output_path: str = "./embeddings/",
        overwrite: bool = False,
        query_filename: str = "queries.pkl",
        corpus_filename: str = "corpus.*.pkl",
        **kwargs,
    ) -> None:
        pass

    @abstractmethod
    def search_from_files(
        self,
        query_embeddings_file: str,
        corpus_embeddings_files: list[str],
        top_k: int,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        pass
