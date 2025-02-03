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
