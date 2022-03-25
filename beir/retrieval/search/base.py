from abc import ABC, abstractmethod
from typing import Dict


class RetrievalModel(ABC):
    @abstractmethod
    def search(self, 
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str],
        top_k: int,
        **kwargs) -> Dict[str, Dict[str, float]]:
        pass
