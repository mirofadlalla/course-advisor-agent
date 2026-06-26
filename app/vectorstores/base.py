from abc import ABC, abstractmethod

class BaseVectorStore(ABC):

    @abstractmethod
    def build_index(self):
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        pass