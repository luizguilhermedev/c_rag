from abc import ABC, abstractmethod
from typing import List, Dict
from app.domain.entities.embedding import Embedding


class IVectorStore(ABC):
    """
    Interface for vector stores.
    """

    @abstractmethod
    def direct_search(self, query: str, n_results: int = 5) -> List[Embedding]:
        """
        Retrieves similar embeddings from the vector store.

        Args:
            query (str): Query text.
            n_results (int): Number of results to return.

        Returns:
            List[Embedding]: List of retrieved embeddings.
        """
        pass

    @abstractmethod
    def add_texts_directly(
        self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None
    ) -> None:
        """
        Adds texts directly to the vector store, using the embedding function to generate embeddings.

        Args:
            texts (List[str]): List of texts to be added.
            metadatas (List[Dict], optional): Metadata associated with each text.
            ids (List[str], optional): Document IDs.
        """
        pass
