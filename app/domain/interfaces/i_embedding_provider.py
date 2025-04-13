from abc import ABC, abstractmethod
from typing import List
from app.domain.entities.chunk import Chunk
from app.domain.entities.embedding import Embedding


class IEmbeddingProcessor(ABC):
    """Interface for embedding providers."""

    @abstractmethod
    def embed_chunk(self, chunk: Chunk) -> Embedding:
        """Generates an embedding for a text chunk."""
        pass

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[Embedding]:
        """Generates embeddings for multiple chunks."""
        pass
