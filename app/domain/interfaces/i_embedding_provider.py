from abc import ABC, abstractmethod
from typing import List
from domain.entities.chunk import Chunk
from domain.entities.embedding import Embedding


class IEmbeddingProcessor(ABC):
    """Interface para provedores de embeddings."""

    @abstractmethod
    def embed_chunk(self, chunk: Chunk) -> Embedding:
        """Gera embedding para um chunk de texto."""
        pass

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[Embedding]:
        """Gera embeddings para múltiplos chunks."""
        pass
