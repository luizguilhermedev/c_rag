from abc import ABC, abstractmethod
from typing import List
from app.domain.entities.chunk import Chunk


class IDocumentProcessor(ABC):
    """Interface para processamento de documentos."""

    @abstractmethod
    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Realiza o chunking de um texto simples.

        Args:
            text (str): Texto a ser dividido em chunks.

        Returns:
            List[Chunk]: Lista de chunks gerados.
        """
        pass
