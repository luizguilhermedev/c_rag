from abc import ABC, abstractmethod
from typing import List
from app.domain.entities.chunk import Chunk


class IDocumentProcessor(ABC):
    """Interface for document processing."""

    @abstractmethod
    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Performs chunking of a plain text.

        Args:
            text (str): Text to be split into chunks.

        Returns:
            List[Chunk]: List of generated chunks.
        """
        pass

    @abstractmethod
    def chunk_json(self, json_data: str) -> List[Chunk]:
        """
        Performs chunking of a JSON file.

        Args:
            json_data (str): Path to the JSON file to be processed.

        Returns:
            List[Chunk]: List of generated chunks.
        """
        pass
