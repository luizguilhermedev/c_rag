from abc import abstractmethod
from typing import Any, Dict


class IngestorService:
    @abstractmethod
    def process_and_ingest(self, json_data: Dict[str, Any]) -> int:
        """
        Processes a JSON document, generates chunks, creates embeddings, and ingests them directly
        into the vector store in a single flow.
        Args:
            json_data (Dict[str, Any]): JSON document to be processed.
        Returns:
            int: Total number of embeddings ingested.
        """
        pass
