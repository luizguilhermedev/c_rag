from dataclasses import dataclass
from typing import Dict, Any
from uuid import uuid4


@dataclass
class Document:
    """
    Represents a document that can be processed into smaller chunks.
    It can represent a text file, PDF, or any other content source.
    """

    content: str
    source: str
    id: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid4())

        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def from_file(cls, file_path: str) -> "Document":
        """
        Create a Document instance from a text file.

        Args:
            file_path: Path to the file

        Returns:
            Document: An instance of Document with content from the file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            return cls(
                content=content, source=file_path, metadata={"file_path": file_path}
            )
        except Exception as e:
            raise IOError(f"Erro ao ler o arquivo {file_path}: {str(e)}")

    def get_summary(self, max_length: int = 100) -> str:
        """
        Returns a summary of the document (first few characters).

        Args:
            max_length: Maximum length of the summary

        Returns:
            str: Document summary
        """
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
