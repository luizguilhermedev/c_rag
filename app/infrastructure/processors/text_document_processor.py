from typing import List
from app.domain.interfaces.i_document_processor import IDocumentProcessor
from app.domain.entities.chunk import Chunk
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from app.settings import settings


class DocumentProcessor(IDocumentProcessor):
    """
    Documnet processor which chunks text using a semantic chunker.
    """

    def __init__(self):
        self.semantic_chunker = SemanticChunker(
            embeddings=OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY
            ),
            breakpoint_threshold_amount=0.7,
            breakpoint_threshold_type="percentile",
        )
        self.text_loader = TextLoader

    def _clean_text(self, text: str) -> str:
        """
        Normalize text by removing extra spaces and newlines.
        """
        return " ".join(text.split())

    def chunk_text(self, path_to_text: str) -> List[Chunk]:
        """
        Chunks a simple text document.
        It uses the TextLoader from langchain to load the text and SemanticChunker to split it into chunks.
        Args:
            text: The path to the text document to be chunked.

        Returns:
            List[Chunk]: List of Chunk objects containing the text chunks.
        """

        loader = self.text_loader(path_to_text)
        documents = loader.load()
        _document_texts = [self._clean_text(doc.page_content) for doc in documents]
        concatenated_text = " ".join(_document_texts)

        text_chunks = self.semantic_chunker.split_text(concatenated_text)

        chunks = [Chunk(text=text, metadata={}) for text in text_chunks]

        return chunks


if __name__ == "__main__":
    processor = DocumentProcessor()
    text = "data/the Origin of Species.txt"
    chunks = processor.chunk_text(text)
    for chunk in chunks:
        print(chunk)
