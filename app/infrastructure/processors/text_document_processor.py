from typing import List
from app.domain.interfaces.i_document_processor import IDocumentProcessor
from app.domain.entities.chunk import Chunk
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from app.settings import settings
import logging
import json


class DocumentProcessor(IDocumentProcessor):
    """
    Document processor that chunks text using a semantic chunker.
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
        self.json_loader = JSONLoader

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
            path_to_text (str): The path to the text document to be chunked.

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

    def chunk_json(self, json_path: str) -> List[Chunk]:
        """
        Chunks a JSON file by extracting specific fields and splitting their content into chunks.

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            List[Chunk]: List of Chunk objects containing the text chunks.
        """
        logging.info(f"Processing JSON: {json_path}")
        
        # Load the JSON file directly
        with open(json_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

        chunks = []

        # Process the book content
        if "book_content" in json_data:
            logging.info("Extracting book content...")
            book_text = self._clean_text(json_data["book_content"])
            logging.info(f"Book text size: {len(book_text)} characters")
            book_chunks = self.semantic_chunker.split_text(book_text)
            logging.info(f"Chunks generated from book: {len(book_chunks)}")
            for chunk_text in book_chunks:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={"source_type": "book_content", "source": json_path}
                ))

        # Process the summary content
        if "summary_content" in json_data:
            logging.info("Extracting summary content...")
            summary_text = self._clean_text(json_data["summary_content"])
            logging.info(f"Summary text size: {len(summary_text)} characters")
            summary_chunks = self.semantic_chunker.split_text(summary_text)
            logging.info(f"Chunks generated from summary: {len(summary_chunks)}")
            for chunk_text in summary_chunks:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={"source_type": "summary_content", "source": json_path}
                ))

        logging.info(f"Total chunks generated: {len(chunks)}")
        return chunks

    def chunk_text(self, text: str):
        """
        Chunks a plain text string.

        Args:
            text (str): The text to be chunked.

        Returns:
            List[Chunk]: List of generated chunks.
        """
        logging.info(f"Processing text to generate chunks. Text size: {len(text)} characters")
        chunks = []  # Your chunking logic here
        logging.info(f"Chunks generated: {len(chunks)}")
        return chunks


if __name__ == "__main__":
    processor = DocumentProcessor()
    text = "data/the Origin of Species.txt"
    chunks = processor.chunk_text(text)
    for chunk in chunks:
        print(chunk)
