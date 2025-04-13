from typing import List, Dict
from app.domain.interfaces.i_document_processor import IDocumentProcessor
from domain.entities.chunk import Chunk
from domain.entities.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from settings import settings


class DocumentProcessor(IDocumentProcessor):
    """
    Processador para realizar chunking em um JSON estruturado com seções específicas.
    """

    def __init__(self):
        self.semantic_chunker = SemanticChunker(
            embeddings=OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY
            ),
            breakpoint_threshold_amount=0.7,
            breakpoint_threshold_type="percentile",
        )

    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Realiza o chunking de um texto simples.

        Args:
            text: Texto a ser dividido em chunks.

        Returns:
            List[Chunk]: Lista de chunks gerados.
        """

        loader = TextLoader(text)
        documents = loader.load()
        document_texts = [doc.page_content for doc in documents]
        concatenated_text = " ".join(document_texts)

        text_chunks = self.semantic_chunker.split_text(concatenated_text)

        # Converte os chunks de texto para objetos Chunk
        chunks = [Chunk(text=text, metadata={}) for text in text_chunks]

        return chunks


# Script de teste
# if __name__ == "__main__":
#     processor = DocumentProcessor()
#     text = "data/the Origin of Species.txt"
#     chunks = processor.chunk_text(text)
#     for chunk in chunks:
#         print(chunk)
