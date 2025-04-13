import json
import logging
import os
from typing import List, Dict, Any
from app.domain.entities.embedding import Embedding
from app.domain.entities.chunk import Chunk
from app.domain.interfaces.i_vector_store import IVectorStore
from app.infrastructure.embeddings.embedding_processor import EmbeddingProcessor
from app.infrastructure.processors.text_document_processor import DocumentProcessor


class IngestorService:
    """
    Service for processing documents, generating embeddings, and ingesting them directly into the vector store,
    without the need for intermediate storage.
    """

    def __init__(self, vector_store: IVectorStore, batch_size: int = 100):
        """
        Initializes the ingestion service.

        Args:
            vector_store (IVectorStore): Instance of the vector store to use for ingestion.
            batch_size (int): Size of the batches for processing and ingestion.
        """
        self.vector_store = vector_store
        self.batch_size = batch_size
        self.embedding_processor = EmbeddingProcessor()
        self.document_processor = DocumentProcessor()

    def process_and_ingest_text(self, text: str) -> int:
        """
        Processes a simple text, generates chunks, creates embeddings, and ingests them directly
        into the vector store in a single flow.

        Args:
            text (str): Text to be processed.

        Returns:
            int: Total number of embeddings ingested.
        """
        logging.info("Iniciando processamento e ingestão direta de texto")

        chunks = self.document_processor.chunk_text(text)
        logging.info(f"Generated {len(chunks)} chunks from the text")

        total_ingested = 0
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            logging.info(
                f"processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)"
            )

            try:
                texts = [chunk.text for chunk in batch_chunks]
                metadatas = [
                    {
                        "chunk_id": chunk.id,
                        "source": "Darwin's Origin of Species",
                        "category": "CONTEÚDO_PRINCIPAL_DO_LIVRO",
                        "section_number": i,
                    }
                    for i, chunk in enumerate(batch_chunks)
                ]
                ids = [f"doc_{total_ingested + j}" for j in range(len(batch_chunks))]

                self.vector_store.add_texts_directly(texts, metadatas, ids)

                total_ingested += len(batch_chunks)
                logging.info(f"Batch {batch_num} processed and ingested successfully")

            except Exception as e:
                logging.error(f"Error processing batch {batch_num}: {str(e)}")

        logging.info(
            f"Succesfuly ingested. Total ingestion: {total_ingested}/{len(chunks)}"
        )
        return total_ingested


if __name__ == "__main__":
    import logging
    from app.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore

    # Configuração de logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def main():
        # Configuração do ChromaDB
        collection_name = "book_embeddings"
        persist_directory = "/home/luizg/projects/cadastra/rag-boticario/data/chroma_db"

        # Inicializa o ChromaDB com função de embedding
        # (necessário para converter textos em embeddings automaticamente)
        vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            use_embedding_function=True,
        )

        # Inicializa o serviço de ingestão direta
        ingestor_service = IngestorService(vector_store=vector_store)

        # Caminho para o arquivo JSON contendo os dados
        file_path = (
            "/home/luizg/projects/cadastra/rag-boticario/data/the Origin of Species.txt"
        )

        # Processa e ingere diretamente
        total_ingested = ingestor_service.process_and_ingest_text(file_path)

        logging.info(f"Total de embeddings ingeridos: {total_ingested}")

    main()
