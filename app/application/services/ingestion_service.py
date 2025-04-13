import json
import logging
import os
from app.domain.interfaces.i_vector_store import IVectorStore
from app.infrastructure.embeddings.embedding_processor import EmbeddingProcessor
from app.infrastructure.processors.text_document_processor import DocumentProcessor


class IngestorService:
    """
    Service for processing documents, generating embeddings, and ingesting them directly into the vector store,
    without the need for intermediate storage.
    """

    def __init__(self, vector_store: IVectorStore, batch_size: int = 500):
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
        logging.info("Starting text processing and direct ingestion")

        chunks = self.document_processor.chunk_text(text)
        logging.info(f"Generated {len(chunks)} chunks from the text")

        total_ingested = 0
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        logging.info(f"Total chunks: {len(chunks)}")
        logging.info(f"Batch size: {self.batch_size}")
        logging.info(f"Total batches: {total_batches}")

        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1

            logging.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)"
            )

            try:
                texts = [chunk.text for chunk in batch_chunks]
                metadatas = [
                    {
                        "chunk_id": chunk.id,
                        "source": "Darwin's Origin of Species",
                        "category": "MAIN_BOOK_CONTENT",
                        "section_number": i,
                    }
                    for i, chunk in enumerate(batch_chunks)
                ]
                ids = [f"doc_{total_ingested + j}" for j in range(len(batch_chunks))]

                logging.info(f"Adding {len(texts)} texts to ChromaDB")
                self.vector_store.add_texts_directly(texts, metadatas, ids)

                total_ingested += len(batch_chunks)
                logging.info(f"Batch {batch_num} processed and ingested successfully")

            except Exception as e:
                logging.error(f"Error processing batch {batch_num}: {str(e)}")

        logging.info(
            f"Successfully ingested. Total ingestion: {total_ingested}/{len(chunks)}"
        )
        return total_ingested
    
    def process_and_ingest_json(self, json_file_path: str, text_field_path: str = None, metadata_fields: dict = None) -> int:
        logging.info(f"Ingesting JSON: {json_file_path}")
        
        try:
            # Check if the file exists
            if not os.path.exists(json_file_path):
                logging.error(f"File not found: {json_file_path}")
                return 0

            # Process the JSON using the chunk_json method
            processor = DocumentProcessor()
            chunks = processor.chunk_json(json_file_path)
            logging.info(f"Generated Chunks: {len(chunks)}")

            if not chunks:
                logging.warning("No chunks generated from the JSON.")
                return 0

            # Calculate the total number of batches
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size  # Ceiling division

            # Ingest the chunks into the vector store in batches
            for i in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[i : i + self.batch_size]
                batch_num = (i // self.batch_size) + 1

                logging.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)"
                )

                try:
                    texts = [chunk.text for chunk in batch_chunks]
                    metadatas = [chunk.metadata for chunk in batch_chunks]
                    ids = [f"doc_{i + j}" for j in range(len(batch_chunks))]

                    # Add the batch to the vector store
                    self.vector_store.add_texts_directly(texts, metadatas, ids)

                    logging.info(f"Batch {batch_num} processed and ingested successfully")

                except Exception as e:
                    logging.error(f"Error processing batch {batch_num}: {str(e)}")

            logging.info(f"Ingestion completed. Total: {len(chunks)}")
            return len(chunks)

        except Exception as e:
            logging.error(f"Error ingesting JSON: {e}")
            return 0

    def _extract_value_from_path(self, json_data, path):
        """
        Extracts a value from a JSON using dot notation.
        Example: "metadata.author" would extract json_data["metadata"]["author"]
        
        Args:
            json_data (dict): JSON data
            path (str): Path using dot notation
            
        Returns:
            any: Extracted value or None if the path does not exist
        """
        keys = path.split('.')
        value = json_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None


if __name__ == "__main__":
    import logging
    from app.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore

    # Logging configuration
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def main():
        # ChromaDB configuration
        collection_json = "json_embeddings"
        persist_directory_json = "/home/luizg/projects/cadastra/rag-boticario/data/chroma_db_json"

        logging.info(f"Initializing ChromaDB with collection '{collection_json}' at '{persist_directory_json}'")

        if not os.path.exists(persist_directory_json):
            logging.error(f"ChromaDB directory not created: {persist_directory_json}")
        else:
            logging.info(f"ChromaDB directory exists: {persist_directory_json}")

        # Initialize ChromaDB with embedding function
        vector_store_json = ChromaVectorStore(
            collection_name=collection_json,
            persist_directory=persist_directory_json,
            use_embedding_function=True,
        )

        # Initialize the ingestion service
        ingestor_service_json = IngestorService(vector_store=vector_store_json)

        # Path to the JSON file containing the data
        main_content_json_ingestion = ingestor_service_json.process_and_ingest_json(
            json_file_path="/home/luizg/projects/cadastra/clean-rag/processed_documents/the Origin of Species.json",
            text_field_path="book_content",
            metadata_fields={
                "author": "metadata.author",
                "title": "metadata.title"
            }
        )

        summary_content_json_ingestion = ingestor_service_json.process_and_ingest_json(
            json_file_path="/home/luizg/projects/cadastra/clean-rag/processed_documents/the Origin of Species.json",
            text_field_path="summary_content",
            metadata_fields={
                "author": "metadata.author",
                "title": "metadata.title"
            }
        )
        logging.info(
            f"Total embeddings ingested from main content: {main_content_json_ingestion}"
        )
        logging.info(
            f"Total embeddings ingested from summary content: {summary_content_json_ingestion}"
        )

    main()
