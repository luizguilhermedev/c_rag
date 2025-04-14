import logging
import os
from langchain_core.documents import Document
from app.logs import get_logger

from app.domain.interfaces.i_vector_store import IVectorStore
from app.infrastructure.processors.text_document_processor import DocumentProcessor


logger = get_logger(__name__)


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
                documents = [
                    Document(
                        page_content=chunk.text,
                        metadata={
                            "chunk_id": chunk.id,
                            "source": "Darwin's Origin of Species",
                            "category": "MAIN_BOOK_CONTENT",
                            "section_number": i + j,
                        },
                    )
                    for j, chunk in enumerate(batch_chunks)
                ]

                ids = [f"doc_{total_ingested + j}" for j in range(len(batch_chunks))]

                logging.info(f"Adding {len(documents)} documents to ChromaDB")

                self.vector_store.add_documents_directly(documents, ids)

                total_ingested += len(batch_chunks)
                logging.info(f"Batch {batch_num} processed and ingested successfully")

            except Exception as e:
                logging.error(f"Error processing batch {batch_num}: {str(e)}")

        logging.info(
            f"Successfully ingested. Total ingestion: {total_ingested}/{len(chunks)}"
        )
        return total_ingested

    def process_and_ingest_json(
        self,
        json_file_path: str,
        text_field_path: str = None,
        metadata_fields: dict = None,
    ) -> int:
        logging.info(f"Ingesting JSON: {json_file_path}")

        try:
            if not os.path.exists(json_file_path):
                logging.error(f"File not found: {json_file_path}")
                return 0

            processor = DocumentProcessor()
            chunks = processor.chunk_json(json_file_path)
            logging.info(f"Generated Chunks: {len(chunks)}")

            if chunks:
                logging.debug(f"Sample chunk text: {chunks[0].text[:100]}...")
                logging.debug(f"Sample chunk metadata: {chunks[0].metadata}")

            if not chunks:
                logging.warning("No chunks generated from the JSON.")
                return 0

            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
            total_ingested = 0

            for i in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[i : i + self.batch_size]
                batch_num = (i // self.batch_size) + 1

                logging.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)"
                )

                try:
                    texts = [chunk.text for chunk in batch_chunks]
                    metadatas = [chunk.metadata for chunk in batch_chunks]

                    logging.debug(
                        f"First text sample in batch {batch_num}: {texts[0][:50]}..."
                    )

                    if metadata_fields:
                        for metadata in metadatas:
                            for field_name, field_path in metadata_fields.items():
                                metadata[field_name] = field_path  # Simplified for now
                        logging.debug(f"Enriched metadata sample: {metadatas[0]}")

                    ids = [
                        f"doc_{total_ingested + j}" for j in range(len(batch_chunks))
                    ]

                    logging.info(
                        f"Adding batch {batch_num} to vector store with {len(texts)} texts"
                    )

                    self.vector_store.add_texts_directly(texts, metadatas, ids)

                    count = getattr(self.vector_store, "_collection", {}).count()
                    logging.info(f"Current document count in vector store: {count}")

                    total_ingested += len(batch_chunks)
                    logging.info(
                        f"Batch {batch_num} processed and ingested successfully"
                    )

                except Exception as e:
                    logging.error(f"Error processing batch {batch_num}: {str(e)}")
                    logging.error(f"Error details: {type(e).__name__}")
                    import traceback

                    logging.error(traceback.format_exc())

            if hasattr(self.vector_store, "_client") and hasattr(
                self.vector_store._client, "persist"
            ):
                self.vector_store._client.persist()
                logging.info("Vector store explicitly persisted to disk")

            try:
                final_count = getattr(self.vector_store, "_collection", {}).count()
                logging.info(f"Final document count in vector store: {final_count}")
            except Exception as e:
                logging.error(f"Error getting final count: {str(e)}")

            logging.info(f"Ingestion completed. Total: {total_ingested}")
            return total_ingested

        except Exception as e:
            logging.error(f"Error ingesting JSON: {e}")
            logging.error(traceback.format_exc())
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
        keys = path.split(".")
        value = json_data

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None


# if __name__ == "__main__":
#     import logging
#     from app.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore


#     def main():
#         collection_json = "the_origin_of_species"
#         persist_directory = "/home/luizg/projects/cadastra/clean-rag/data/vector_store"
#         vector_store = ChromaVectorStore(
#             collection_name=collection_json, persist_directory=persist_directory
#         )
#         ingestion_service = IngestorService(vector_store=vector_store)

#         ingestion_service.process_and_ingest_text(
#             "/home/luizg/projects/cadastra/clean-rag/data/the Origin of Species_book.txt"
#         )

#     main()
