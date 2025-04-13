import logging
from app.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore


class Retriever(ChromaVectorStore):
    def __init__(self, collection_name: str, persist_directory: str):
        super().__init__(collection_name, persist_directory)
        logging.info(
            f"Inicializando o retriever com coleção '{collection_name}' em '{persist_directory}'"
        )

    def as_retriever(self, n_results=5):
        return super().direct_search(n_results=n_results)
