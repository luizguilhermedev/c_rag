from typing import List, Dict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.domain.entities.embedding import Embedding
from app.domain.interfaces.i_vector_store import IVectorStore
from app.settings import settings
from app.logs import get_logger

logger = get_logger(__name__)


class ChromaVectorStore(IVectorStore):
    """
    Vector store implementation using ChromaDB via LangChain.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        use_embedding_function: bool = True,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        logger.info(
            f"Inicializando ChromaDB com coleção '{collection_name}' em '{persist_directory}'"
        )
        self.embedding_function = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY
        )
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )

    def add_texts_directly(
        self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None
    ) -> None:
        """
        Adds texts directly to ChromaDB using the embedding function to generate embeddings.

        Args:
            texts (List[str]): Text list to be added.
            metadatas (List[Dict], optional): Metadata for each text.
            ids (List[str], optional): Document IDs for each text.
        """

        try:
            self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logger.info("Added succesfully to ChromaDB.")
        except Exception as e:
            logger.error(f"Error Adding: {str(e)}")
            raise

    def add_documents_directly(self, documents, ids: List[str] = None) -> None:
        """
        Adds documents directly to ChromaDB.

        Args:
            documents (List[Embedding]): Documents to be added.
            ids (List[str], optional): Document IDs for each document.
        """
        try:
            self.vector_store.add_documents(documents=documents, ids=ids)
            logger.info("Added succesfully to ChromaDB.")
        except Exception as e:
            logger.error(f"Error Adding: {str(e)}")
            raise

    def direct_search(self, query: str, n_results: int = 5) -> List[Embedding]:
        results = self.vector_store.similarity_search(query, k=n_results)
        return results
