import logging
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from app.domain.entities.embedding import Embedding
from app.domain.interfaces.i_vector_store import IVectorStore
from app.settings import settings


class ChromaVectorStore(IVectorStore):
    """
    Implementação do banco vetorial usando ChromaDB via LangChain.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        use_embedding_function: bool = True,
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        logging.info(
            f"Inicializando ChromaDB com coleção '{collection_name}' em '{persist_directory}'"
        )

        # Inicializa a função de embedding
        self.embedding_function = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY
        )

        # Inicializa o ChromaDB
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory,
        )
        logging.info(f"ChromaDB inicializado com sucesso com função de embedding")

    def add_texts_directly(
        self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None
    ) -> None:
        """
        Adiciona textos diretamente ao ChromaDB, usando a função de embedding para gerar os embeddings.

        Args:
            texts (List[str]): Lista de textos a serem adicionados.
            metadatas (List[Dict], optional): Metadados associados a cada texto.
            ids (List[str], optional): IDs dos documentos.
        """
        logging.info(
            f"Adicionando {len(texts)} textos ao ChromaDB usando a função de embedding"
        )

        try:
            # Usa add_texts para adicionar textos e gerar embeddings automaticamente
            self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            logging.info(f"Textos adicionados com sucesso")
        except Exception as e:
            logging.error(f"Erro ao adicionar textos: {str(e)}")
            raise

    def direct_search(self, query: str, n_results: int = 5) -> List[Embedding]:
        """ """
        logging.info(
            f"Recuperando {n_results} embeddings semelhantes para a consulta: '{query}'"
        )

        results = self.vector_store.similarity_search(query, k=n_results)

        return results
