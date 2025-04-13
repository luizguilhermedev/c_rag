from abc import ABC, abstractmethod
from typing import List, Dict
from app.domain.entities.embedding import Embedding


class IVectorStore(ABC):
    """
    Interface para bancos vetoriais.
    """

    @abstractmethod
    def direct_search(self, query: str, n_results: int = 5) -> List[Embedding]:
        """
        Recupera embeddings semelhantes a partir do banco vetorial.

        Args:
            query (str): Texto de consulta.
            n_results (int): Número de resultados a serem retornados.

        Returns:
            List[Embedding]: Lista de embeddings recuperados.
        """
        pass

    @abstractmethod
    def add_texts_directly(
        self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None
    ) -> None:
        """
        Adiciona textos diretamente ao banco vetorial, usando a função de embedding para gerar os embeddings.

        Args:
            texts (List[str]): Lista de textos a serem adicionados.
            metadatas (List[Dict], optional): Metadados associados a cada texto.
            ids (List[str], optional): IDs dos documentos.
        """
        pass
