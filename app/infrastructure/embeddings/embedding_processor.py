from typing import List
from app.domain.entities.embedding import Embedding
from app.domain.interfaces.i_embedding_provider import IEmbeddingProcessor
from app.domain.entities.chunk import Chunk
from langchain_openai.embeddings import OpenAIEmbeddings
from app.settings import settings


class EmbeddingProcessor(IEmbeddingProcessor):
    """
    Provedor de embeddings usando OpenAI.
    """

    def __init__(self):
        self.embedder = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY
        )

    def embed_chunk(self, chunk: Chunk) -> Embedding:
        """
        Gera um embedding para um chunk de texto.

        Args:
            chunk (Chunk): O chunk a ser processado.

        Returns:
            Embedding: O embedding gerado.
        """
        return self.embedder.embed_query(chunk.text)

    def embed_chunks(self, chunks: List[Chunk]) -> List[Embedding]:
        """
        Gera embeddings para múltiplos chunks.

        Args:
            chunks (List[Chunk]): Lista de chunks a serem processados.

        Returns:
            List[Embedding]: Lista de embeddings gerados.
        """
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        return [
            Embedding(vector=vector, chunk_id=chunk.id)
            for vector, chunk in zip(vectors, chunks)
        ]


# if __name__ == "__main__":
#     # Teste rápido da classe
#     processor = EmbeddingProcessor()

#     test_chunk = Chunk(text="")
