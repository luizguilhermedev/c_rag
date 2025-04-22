from typing import List, Dict, Any, Optional
import logging
import time
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_qdrant import QdrantVectorStore as LangchainQdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from app.settings import settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QDrantVectorStore:
    def __init__(self, url: str, collection_name: str, timeout: int = 300):
        """
        Inicializa o cliente Qdrant e configura o embeddings model.
        
        Args:
            url: URL do servidor Qdrant
            collection_name: Nome da coleção a ser usada
            timeout: Tempo limite para operações em segundos (padrão: 300s)
        """
        logger.info(f"Inicializando conexão com Qdrant em {url}")
        self.client = QdrantClient(
            url=url, 
            api_key=settings.QDRANT_API_KEY,
            timeout=timeout
        )
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.vector_store = None
        self.timeout = timeout
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def create_collection(self, vector_size: int = 1536) -> None:
        """
        Cria uma nova coleção no Qdrant com retry automático.
        
        Args:
            vector_size: Tamanho do vetor de embedding (1536 para OpenAI embeddings)
        """
        try:
            logger.info(f"Criando coleção: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Coleção {self.collection_name} criada com sucesso")
        except Exception as e:
            logger.error(f"Erro ao criar coleção: {e}")
            raise
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def collection_exists(self) -> bool:
        """
        Verifica se a coleção existe com retry automático.
        
        Returns:
            bool: True se a coleção existe
        """
        try:
            logger.info(f"Verificando se coleção {self.collection_name} existe")
            collections = self.client.get_collections().collections
            return any(collection.name == self.collection_name for collection in collections)
        except Exception as e:
            logger.error(f"Erro ao verificar existência da coleção: {e}")
            raise
            
    def _init_vector_store(self) -> None:
        """
        Inicializa o vector store da Langchain para operações de alto nível.
        """
        if self.vector_store is None:
            logger.info(f"Inicializando LangChain vector store para {self.collection_name}")
            self.vector_store = LangchainQdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )
    
    def add_documents_in_batches(self, documents: List[Document], batch_size: int = 10) -> List[str]:
        """
        Adiciona documentos à coleção em lotes para evitar timeouts.
        
        Args:
            documents: Lista de documentos para adicionar
            batch_size: Tamanho do lote para processamento
            
        Returns:
            List[str]: IDs dos documentos adicionados
        """
        self._init_vector_store()
        all_ids = []
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            end_idx = min(i + batch_size, total_docs)
            batch = documents[i:end_idx]
            
            logger.info(f"Processando lote {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}: "
                       f"documentos {i+1} até {end_idx} de {total_docs}")
            
            for attempt in range(3):
                try:
                    batch_ids = self.vector_store.add_documents(batch)
                    all_ids.extend(batch_ids)
                    logger.info(f"Lote {i//batch_size + 1} inserido com sucesso")
                    break
                except Exception as e:
                    logger.warning(f"Tentativa {attempt+1} falhou: {e}")
                    if attempt == 2:
                        logger.error(f"Falha ao processar lote {i//batch_size + 1}: {e}")
                        raise
                    time.sleep((attempt + 1) * 5)
        
        logger.info(f"Total de {len(all_ids)} documentos inseridos com sucesso")
        return all_ids
            
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Adiciona documentos à coleção.
        
        Args:
            documents: Lista de documentos para adicionar
            
        Returns:
            List[str]: IDs dos documentos adicionados
        """
        if len(documents) > 20:
            return self.add_documents_in_batches(documents)
            
        self._init_vector_store()
        try:
            logger.info(f"Adicionando {len(documents)} documentos à coleção {self.collection_name}")
            result = self.vector_store.add_documents(documents)
            logger.info(f"Documentos adicionados com sucesso. Total: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"Erro ao adicionar documentos: {e}")
            raise
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """
        Adiciona textos à coleção.
        
        Args:
            texts: Lista de textos para adicionar
            metadatas: Lista opcional de metadados associados aos textos
            
        Returns:
            List[str]: IDs dos textos adicionados
        """
        self._init_vector_store()
        return self.vector_store.add_texts(texts=texts, metadatas=metadatas)
        
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Realiza uma busca semântica por similaridade.
        
        Args:
            query: Texto de consulta
            k: Número de resultados a retornar
            
        Returns:
            List[Document]: Documentos mais similares à consulta
        """
        self._init_vector_store()
        return self.vector_store.similarity_search(query=query, k=k)
    
    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Deleta documentos pelos IDs.
        
        Args:
            ids: Lista de IDs de documentos a serem deletados
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=ids
            )
        )
        
    def delete_by_filter(self, filter_query: Dict[str, Any]) -> None:
        """
        Deleta documentos que correspondem a um filtro.
        
        Args:
            filter_query: Dicionário com os filtros para selecionar os documentos
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=k, match=models.MatchValue(value=v)
                        ) for k, v in filter_query.items()
                    ]
                )
            )
        )


if __name__ == "__main__":
    import os
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_community.document_loaders import TextLoader
    
    # Example usage
    try:
        # Configuração de alto timeout para operações longas
        qdrant_store = QDrantVectorStore(
            url="https://cca489f5-2d65-49bd-b23d-2db06125115d.europe-west3-0.gcp.cloud.qdrant.io:6333",
            collection_name="teste_geração_collection",
            timeout=600
        )
        
        # Verifica se a coleção existe
        collection_exists = qdrant_store.collection_exists()
        if not collection_exists:
            qdrant_store.create_collection()
            print("Coleção criada com sucesso!")
        else:
            print("Coleção já existe!")
        
        # Carrega e processa documentos
        semantic_chunker = SemanticChunker(
            embeddings=OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL, 
                openai_api_key=settings.OPENAI_API_KEY
            ),
            breakpoint_threshold_amount=0.7,
            breakpoint_threshold_type="percentile",
        )
        
        text_path = "data/clean_origin_complete.txt"
        print(f"Carregando arquivo: {text_path}")
        loader = TextLoader(text_path)
        documents = loader.load()
        print(f"Carregados {len(documents)} documentos")
        
        print("Dividindo documentos em chunks semânticos...")
        text_chunks = semantic_chunker.split_documents(documents)
        print(f"Gerados {len(text_chunks)} chunks para inserção")
        
        print(f"Inserindo chunks no Qdrant... (isso pode demorar)")
        # Usar o método que inclui batching
        qdrant_store.add_documents_in_batches(text_chunks, batch_size=5)
        print("Documentos adicionados com sucesso!")
        
    except Exception as e:
        import traceback
        print(f"Erro durante a execução: {e}")
        traceback.print_exc()