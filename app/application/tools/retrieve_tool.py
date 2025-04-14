from langchain_core.tools import tool
from app.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore
from app.settings import settings
from app.logs import get_logger

logger = get_logger(__name__)


@tool(response_format="content_and_artifact")
def retriever_tool(query: str):
    """Retrieve information related to a query."""
    logger.info(f"Retrieving information for query: '{query}'")

    vector_store = ChromaVectorStore(
        collection_name="the_origin_of_species",
        persist_directory=settings.VECTOR_STORE_PATH,
        use_embedding_function=True,
    )

    if hasattr(vector_store, "_collection"):
        count = vector_store._collection.count()
        logger.info(
            f"Vector store contains {count} documents in collection 'book_embeddings'"
        )

        if count == 0:
            logger.warning(
                "The vector store is empty! No documents available for retrieval."
            )
            return "No documents available in the knowledge base.", []

    retrieved_docs = vector_store.direct_search(query=query, n_results=6)
    logger.info(f"Retrieved {len(retrieved_docs)} documents")

    for i, doc in enumerate(retrieved_docs):
        logger.info(f"Document {i+1} content preview: {doc.page_content[:100]}...")
        logger.info(f"Document {i+1} metadata: {doc.metadata}")

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs
