from langchain_core.tools import tool
from app.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore

from app.settings import settings


@tool(response_format="content_and_artifact")
def retriever_tool(query: str):
    """Retrieve information related to a query."""
    vector_store = ChromaVectorStore(
        collection_name="book_embeddings",
        persist_directory="/home/luizg/projects/cadastra/rag-boticario/data/chroma_db",
        use_embedding_function=True,
    )
    retrieved_docs = vector_store.direct_search(query=query, n_results=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
