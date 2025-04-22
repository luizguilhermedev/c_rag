import requests

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate

from app.infrastructure.vector_store.chroma_vector_store import ChromaVectorStore
from app.infrastructure.vector_store.vector_store import QDrantVectorStore
from app.settings import settings
from app.logs import get_logger
from app.domain.prompts.rewrite_prompt import rewrite_question_template
from app.domain.prompts.ground_truth_prompt import ground_truth_prompt
from app.infrastructure.initialize_llm import initialize_llm

logger = get_logger(__name__)


@tool()
def retriever_tool(query: str):
    """Retrieve information related to a query."""
    logger.info(f"Retrieving information for query: '{query}'")

    vector_store = QDrantVectorStore(
        url=settings.QDRANT_URL,
        collection_name="teste_geração_collection",
        timeout=settings.QDRANT_TIMEOUT,
    )

    retrieved_docs = vector_store.similarity_search(query=query, k=4)
    logger.info(f"Retrieved {len(retrieved_docs)} documents")

    for i, doc in enumerate(retrieved_docs):
        logger.info(f"Document {i+1} content preview: {doc.page_content}")
        logger.info(f"Document {i+1} metadata: {doc.metadata}")

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs



@tool
def book_info_tool(query: str) -> str:
    """Fetch book metadata or summary using an external API like Open Library."""
    url = f"https://openlibrary.org/search.json?q={query}"
    response = requests.get(url)
    if response.status_code != 200:
        return "Could not fetch information at the moment."
    
    data = response.json()
    if not data.get("docs"):
        return "No relevant book information found."
    
    doc = data["docs"][0]
    title = doc.get("title", "Unknown title")
    author = doc.get("author_name", ["Unknown author"])[0]
    year = doc.get("first_publish_year", "Unknown year")

    return f"Title: {title}\nAuthor: {author}\nYear: {year}"

@tool
def query_rewrite_tool(query: str) -> str:
    """Rewrite the query to be more specific or clear."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", rewrite_question_template),
        ("user", f"Rewrite this query: {query}")
    ])
    
    llm = initialize_llm()
    response = llm.invoke(prompt.format_messages())
    
    return response.content

@tool
def grounded_or_not_tool(query: str, answer: str):
    """This function is inteded to check if the answer is grounded or not."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", ground_truth_prompt),
        ("user", f"Query: {query}\nAnswer: {answer}")
    ])
    llm = initialize_llm()
    response = llm.invoke(prompt.format_messages())

    return response.content

# if __name__ == "__main__":
#     # Example usage
#     query = "What is the origin of species?"
#     print(retriever_tool(query))
#     print(book_info_tool(query))
#     print(query_rewrite_tool(query))


