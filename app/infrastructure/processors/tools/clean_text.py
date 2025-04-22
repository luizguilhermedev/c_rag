import re
from typing import Any, Dict
from langchain_core.tools import tool
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, JSONLoader
from app.domain.entities.chunk import Chunk
from app.settings import settings


from langchain.tools import tool
import re

@tool
def clean_text_(raw_text: str) -> str:
    """
    Cleans text by removing unnecessary spaces, newlines, and headers.
    """
    pass

@tool
def preprocess_text(text: str):
    """
    Function used to chunk text using SemanticChunker.
    It uses the TextLoader from langchain to load the text and SemanticChunker to split it into chunks.
    Args:
        text (str): The Path of the text document to be chunked.
    """

    semantic_chunker = SemanticChunker(
            embeddings=OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY
            ),
            breakpoint_threshold_amount=0.7,
            breakpoint_threshold_type="percentile",
        )
    loader = TextLoader
    loader = loader(text)
    documents = loader.load()
    text_chunks = semantic_chunker.split_text(documents)

    chunks = [Chunk(text=text, metadata={}) for text in text_chunks]

    #save chunks to a file
    with open("chunks.json", "w") as f:
        for chunk in chunks:
            f.write(chunk.text + "\n")

    return chunks


@tool
def chunk_json(json: str):
    """
    Function used to chunk json using SemanticChunker."""
    loader = JSONLoader
    loaded_json = loader(json).load()
    semantic_chunker = SemanticChunker(
            embeddings=OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL, openai_api_key=settings.OPENAI_API_KEY
            ),
            breakpoint_threshold_amount=0.7,
            breakpoint_threshold_type="percentile",
        )
    chunking = semantic_chunker.split_text(loaded_json)
    chunks = [Chunk(text=text, metadata={}) for text in chunking]

    with open("chunks.json", "w") as f:
        for chunk in chunks:
            f.write(chunk.text + "\n")
    return chunks



    
