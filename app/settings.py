from functools import cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    ENVIRONMENT: str = "dev"
    SERVICE_ACCOUNT_FILE: Optional[str] = None
    SERVICE_ACCOUNT_DATA: Optional[str] = None
    LOGGING_LEVEL: str = "DEBUG"
    LOGGING_FILE: Optional[str] = None

    # OpenAI Config
    OPENAI_API_KEY: Optional[str] = None

    # RAG Configurations
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o"

    # Chunking Configurations
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Vector Store Configurations
    VECTOR_STORE_TYPE: str = "in_memory"  # Opções: in_memory, chroma, faiss, etc.
    VECTOR_STORE_PATH: Optional[str] = (
        "./vector_store"  # Caminho para armazenamento persistente
    )

    CHAT_MODEL: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=[".env"], env_file_encoding="utf-8", extra="ignore"
    )


@cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
