from functools import cache
from typing import Optional
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    ENVIRONMENT: str = "dev"
    SERVICE_ACCOUNT_FILE: Optional[str] = None
    SERVICE_ACCOUNT_DATA: Optional[str] = None
    LOGGING_LEVEL: str = "DEBUG"
    LOGGING_FILE: Optional[str] = None

    OPENAI_API_KEY: Optional[str] = None

    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    VECTOR_STORE_TYPE: str = "in_memory"
    VECTOR_STORE_PATH: str = os.path.join(BASE_DIR, "data/vector_store")

    CHAT_MODEL: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=[".env"], env_file_encoding="utf-8", extra="ignore"
    )


@cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
