"""Unified application settings using pydantic-settings.

This module provides a single source of truth for all configuration parameters.
All modules should import settings from here instead of reading environment variables directly.
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings

_base_dir = Path(__file__).resolve().parent.parent
load_dotenv(_base_dir / ".env")


class SearchSettings(BaseSettings):
    """Search and retrieval configuration."""

    search_api_url: str
    search_api_timeout_seconds: float = Field(default=300.0)
    search_documents: int = Field(default=5)
    search_chunks: int = Field(default=64)
    search_alpha: float = Field(default=0.5)
    chunk_max_chars: int = Field(default=1200)
    chunk_overlap: int = Field(default=200)
    retrieval_in_force_only: bool = Field(default=True)
    retriever_enable_reranker: bool = Field(default=True)
    search_api_max_retries: int = Field(default=1)
    search_api_retry_backoff_seconds: float = Field(default=0.75)

    class Config:
        env_file = ".env"
        extra = "ignore"


class ChatSettings(BaseSettings):
    """Chat manager configuration."""

    openai_model: str = Field(
        default="gpt-5-mini",
        validation_alias=AliasChoices("OPENAI_CHAT_MODEL", "OPENAI_MODEL"),
    )
    max_retrieval_passes: int = Field(default=1, validation_alias="CHAT_MAX_RETRIEVAL_PASSES")
    max_retrieval_refines: int = Field(default=3, validation_alias="CHAT_MAX_RETRIEVAL_REFINES")
    max_mode_steps: int = Field(default=20, validation_alias="CHAT_MAX_MODE_STEPS")
    stream_idle_timeout_seconds: float = Field(default=3600.0, validation_alias="CHAT_STREAM_IDLE_TIMEOUT_SECONDS")
    stream_total_timeout_seconds: float = Field(default=3600.0, validation_alias="CHAT_STREAM_TOTAL_TIMEOUT_SECONDS")
    max_docs_per_type: int = Field(default=6, validation_alias="SEARCH_MAX_DOCS_PER_TYPE")
    max_chunks_per_doc: int = Field(default=6, validation_alias="SEARCH_MAX_CHUNKS_PER_DOC")
    max_user_docs: int = Field(default=5, validation_alias="CHAT_MAX_USER_DOCS")
    max_user_doc_tokens: int = Field(default=35000, validation_alias="CHAT_MAX_USER_DOC_TOKENS")
    max_user_doc_chunks: int = Field(default=30, validation_alias="CHAT_MAX_USER_DOC_CHUNKS")
    user_doc_chunk_max_chars: int = Field(default=1200, validation_alias="USER_DOC_CHUNK_MAX_CHARS")
    user_doc_stale_hours: int = Field(default=72, validation_alias="USER_DOC_STALE_HOURS")
    user_doc_stale_messages: int = Field(default=20, validation_alias="USER_DOC_STALE_MESSAGES")
    source_stale_messages: int = Field(default=6, validation_alias="CHAT_SOURCE_STALE_MESSAGES")
    base_url: str = Field(validation_alias="BASE_URL")
    context_max_tokens: int = Field(default=24000, validation_alias="CHAT_CONTEXT_MAX_TOKENS")
    max_stored_messages: int = Field(default=200, validation_alias="CHAT_MAX_STORED_MESSAGES")
    openai_max_retries: int = Field(default=1, validation_alias="OPENAI_MAX_RETRIES")
    openai_retry_backoff_seconds: float = Field(default=0.75, validation_alias="OPENAI_RETRY_BACKOFF_SECONDS")
    structured_output_max_retries: int = Field(default=1, validation_alias="OPENAI_STRUCTURED_OUTPUT_MAX_RETRIES")

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def search_api_url(self) -> str:
        return search_settings.search_api_url

    @property
    def search_chunks(self) -> int:
        return search_settings.search_chunks

    @property
    def search_alpha(self) -> float:
        return search_settings.search_alpha

    @property
    def search_timeout_seconds(self) -> float:
        return search_settings.search_api_timeout_seconds

    @property
    def search_max_retries(self) -> int:
        return search_settings.search_api_max_retries

    @property
    def search_retry_backoff_seconds(self) -> float:
        return search_settings.search_api_retry_backoff_seconds

    @property
    def retrieval_in_force_only(self) -> bool:
        return search_settings.retrieval_in_force_only

    @property
    def chunk_max_chars(self) -> int:
        return search_settings.chunk_max_chars


class EmbeddingSettings(BaseSettings):
    """Embedding service configuration."""

    embedding_api_url: str
    embedding_model: str = Field(default="Qwen/Qwen3-Embedding-4B")
    embedding_max_retries: int = Field(default=2)
    embedding_retry_backoff_seconds: float = Field(default=1.0)

    class Config:
        env_file = ".env"
        extra = "ignore"


class UploadSettings(BaseSettings):
    """Document upload and processing limits."""

    # Per-conversation limits
    max_docs_per_conversation: int = Field(default=5)
    max_pages_per_document: int = Field(default=50)
    max_total_pages_per_conversation: int = Field(default=500)

    # Message limits
    max_message_chars: int = Field(default=40000)

    # Queue backpressure limits
    max_global_pending_docs: int = Field(default=100)
    max_user_pending_docs: int = Field(default=10)

    # Document size limits (text extraction)
    max_extracted_chars: int = Field(default=500000)  # 500k chars, truncate beyond
    max_generated_markdown_chars: int = Field(default=500000)

    class Config:
        env_file = ".env"
        extra = "ignore"


class AppSettings(BaseSettings):
    """Global application settings."""

    language: str = Field(default="nb", validation_alias="APP_LANGUAGE")

    class Config:
        env_file = ".env"
        extra = "ignore"


search_settings = SearchSettings()
chat_settings = ChatSettings()
embedding_settings = EmbeddingSettings()
upload_settings = UploadSettings()
app_settings = AppSettings()
