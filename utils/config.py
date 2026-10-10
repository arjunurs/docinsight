"""
Runtime configuration read from environment variables (see .env.example).
"""
import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class AppConfig:
    """
    Settings shared by ingestion and querying.
    """

    chunk_size: int = 512
    chunk_overlap: int = 50
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    embedding_model: str = "text-embedding-3-small"
    persist_dir: str = "./data/chroma"
    collection_name: str = "document_collection"
    retrieval_mode: str = "hybrid"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Build a config from environment variables, falling back to the defaults above.
        """
        defaults = cls()
        return cls(
            chunk_size=_int_env("CHUNK_SIZE", defaults.chunk_size),
            chunk_overlap=_int_env("CHUNK_OVERLAP", defaults.chunk_overlap),
            llm_model=os.getenv("LLM_MODEL") or defaults.llm_model,
            llm_temperature=_float_env("LLM_TEMPERATURE", defaults.llm_temperature),
            embedding_model=os.getenv("EMBEDDING_MODEL") or defaults.embedding_model,
            persist_dir=os.getenv("CHROMA_PERSIST_DIR") or defaults.persist_dir,
            collection_name=os.getenv("CHROMA_COLLECTION") or defaults.collection_name,
            retrieval_mode=os.getenv("RETRIEVAL_MODE") or defaults.retrieval_mode,
        )
