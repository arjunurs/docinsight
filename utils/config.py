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
    llm_model: str = "claude-haiku-5-5"
    llm_temperature: float = 0.0
    # "none" answers directly: no thinking on Claude Haiku 5.5 and GPT-6 Luna, like gpt-4o-mini
    llm_reasoning_effort: str = "none"
    embedding_model: str = "text-embedding-3-small"
    persist_dir: str = "./data/chroma"
    collection_name: str = "document_collection"
    retrieval_mode: str = "hybrid"
    reranker: str = ""
    rerank_candidates: int = 10
    hybrid_fusion: str = "score"
    hybrid_dense_weight: float = 0.7

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
            llm_reasoning_effort=os.getenv("LLM_REASONING_EFFORT") or defaults.llm_reasoning_effort,
            embedding_model=os.getenv("EMBEDDING_MODEL") or defaults.embedding_model,
            persist_dir=os.getenv("CHROMA_PERSIST_DIR") or defaults.persist_dir,
            collection_name=os.getenv("CHROMA_COLLECTION") or defaults.collection_name,
            retrieval_mode=os.getenv("RETRIEVAL_MODE") or defaults.retrieval_mode,
            reranker=os.getenv("RERANKER") or defaults.reranker,
            rerank_candidates=_int_env("RERANK_CANDIDATES", defaults.rerank_candidates),
            hybrid_fusion=os.getenv("HYBRID_FUSION") or defaults.hybrid_fusion,
            hybrid_dense_weight=_float_env("HYBRID_DENSE_WEIGHT", defaults.hybrid_dense_weight),
        )
