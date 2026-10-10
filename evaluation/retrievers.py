"""
Retrievers under evaluation. Each one ranks chunk ids for a batch of questions.

- bm25:   lexical baseline, runs offline with no API key.
- dense:  the app's stack (OpenAI text-embedding-3-small + ChromaDB).
- hybrid: reciprocal rank fusion of bm25 and dense.

Tokenization, BM25 input text and RRF come from utils.retrieval, which the
app's hybrid QueryEngine uses too, so these rankings match the app's.
"""
from __future__ import annotations

import uuid
from typing import List, Protocol, Sequence

from evaluation.chunking import Chunk
from utils.retrieval import CANDIDATE_DEPTH, BM25Index, reciprocal_rank_fusion

__all__ = ["BM25Retriever", "DenseRetriever", "HybridRetriever", "reciprocal_rank_fusion"]


class Retriever(Protocol):
    name: str

    def retrieve_batch(self, questions: Sequence[str], k: int) -> List[List[str]]:
        """Return the top-k chunk ids for each question, best first."""
        ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: List[Chunk]):
        # Indexes the same string dense embeds, so both retrievers see identical input.
        self.index = BM25Index([c.node for c in chunks], ids=[c.chunk_id for c in chunks])

    def retrieve_batch(self, questions: Sequence[str], k: int) -> List[List[str]]:
        return [self.index.rank(question, k) for question in questions]


class DenseRetriever:
    """
    Embeds chunks with the app's embedding model into an in-memory Chroma
    collection, the same vector store the app persists to disk.
    """

    name = "dense"

    def __init__(
        self,
        chunks: List[Chunk],
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 256,
        embed_model=None,
    ):
        import chromadb
        import chromadb.config
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.core.schema import TextNode
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.vector_stores.chroma import ChromaVectorStore

        # embed_model overrides the OpenAI model (tests use an offline fake).
        self.embed_model = embed_model or OpenAIEmbedding(
            model=embedding_model, embed_batch_size=batch_size
        )
        self.batch_size = batch_size

        # Keep the app's metadata and exclusions so the embedded text matches the app's.
        nodes = [
            TextNode(
                id_=c.chunk_id,
                text=c.text,
                metadata=dict(c.node.metadata),
                excluded_embed_metadata_keys=list(c.node.excluded_embed_metadata_keys),
                excluded_llm_metadata_keys=list(c.node.excluded_llm_metadata_keys),
            )
            for c in chunks
        ]
        client = chromadb.EphemeralClient(settings=chromadb.config.Settings(anonymized_telemetry=False))
        self.collection = client.create_collection(f"eval-{uuid.uuid4().hex[:8]}")
        storage_context = StorageContext.from_defaults(
            vector_store=ChromaVectorStore(chroma_collection=self.collection)
        )
        self.index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )

    def retrieve_batch(self, questions: Sequence[str], k: int) -> List[List[str]]:
        results: List[List[str]] = []
        for i in range(0, len(questions), self.batch_size):
            batch = list(questions[i : i + self.batch_size])
            embeddings = self.embed_model.get_text_embedding_batch(batch)
            response = self.collection.query(query_embeddings=embeddings, n_results=k)
            results.extend(response["ids"])
        return results


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], candidate_depth: int = CANDIDATE_DEPTH):
        self.retrievers = retrievers
        self.candidate_depth = candidate_depth

    def retrieve_batch(self, questions: Sequence[str], k: int) -> List[List[str]]:
        per_retriever = [
            r.retrieve_batch(questions, max(k, self.candidate_depth)) for r in self.retrievers
        ]
        return [
            reciprocal_rank_fusion([rankings[i] for rankings in per_retriever], k)
            for i in range(len(questions))
        ]
