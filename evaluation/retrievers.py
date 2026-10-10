"""
Retrievers under evaluation. Each one ranks chunk ids for a batch of questions.

- bm25:   lexical baseline, runs offline with no API key.
- dense:  the app's stack (OpenAI text-embedding-3-small + ChromaDB).
- hybrid: reciprocal rank fusion of bm25 and dense.
"""
from __future__ import annotations

import re
import uuid
from typing import Dict, List, Protocol, Sequence

from evaluation.chunking import Chunk

# Small English stopword list; enough to stop BM25 from rewarding "the"/"of".
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it its "
    "of on or that the their this to was were what when where which who whom whose "
    "why with".split()
)
_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class Retriever(Protocol):
    name: str

    def retrieve_batch(self, questions: Sequence[str], k: int) -> List[List[str]]:
        """Return the top-k chunk ids for each question, best first."""
        ...


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: List[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunk_ids = [c.chunk_id for c in chunks]
        # Index the same string dense embeds, so both retrievers see identical input.
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in chunks])

    def retrieve_batch(self, questions: Sequence[str], k: int) -> List[List[str]]:
        import numpy as np

        results = []
        for question in questions:
            scores = self.bm25.get_scores(tokenize(question))
            top = np.argsort(-scores, kind="stable")[:k]
            results.append([self.chunk_ids[i] for i in top])
        return results


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


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int, rrf_k: int = 60) -> List[str]:
    """Fuse ranked lists with RRF (Cormack et al., SIGIR 2009): score = sum 1/(rrf_k + rank)."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: Sequence[Retriever], candidate_depth: int = 50):
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
