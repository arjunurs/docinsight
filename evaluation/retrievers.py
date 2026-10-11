"""
Retrievers under evaluation. Each one ranks chunk ids for a batch of questions.

- bm25:   lexical baseline, runs offline with no API key.
- dense:  the app's stack (OpenAI text-embedding-3-small + ChromaDB).
- hybrid: reciprocal rank fusion of bm25 and dense.
- rerank: hybrid's top candidates reordered by a cross-encoder (optional extra).

Tokenization, BM25 input text and RRF come from utils.retrieval, which the
app's hybrid QueryEngine uses too, so these rankings match the app's.
"""
from __future__ import annotations

import uuid
from typing import Iterator, List, Protocol, Sequence, Tuple

from evaluation.chunking import Chunk
from utils.retrieval import (
    CANDIDATE_DEPTH,
    DENSE_WEIGHT,
    FUSION_MODES,
    BM25Index,
    reciprocal_rank_fusion,
    score_fusion,
    stored_dense_scores,
)

__all__ = ["BM25Retriever", "DenseRetriever", "HybridRetriever", "RerankRetriever", "reciprocal_rank_fusion"]


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
        return [[cid for cid, _ in hits] for hits, _ in self.search_batch(questions, k)]

    def search_batch(self, questions: Sequence[str], k: int) -> Iterator[Tuple[List[Tuple[str, float]], List[float]]]:
        """
        Per question, in order: the top-k (chunk id, dense score), best first, and
        the query embedding. The score is minus Chroma's squared L2 distance, as
        the app uses. Yields batch by batch, so only one batch of query
        embeddings is held in memory at a time.
        """
        for i in range(0, len(questions), self.batch_size):
            batch = list(questions[i : i + self.batch_size])
            embeddings = self.embed_model.get_text_embedding_batch(batch)
            response = self.collection.query(query_embeddings=embeddings, n_results=k)
            for ids, distances, embedding in zip(response["ids"], response["distances"], embeddings):
                yield [(cid, -dist) for cid, dist in zip(ids, distances)], embedding


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        candidate_depth: int = CANDIDATE_DEPTH,
        fusion: str = "score",
        dense_weight: float = DENSE_WEIGHT,
    ):
        """retrievers: the bm25 and dense retrievers, fused the way the app's HybridRetriever fuses them."""
        if fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES}, got {fusion!r}")
        self.retrievers = retrievers
        self.candidate_depth = candidate_depth
        self.fusion = fusion
        self.dense_weight = dense_weight

    def retrieve_batch(self, questions: Sequence[str], k: int) -> List[List[str]]:
        depth = max(k, self.candidate_depth)
        if self.fusion == "score":
            by_name = {r.name: r for r in self.retrievers}
            bm25, dense = by_name["bm25"].index, by_name["dense"]
            results = []
            for question, (hits, embedding) in zip(questions, dense.search_batch(questions, depth)):
                dense_scores = dict(hits)
                keyword_ids = bm25.rank(question, depth)
                # As in the app: exact dense scores for chunks only BM25 found, from stored embeddings
                bm25_only = [cid for cid in keyword_ids if cid not in dense_scores]
                dense_scores.update(stored_dense_scores(dense.collection, embedding, bm25_only))
                keyword = bm25.scores_for(question, list(dense_scores) + keyword_ids)
                results.append(score_fusion(dense_scores, keyword, k, self.dense_weight))
            return results
        per_retriever = [
            r.retrieve_batch(questions, depth) for r in self.retrievers
        ]
        return [
            reciprocal_rank_fusion([rankings[i] for rankings in per_retriever], k)
            for i in range(len(questions))
        ]


class RerankRetriever:
    """
    Reorders a base retriever's top candidates with the app's cross-encoder reranker.
    """

    name = "rerank"

    def __init__(self, base: Retriever, chunks: List[Chunk], reranker, candidates: int):
        self.base = base
        # Same text the app's reranker scores: the chunk without metadata
        self.text = {c.chunk_id: c.node.get_content() for c in chunks}
        self.reranker = reranker
        self.candidates = candidates

    def retrieve_batch(self, questions: Sequence[str], k: int) -> List[List[str]]:
        pools = self.base.retrieve_batch(questions, max(k, self.candidates))
        results = []
        for question, pool in zip(questions, pools):
            head, tail = pool[: self.candidates], pool[self.candidates :]
            scores = self.reranker.score(question, [self.text[cid] for cid in head])
            # Stable: ties keep the base order
            order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
            # Past the candidate cut the base order stays, so recall@k for k > candidates is still defined
            results.append(([head[i] for i in order] + tail)[:k])
        return results
