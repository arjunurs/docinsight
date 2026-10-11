"""
Hybrid retrieval: BM25 keyword search and dense vector search, fused either by
their scores (the default) or by reciprocal rank fusion (RRF). Shared by the
app's QueryEngine and the evaluation harness, so both rank chunks the same way.
"""
import math
import re
from typing import Dict, List, Mapping, Optional, Sequence

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore
from llama_index.core.vector_stores.utils import metadata_dict_to_node

from utils.tracing import stage

RRF_K = 60
CANDIDATE_DEPTH = 50
FUSION_MODES = ("score", "rrf")
# Weight on dense in score fusion, chosen on SQuAD and Natural Questions (see README)
DENSE_WEIGHT = 0.7

# Small English stopword list; enough to stop BM25 from rewarding "the"/"of".
_STOPWORDS = frozenset(
    "a an and are as at be by did do does for from had has have how in is it its "
    "of on or that the their this to was were what when where which who whom whose "
    "why with".split()
)
_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens with stopwords removed, as BM25 indexes them."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def bm25_text(node: BaseNode) -> str:
    """The string BM25 indexes for a chunk: the same text and metadata the embedder sees."""
    return node.get_content(metadata_mode=MetadataMode.EMBED)


def rrf_scores(rankings: Sequence[Sequence[str]], rrf_k: int = RRF_K) -> Dict[str, float]:
    """RRF score per id (Cormack et al., SIGIR 2009): sum over rankings of 1/(rrf_k + rank)."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int, rrf_k: int = RRF_K) -> List[str]:
    """Fuse ranked lists with RRF; ties break on id so the order is deterministic."""
    scores = rrf_scores(rankings, rrf_k)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]


def _min_max(scores: Mapping[str, float]) -> Dict[str, float]:
    lo, hi = min(scores.values()), max(scores.values())
    return {cid: (v - lo) / (hi - lo) if hi > lo else 0.0 for cid, v in scores.items()}


def fused_scores(
    dense: Mapping[str, float], keyword: Mapping[str, float], dense_weight: float = DENSE_WEIGHT
) -> Dict[str, float]:
    """
    Score fusion: a weighted sum of each retriever's scores, min-max scaled per query.

    Candidates are the ids in either map, and higher scores are better. A candidate
    missing from one map takes that map's lowest score, since it fell below that
    retriever's cutoff. Unlike RRF this sees how strong a match is, so a confident
    keyword match can win while weak keyword matches barely move the dense order.
    """
    ids = set(dense) | set(keyword)
    if not ids:
        return {}
    dense_floor = min(dense.values()) if dense else 0.0
    keyword_floor = min(keyword.values()) if keyword else 0.0
    d = _min_max({cid: dense.get(cid, dense_floor) for cid in ids})
    kw = _min_max({cid: keyword.get(cid, keyword_floor) for cid in ids})
    return {cid: dense_weight * d[cid] + (1 - dense_weight) * kw[cid] for cid in ids}


def score_fusion(
    dense: Mapping[str, float], keyword: Mapping[str, float], k: int, dense_weight: float = DENSE_WEIGHT
) -> List[str]:
    """Top-k ids by fused_scores; ties break on id so the order is deterministic."""
    scores = fused_scores(dense, keyword, dense_weight)
    return sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]


def chroma_score_to_dense(score: float) -> float:
    """
    LlamaIndex reports a Chroma hit as exp(-squared L2 distance). Undo that, so
    score fusion scales a value linear in cosine similarity (for unit-length
    embeddings, squared L2 distance is 2 - 2 * cosine).
    """
    return math.log(score) if score > 0 else -math.inf


def stored_dense_scores(collection, query_embedding: Sequence[float], ids: Sequence[str]) -> Dict[str, float]:
    """
    Dense scores for chunks dense search did not return, from their embeddings
    stored in Chroma: minus the squared L2 distance to the query embedding, the
    same scale as chroma_score_to_dense gives search hits. No embedding API call.
    """
    if not ids:
        return {}
    import numpy as np

    stored = collection.get(ids=list(ids), include=["embeddings"])
    query = np.asarray(query_embedding, dtype=float)
    # Chroma may return ids in a different order than asked, so match by id
    return {
        cid: -float(((np.asarray(vector, dtype=float) - query) ** 2).sum())
        for cid, vector in zip(stored["ids"], stored["embeddings"])
    }


def embed_query(index: VectorStoreIndex, query_bundle: QueryBundle) -> None:
    """
    Embed the query into the bundle in its own traced stage, so the embedding API
    call is timed apart from the vector search (which then reuses the embedding).
    """
    if query_bundle.embedding is not None or not query_bundle.embedding_strs:
        return
    # The index's own embed model, the one its retrievers would use
    embed_model = index._embed_model
    with stage("retrieval.embed_query", {"gen_ai.operation.name": "embeddings",
                                         "gen_ai.request.model": getattr(embed_model, "model_name", None)}):
        query_bundle.embedding = embed_model.get_agg_embedding_from_queries(query_bundle.embedding_strs)


def dense_search(index: VectorStoreIndex, query_bundle: QueryBundle, top_k: int) -> List[NodeWithScore]:
    """Vector search over the index, traced as two stages: embedding the query, then the search itself."""
    embed_query(index, query_bundle)
    with stage("retrieval.dense", {"docinsight.retrieval.top_k": top_k}) as span:
        hits = index.as_retriever(similarity_top_k=top_k).retrieve(query_bundle)
        span.set_attribute("docinsight.retrieval.hits", len(hits))
    return hits


class DenseRetriever(BaseRetriever):
    """Vector search only (RETRIEVAL_MODE=dense), traced like the dense half of HybridRetriever."""

    def __init__(self, index: VectorStoreIndex, similarity_top_k: int = 3):
        super().__init__()
        self.index = index
        self.similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return dense_search(self.index, query_bundle, self.similarity_top_k)


class BM25Index:
    """
    BM25 over a fixed list of chunk nodes, ranked by node id (or by the ids given).
    """

    def __init__(self, nodes: Sequence[BaseNode], ids: Optional[Sequence[str]] = None):
        from rank_bm25 import BM25Okapi

        self.nodes = list(nodes)
        self.node_ids = list(ids) if ids is not None else [node.node_id for node in self.nodes]
        self.by_id = dict(zip(self.node_ids, self.nodes))
        self.position = {node_id: i for i, node_id in enumerate(self.node_ids)}
        self.bm25 = BM25Okapi([tokenize(bm25_text(node)) for node in self.nodes]) if self.nodes else None

    def rank(self, query: str, k: int) -> List[str]:
        """
        Top-k node ids for a query, best first. Chunks that share no word with
        the query are left out, so a query with no keyword match returns an
        empty ranking instead of arbitrary chunks for RRF to reward.
        """
        if self.bm25 is None:
            return []
        import numpy as np

        terms = tokenize(query)
        scores = self.bm25.get_scores(terms)
        order = np.argsort(-scores, kind="stable")
        # Test for shared words, not score > 0: Okapi IDF is exactly 0 for a
        # word in half the chunks, so a real match can still score 0. Filter
        # before cutting to k, or zero-score matches past the cut would be lost.
        matches = [self.node_ids[i] for i in order if any(t in self.bm25.doc_freqs[i] for t in terms)]
        return matches[:k]

    def scores_for(self, query: str, ids: Sequence[str]) -> Dict[str, float]:
        """BM25 score of each given id for a query (ids not in the index are skipped)."""
        if self.bm25 is None:
            return {}
        scores = self.bm25.get_scores(tokenize(query))
        return {cid: float(scores[self.position[cid]]) for cid in ids if cid in self.position}


class HybridRetriever(BaseRetriever):
    """
    Fuses dense and BM25 results for the chunks stored in a Chroma-backed index,
    by score (fusion="score", the default) or by rank (fusion="rrf").

    The BM25 side is built from the persisted Chroma collection itself, so it
    covers exactly the chunks the app stores and is rebuilt after a restart
    without any extra files. It is rebuilt whenever the collection's size
    changes, which is how new uploads show up (chunks are only ever added).
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 3,
        candidate_depth: int = CANDIDATE_DEPTH,
        rrf_k: int = RRF_K,
        fusion: str = "score",
        dense_weight: float = DENSE_WEIGHT,
    ):
        if fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES}, got {fusion!r}")
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError(f"dense_weight must be between 0 and 1, got {dense_weight}")
        super().__init__()
        self.index = index
        self.collection = index.vector_store.client
        self.similarity_top_k = similarity_top_k
        self.candidate_depth = candidate_depth
        self.rrf_k = rrf_k
        self.fusion = fusion
        self.dense_weight = dense_weight
        self._bm25: Optional[BM25Index] = None
        self._bm25_size = -1

    def _bm25_index(self) -> BM25Index:
        size = self.collection.count()
        if self._bm25 is None or size != self._bm25_size:
            stored = self.collection.get(include=["documents", "metadatas"])
            nodes = []
            for text, metadata in zip(stored["documents"], stored["metadatas"]):
                node = metadata_dict_to_node(metadata)
                node.set_content(text)
                nodes.append(node)
            self._bm25 = BM25Index(nodes)
            self._bm25_size = size
        return self._bm25

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        depth = max(self.similarity_top_k, self.candidate_depth)
        dense_hits = dense_search(self.index, query_bundle, depth)
        with stage("retrieval.bm25", {"docinsight.retrieval.top_k": depth}) as span:
            previous = self._bm25
            bm25 = self._bm25_index()
            keyword_ids = bm25.rank(query_bundle.query_str, depth)
            # Rebuilding BM25 after an upload is the one slow path here; flag it on the span
            span.set_attribute("docinsight.bm25.rebuilt", bm25 is not previous)
            span.set_attribute("docinsight.retrieval.hits", len(keyword_ids))

        nodes = {hit.node.node_id: hit.node for hit in dense_hits}
        with stage("retrieval.fusion", {"docinsight.fusion.mode": self.fusion}) as span:
            if self.fusion == "rrf":
                rankings = [[hit.node.node_id for hit in dense_hits], keyword_ids]
                scores = rrf_scores(rankings, self.rrf_k)
                fused = reciprocal_rank_fusion(rankings, self.similarity_top_k, self.rrf_k)
            else:
                span.set_attribute("docinsight.fusion.dense_weight", self.dense_weight)
                dense = {hit.node.node_id: chroma_score_to_dense(hit.score) for hit in dense_hits}
                # Dense search embedded the query into the bundle; reuse it to score the
                # chunks only BM25 found from their stored embeddings, so every candidate
                # gets an exact dense score
                if query_bundle.embedding is not None:
                    bm25_only = [cid for cid in keyword_ids if cid not in dense]
                    span.set_attribute("docinsight.fusion.bm25_only", len(bm25_only))
                    dense.update(stored_dense_scores(self.collection, query_bundle.embedding, bm25_only))
                # Exact BM25 scores for every candidate, dense-only ones included
                keyword = bm25.scores_for(query_bundle.query_str, list(dense) + keyword_ids)
                scores = fused_scores(dense, keyword, self.dense_weight)
                fused = sorted(scores, key=lambda cid: (-scores[cid], cid))[: self.similarity_top_k]
            span.set_attribute("docinsight.fusion.candidates", len(scores))
        return [
            NodeWithScore(node=nodes.get(node_id) or bm25.by_id[node_id], score=scores[node_id])
            for node_id in fused
        ]
