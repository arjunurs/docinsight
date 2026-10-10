"""
Hybrid retrieval: BM25 keyword search and dense vector search fused with
reciprocal rank fusion (RRF). Shared by the app's QueryEngine and the
evaluation harness, so both rank chunks the same way.
"""
import re
from typing import Dict, List, Optional, Sequence

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore
from llama_index.core.vector_stores.utils import metadata_dict_to_node

RRF_K = 60
CANDIDATE_DEPTH = 50

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


class BM25Index:
    """
    BM25 over a fixed list of chunk nodes, ranked by node id (or by the ids given).
    """

    def __init__(self, nodes: Sequence[BaseNode], ids: Optional[Sequence[str]] = None):
        from rank_bm25 import BM25Okapi

        self.nodes = list(nodes)
        self.node_ids = list(ids) if ids is not None else [node.node_id for node in self.nodes]
        self.by_id = dict(zip(self.node_ids, self.nodes))
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


class HybridRetriever(BaseRetriever):
    """
    Fuses dense and BM25 rankings of the chunks stored in a Chroma-backed index.

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
    ):
        super().__init__()
        self.index = index
        self.collection = index.vector_store.client
        self.similarity_top_k = similarity_top_k
        self.candidate_depth = candidate_depth
        self.rrf_k = rrf_k
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
        dense_hits = self.index.as_retriever(similarity_top_k=depth).retrieve(query_bundle)
        bm25 = self._bm25_index()

        nodes = {hit.node.node_id: hit.node for hit in dense_hits}
        rankings = [[hit.node.node_id for hit in dense_hits], bm25.rank(query_bundle.query_str, depth)]
        scores = rrf_scores(rankings, self.rrf_k)
        fused = reciprocal_rank_fusion(rankings, self.similarity_top_k, self.rrf_k)
        return [
            NodeWithScore(node=nodes.get(node_id) or bm25.by_id[node_id], score=scores[node_id])
            for node_id in fused
        ]
