"""
Optional cross-encoder reranking. Off by default; enable with RERANKER=minilm
after installing the extra (pip install -r requirements-rerank.txt).

The retriever returns more candidates than the LLM needs (RERANK_CANDIDATES,
10 by default), a cross-encoder scores each (query, chunk) pair, and only the
best top_n go to the LLM.
"""
import importlib.util
from typing import Callable, List, Optional, Sequence

from llama_index.core import QueryBundle
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

# Short names accepted by RERANKER, mapped to Hugging Face cross-encoder models
RERANKERS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}
DEFAULT_CANDIDATES = 10

Scorer = Callable[[str, Sequence[str]], List[float]]

MISSING_EXTRA ="Reranking needs the optional extra: pip install -r requirements-rerank.txt"


def _load_cross_encoder(model_name: str) -> Scorer:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as err:
        raise ImportError(MISSING_EXTRA) from err
    model = CrossEncoder(model_name, max_length=512)

    def score(query: str, texts: Sequence[str]) -> List[float]:
        return [float(s) for s in model.predict([(query, text) for text in texts], show_progress_bar=False)]

    return score


class CrossEncoderReranker(BaseNodePostprocessor):
    """
    Reorders retrieved chunks by cross-encoder relevance and keeps the best top_n.
    """

    model_name: str
    top_n: int = 3
    _scorer: Optional[Scorer] = PrivateAttr(default=None)

    def __init__(self, model_name: str, top_n: int = 3, scorer: Optional[Scorer] = None):
        """
        Args:
            model_name: Hugging Face cross-encoder model id
            top_n: Number of chunks kept for the LLM
            scorer: Replaces the model (tests use an offline fake); loaded lazily otherwise
        """
        super().__init__(model_name=model_name, top_n=top_n)
        self._scorer = scorer

    @classmethod
    def class_name(cls) -> str:
        return "CrossEncoderReranker"

    def score(self, query: str, texts: Sequence[str]) -> List[float]:
        """Relevance score for each text against the query; higher is better."""
        if self._scorer is None:
            self._scorer = _load_cross_encoder(self.model_name)
        return self._scorer(query, texts)

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if not nodes or query_bundle is None:
            return nodes[: self.top_n]
        scores = self.score(query_bundle.query_str, [n.node.get_content() for n in nodes])
        # Stable sort: ties keep the retriever's order
        ranked = sorted(zip(scores, range(len(nodes)), nodes), key=lambda t: (-t[0], t[1]))
        return [NodeWithScore(node=n.node, score=s) for s, _, n in ranked[: self.top_n]]


def reranker_key(name: Optional[str]) -> str:
    """A RERANKER setting normalized: lowercase, with "", "none" and "off" all meaning off ("")."""
    key = (name or "").strip().lower()
    return "" if key in ("none", "off") else key


def build_reranker(name: Optional[str], top_n: int = 3) -> Optional[CrossEncoderReranker]:
    """
    The reranker for a RERANKER setting, or None when reranking is off.

    Accepts a short name from RERANKERS or "" / "none" / "off" to disable.
    """
    key = reranker_key(name)
    if not key:
        return None
    if key not in RERANKERS:
        raise ValueError(f"Unknown reranker {name!r}; choose one of {sorted(RERANKERS)} or leave it unset")
    # Fail at startup, not on the first question, when the extra is not installed
    if importlib.util.find_spec("sentence_transformers") is None:
        raise ImportError(MISSING_EXTRA)
    return CrossEncoderReranker(RERANKERS[key], top_n=top_n)
