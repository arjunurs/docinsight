"""
Query engine module for processing queries and generating responses.
"""
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response import Response

from utils.prompts import REFINE_PROMPT, TEXT_QA_PROMPT
from utils.reranking import DEFAULT_CANDIDATES, CrossEncoderReranker
from utils.retrieval import DENSE_WEIGHT, HybridRetriever

RETRIEVAL_MODES = ("hybrid", "dense")


class QueryEngine:
    """
    Class for handling queries against a vector index.
    """

    def __init__(
        self,
        index: Optional[VectorStoreIndex] = None,
        retrieval_mode: str = "hybrid",
        similarity_top_k: int = 3,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        rerank_candidates: int = DEFAULT_CANDIDATES,
        fusion: str = "score",
        dense_weight: float = DENSE_WEIGHT,
    ):
        """
        Initialize the query engine with an optional index.
        
        Args:
            index: VectorStoreIndex to query against (optional)
            retrieval_mode: "hybrid" (BM25 + dense, fused) or "dense" (vector search only)
            similarity_top_k: Number of chunks passed to the LLM
            text_qa_template: Answer prompt; defaults to one that answers only from the
                retrieved chunks and says so when they do not contain the answer
            refine_template: Prompt for refining an answer over more chunks (same default rule)
            reranker: Optional cross-encoder; when set, the retriever returns rerank_candidates
                chunks and the reranker keeps the best similarity_top_k for the LLM
            rerank_candidates: Chunks retrieved for the reranker to choose from
            fusion: How hybrid mode fuses BM25 and dense: "score" (weighted, normalized scores) or "rrf"
            dense_weight: Weight on dense in score fusion (0 to 1)
        """
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {retrieval_mode!r}")
        self.index = index
        self.retrieval_mode = retrieval_mode
        self.similarity_top_k = similarity_top_k
        self.text_qa_template = text_qa_template or TEXT_QA_PROMPT
        self.refine_template = refine_template or REFINE_PROMPT
        self.reranker = reranker
        self.fusion = fusion
        self.dense_weight = dense_weight
        if self.reranker is not None:
            self.reranker.top_n = similarity_top_k
        self.rerank_candidates = max(rerank_candidates, similarity_top_k)
        self.query_engine = None
        if self.index:
            self._setup_query_engine()
    
    def set_index(self, index: VectorStoreIndex):
        """
        Set the index for the query engine.
        
        Args:
            index: VectorStoreIndex to query against
        """
        self.index = index
        self._setup_query_engine()
    
    def _setup_query_engine(self):
        """Set up the query engine with the current index."""
        if not self.index:
            return
        top_k = self.rerank_candidates if self.reranker else self.similarity_top_k
        if self.retrieval_mode == "hybrid":
            retriever = HybridRetriever(
                self.index, similarity_top_k=top_k, fusion=self.fusion, dense_weight=self.dense_weight
            )
        else:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever,
            response_mode="compact",
            text_qa_template=self.text_qa_template,
            refine_template=self.refine_template,
            node_postprocessors=[self.reranker] if self.reranker else None,
        )
    
    def query(self, query_text: str) -> Optional[Response]:
        """
        Process a query and return a response.
        
        Args:
            query_text: Query text to process
            
        Returns:
            Response object or None if no index is set
        """
        if not self.query_engine:
            return None
        
        response = self.query_engine.query(query_text)
        return response 