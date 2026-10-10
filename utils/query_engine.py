"""
Query engine module for processing queries and generating responses.
"""
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response import Response

from utils.prompts import REFINE_PROMPT, TEXT_QA_PROMPT
from utils.retrieval import HybridRetriever

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
    ):
        """
        Initialize the query engine with an optional index.
        
        Args:
            index: VectorStoreIndex to query against (optional)
            retrieval_mode: "hybrid" (BM25 + dense, fused with RRF) or "dense" (vector search only)
            similarity_top_k: Number of chunks passed to the LLM
            text_qa_template: Answer prompt; defaults to one that answers only from the
                retrieved chunks and says so when they do not contain the answer
            refine_template: Prompt for refining an answer over more chunks (same default rule)
        """
        if retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {retrieval_mode!r}")
        self.index = index
        self.retrieval_mode = retrieval_mode
        self.similarity_top_k = similarity_top_k
        self.text_qa_template = text_qa_template or TEXT_QA_PROMPT
        self.refine_template = refine_template or REFINE_PROMPT
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
        if self.retrieval_mode == "hybrid":
            retriever = HybridRetriever(self.index, similarity_top_k=self.similarity_top_k)
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever,
                response_mode="compact",
                text_qa_template=self.text_qa_template,
                refine_template=self.refine_template,
            )
        else:
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=self.similarity_top_k,
                response_mode="compact",
                text_qa_template=self.text_qa_template,
                refine_template=self.refine_template,
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