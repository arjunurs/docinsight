"""
Query engine module for processing queries and generating responses.
"""
from typing import Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.response import Response


class QueryEngine:
    """
    Class for handling queries against a vector index.
    """

    def __init__(self, index: Optional[VectorStoreIndex] = None):
        """
        Initialize the query engine with an optional index.
        
        Args:
            index: VectorStoreIndex to query against (optional)
        """
        self.index = index
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
        if self.index:
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=3,
                response_mode="compact"
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