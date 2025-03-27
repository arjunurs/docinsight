"""
Index manager module for creating and managing vector indices.
"""
import os
from typing import List, Optional

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import Document
from llama_index.core.node_parser.text.sentence import SentenceSplitter


class IndexManager:
    """
    Class for managing vector indices and storage.
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "document_collection",
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize the index manager with the specified parameters.
        
        Args:
            persist_dir: Directory where ChromaDB data will be stored
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the OpenAI embedding model to use
            llm_model: Name of the OpenAI LLM model to use
            temperature: Temperature parameter for the LLM
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)
        
        # Create vector store
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Initialize embedding and LLM models
        self.embed_model = OpenAIEmbedding(model=self.embedding_model)
        self.llm = OpenAI(model=self.llm_model, temperature=self.temperature)
        
        # Configure settings with node parser for automatic document chunking
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        self.index = None

    def create_index(self, documents: List[Document]) -> VectorStoreIndex:
        """
        Create a vector index from the provided documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            VectorStoreIndex created from the documents
        """
        # Let VectorStoreIndex handle node chunking internally
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=True
        )
        return self.index
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """
        Get the current index.
        
        Returns:
            The current VectorStoreIndex or None if not created
        """
        return self.index 