"""
Index manager module for creating and managing vector indices.
"""
import os
from typing import List, Optional, Sequence

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import BaseNode


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
        temperature: float = 0.0
    ):
        """
        Initialize the index manager with the specified parameters.
        
        Args:
            persist_dir: Directory where ChromaDB data will be stored
            collection_name: Name of the ChromaDB collection
            embedding_model: Name of the OpenAI embedding model to use
            llm_model: Name of the OpenAI LLM model to use
            temperature: Temperature parameter for the LLM (0 keeps answers grounded and repeatable)
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.temperature = temperature
        
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
        
        # Chunking is done by DocumentProcessor; the index only embeds and stores nodes
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
        
        # Reattach to whatever is already persisted, so restarts don't lose the index
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            embed_model=self.embed_model
        )

    def has_documents(self) -> bool:
        """
        Whether the persisted collection holds any chunks.
        """
        return self.chroma_collection.count() > 0

    def is_indexed(self, ref_doc_id: str) -> bool:
        """
        Whether chunks for the given source document are already stored.
        """
        result = self.chroma_collection.get(where={"document_id": ref_doc_id}, limit=1)
        return len(result["ids"]) > 0

    def add_nodes(self, nodes: Sequence[BaseNode]) -> List[BaseNode]:
        """
        Embed and store chunk nodes, skipping source documents that are already indexed.
        
        Args:
            nodes: Chunk nodes produced by DocumentProcessor
            
        Returns:
            The nodes that were newly added (empty if everything was already indexed)
        """
        known = {}
        new_nodes = []
        for node in nodes:
            doc_id = node.ref_doc_id
            if doc_id not in known:
                known[doc_id] = self.is_indexed(doc_id)
            if not known[doc_id]:
                new_nodes.append(node)

        if new_nodes:
            self.index.insert_nodes(new_nodes, show_progress=True)
        return new_nodes
    
    def get_index(self) -> Optional[VectorStoreIndex]:
        """
        Get the current index.
        
        Returns:
            The current VectorStoreIndex or None if not created
        """
        return self.index 