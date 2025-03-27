"""
Document processor module for loading and processing documents.
"""
import os
from typing import List

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser.text.sentence import SentenceSplitter


class DocumentProcessor:
    """
    Class for loading and processing documents from different file formats.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document processor with configurable chunking parameters.
        
        Args:
            chunk_size: Number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from a list of file paths.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of Document objects
        """
        reader = SimpleDirectoryReader(input_files=file_paths)
        documents = reader.load_data()
        return documents
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents by splitting them into manageable chunks.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of processed Document objects with appropriate chunking
        """
        # Create a list to hold our processed documents
        processed_documents = []
        
        for doc in documents:
            # For each document, split the text into chunks
            text_chunks = self.text_splitter.split_text(doc.text)
            
            # Create a new Document object for each chunk
            for i, chunk in enumerate(text_chunks):
                # Copy metadata from original document
                metadata = doc.metadata.copy() if doc.metadata else {}
                # Add chunk information to metadata
                metadata["chunk_id"] = i
                metadata["total_chunks"] = len(text_chunks)
                
                # Create a new Document with the chunk text and metadata
                chunk_doc = Document(
                    text=chunk,
                    metadata=metadata,
                    id_=f"{doc.id_}_{i}" if doc.id_ else f"doc_{len(processed_documents)}"
                )
                processed_documents.append(chunk_doc)
                
        return processed_documents 