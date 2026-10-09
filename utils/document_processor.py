"""
Document processor module for loading and processing documents.
"""
import hashlib
from collections import defaultdict
from typing import Dict, List

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import BaseNode


def _file_digest(path: str) -> str:
    """Return a short SHA-256 digest of a file's bytes."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            sha.update(block)
    return sha.hexdigest()[:16]


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
            chunk_overlap=self.chunk_overlap,
            # Deterministic chunk ids, so re-ingesting the same file yields the same nodes
            id_func=lambda i, doc: f"{doc.id_}_{i}",
        )

    def load_documents(self, file_paths: List[str]) -> List[Document]:
        """
        Load documents from a list of file paths.

        Document ids are derived from file content, so the same file uploaded
        twice (even under a different name or temp path) gets the same ids.

        Args:
            file_paths: List of file paths to load

        Returns:
            List of Document objects
        """
        digests: Dict[str, str] = {path: _file_digest(path) for path in file_paths}
        reader = SimpleDirectoryReader(input_files=file_paths)
        documents = reader.load_data()

        # PDFs load as one Document per page; number pages within each file
        page_counter: Dict[str, int] = defaultdict(int)
        for doc in documents:
            file_path = doc.metadata.get("file_path", "")
            digest = digests.get(file_path) or hashlib.sha256(doc.text.encode()).hexdigest()[:16]
            doc.id_ = f"{digest}_p{page_counter[file_path]}"
            doc.metadata["content_hash"] = digest
            page_counter[file_path] += 1
        return documents

    def process_documents(self, documents: List[Document]) -> List[BaseNode]:
        """
        Split documents into chunk nodes. This is the only place chunking happens.

        Args:
            documents: List of Document objects

        Returns:
            List of chunk nodes, each linked to its source document
        """
        # Identical files in one batch share document ids; split each document once
        unique_documents = []
        seen_doc_ids = set()
        for doc in documents:
            if doc.id_ not in seen_doc_ids:
                seen_doc_ids.add(doc.id_)
                unique_documents.append(doc)
        nodes = self.text_splitter.get_nodes_from_documents(unique_documents)

        chunks_per_doc: Dict[str, int] = defaultdict(int)
        for node in nodes:
            chunks_per_doc[node.ref_doc_id] += 1

        seen: Dict[str, int] = defaultdict(int)
        for node in nodes:
            node.metadata["chunk_id"] = seen[node.ref_doc_id]
            node.metadata["total_chunks"] = chunks_per_doc[node.ref_doc_id]
            seen[node.ref_doc_id] += 1

        return nodes
