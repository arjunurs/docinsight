"""
Chunks evaluation documents with the same splitter settings the app uses and
labels which chunks are relevant to each query.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from evaluation.dataset import EvalDocument, EvalQuery


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start: int
    end: int


def chunk_documents(
    documents: List[EvalDocument], chunk_size: int = 512, chunk_overlap: int = 50
) -> List[Chunk]:
    """
    Split documents with LlamaIndex's SentenceSplitter, keeping character offsets.

    Defaults match DocumentProcessor and IndexManager (512 tokens, 50 overlap).
    Only the title is attached as metadata so the token budget per chunk is
    the same as the app's for a single uploaded file.
    """
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[Chunk] = []
    for doc in documents:
        llama_doc = Document(text=doc.text, id_=doc.doc_id, metadata={"file_name": doc.title})
        for i, node in enumerate(splitter.get_nodes_from_documents([llama_doc])):
            if node.start_char_idx is None or node.end_char_idx is None:
                raise ValueError(f"Chunk {i} of {doc.doc_id} has no character offsets")
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{i}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=node.get_content(),
                    start=node.start_char_idx,
                    end=node.end_char_idx,
                )
            )
    return chunks


def label_relevant_chunks(queries: List[EvalQuery], chunks: List[Chunk]) -> Dict[str, Set[str]]:
    """
    Map each query to the chunks that fully contain one of its gold answer spans.

    A query whose answer straddles every chunk boundary gets an empty set; the
    runner reports and excludes those rather than counting them as misses.
    """
    chunks_by_doc: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk)

    relevant: Dict[str, Set[str]] = {}
    for query in queries:
        relevant[query.query_id] = {
            chunk.chunk_id
            for chunk in chunks_by_doc.get(query.doc_id, [])
            for start, end in query.answer_spans
            if chunk.start <= start and end <= chunk.end
        }
    return relevant
