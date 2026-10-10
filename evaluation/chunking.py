"""
Chunks evaluation documents through the app's own ingestion path and labels
which chunks are relevant to each query.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Set

from llama_index.core.schema import BaseNode

from evaluation.dataset import EvalDocument, EvalQuery
from utils.document_processor import DocumentProcessor


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    start: int
    end: int
    # The node the app would index, with its metadata and metadata exclusions.
    node: BaseNode = field(compare=False, repr=False)


def chunk_documents(
    documents: List[EvalDocument], chunk_size: int = 512, chunk_overlap: int = 50
) -> List[Chunk]:
    """
    Chunk documents exactly as an upload is chunked in the app.

    Each article is written to a .txt file in a temp directory, as app.py does
    with uploads, then loaded and split by DocumentProcessor. That keeps the
    reader's metadata (file_path, content_hash, ...) and its exclusions, which
    change both the splitter's token budget and the text that gets embedded.
    """
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[Chunk] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for doc in documents:
            path = os.path.join(temp_dir, f"{doc.title}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(doc.text)
            loaded = processor.load_documents([path])
            if len(loaded) != 1 or loaded[0].text != doc.text:
                raise ValueError(f"Ingestion changed the text of {doc.doc_id}; answer offsets would not match")
            for i, node in enumerate(processor.process_documents(loaded)):
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
                        node=node,
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
