"""
Loads SQuAD v1.1 (dev) as a document-level QA benchmark.

Each Wikipedia article becomes one document (its paragraphs joined by blank
lines), mirroring how a user uploads a whole file to DocInsight. Every question
keeps the character span of its gold answer inside that document, so relevance
can be judged against whatever chunks the pipeline produces.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SQUAD_DEV_URL = (
    "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json"
)
DEFAULT_CACHE = Path("data/eval/squad-dev-v1.1.json")
PARAGRAPH_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class EvalDocument:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    doc_id: str
    question: str
    answers: Tuple[str, ...]
    # (start, end) character offsets of each gold answer within the document text.
    answer_spans: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class EvalDataset:
    documents: List[EvalDocument]
    queries: List[EvalQuery]


def download_squad(cache_path: Path = DEFAULT_CACHE) -> Path:
    """Download the SQuAD v1.1 dev set once and reuse the cached copy."""
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # A unique temp file plus an atomic replace, so concurrent runs never
        # share a partial download or read a half-written cache.
        fd, tmp_name = tempfile.mkstemp(dir=cache_path.parent, suffix=".part")
        os.close(fd)
        try:
            urllib.request.urlretrieve(SQUAD_DEV_URL, tmp_name)
            os.replace(tmp_name, cache_path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
    return cache_path


def build_dataset(raw: Dict) -> EvalDataset:
    """Convert the raw SQuAD JSON structure into documents and queries."""
    documents: List[EvalDocument] = []
    queries: List[EvalQuery] = []

    for article_idx, article in enumerate(raw["data"]):
        doc_id = f"squad-{article_idx:02d}"
        offset = 0
        paragraphs = []
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                spans = []
                answers = []
                for answer in qa["answers"]:
                    start = offset + answer["answer_start"]
                    end = start + len(answer["text"])
                    spans.append((start, end))
                    answers.append(answer["text"])
                queries.append(
                    EvalQuery(
                        query_id=qa["id"],
                        doc_id=doc_id,
                        question=qa["question"].strip(),
                        answers=tuple(dict.fromkeys(answers)),
                        answer_spans=tuple(dict.fromkeys(spans)),
                    )
                )
            paragraphs.append(context)
            offset += len(context) + len(PARAGRAPH_SEPARATOR)

        documents.append(
            EvalDocument(
                doc_id=doc_id,
                title=article["title"].replace("_", " "),
                text=PARAGRAPH_SEPARATOR.join(paragraphs),
            )
        )

    return EvalDataset(documents=documents, queries=queries)


def load_squad(
    cache_path: Path = DEFAULT_CACHE,
    max_articles: Optional[int] = None,
    max_questions: Optional[int] = None,
    seed: int = 13,
) -> EvalDataset:
    """
    Load SQuAD dev as an EvalDataset.

    Args:
        cache_path: Where the raw JSON is cached.
        max_articles: Keep only the first N articles (smaller corpus, faster runs).
        max_questions: Randomly sample N questions (seeded) from the kept articles.
        seed: Sampling seed so runs are reproducible.
    """
    with open(download_squad(cache_path), encoding="utf-8") as f:
        raw = json.load(f)
    if max_articles is not None:
        raw = {**raw, "data": raw["data"][:max_articles]}

    dataset = build_dataset(raw)
    queries = dataset.queries
    if max_questions is not None and max_questions < len(queries):
        queries = random.Random(seed).sample(queries, max_questions)
    return EvalDataset(documents=dataset.documents, queries=queries)
