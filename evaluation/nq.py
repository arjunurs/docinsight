"""
Loads Natural Questions (the MRQA 2019 shared-task dev split) as a second
document QA benchmark.

SQuAD questions were written by annotators reading the paragraph, so they
reuse its words. NQ questions are real Google search queries, written before
anyone saw the answer passage (Kwiatkowski et al., 2019), which makes them a
harder, less lexical test of retrieval.

MRQA ships NQ in the same span format as SQuAD: each example is one Wikipedia
passage (the NQ "long answer": a paragraph, list or table) with character
spans for its short answers. Passages keep NQ's HTML tags (<P>, <Td>, ...);
those are stripped here so documents read like the plain text a user would
upload, and answer spans are remapped to the cleaned text. Each distinct
passage becomes one document.
"""
from __future__ import annotations

import gzip
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from evaluation.dataset import EvalDataset, EvalDocument, EvalQuery, download

NQ_DEV_URL = "https://s3.us-east-2.amazonaws.com/mrqa/release/v2/dev/NaturalQuestionsShort.jsonl.gz"
DEFAULT_NQ_CACHE = Path("data/eval/mrqa-nq-dev.jsonl.gz")

_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9]*)>")
# Tags that end a line in the rendered text; any other tag (table cells) becomes a space.
_BLOCK_TAGS = {"p", "tr", "li", "dd", "dt", "table", "ul", "ol", "dl"}


def strip_tags(context: str) -> Tuple[str, Dict[int, int]]:
    """
    Remove NQ's HTML tag tokens from a passage.

    MRQA passages are whitespace-tokenized and every tag is its own token, so
    the text is rebuilt from the non-tag tokens: joined by a newline when a
    block tag (paragraph, row, list item) separated them, else by a space.

    Returns the cleaned text and a map from each kept character's offset in
    the original passage to its offset in the cleaned text.
    """
    parts: List[str] = []
    mapping: Dict[int, int] = {}
    length = 0
    pending = ""
    for match in re.finditer(r"\S+", context):
        token = match.group()
        tag = _TAG.fullmatch(token)
        if tag:
            if length and tag.group(1).lower() in _BLOCK_TAGS:
                pending = "\n"
            continue
        if length:
            sep = pending or " "
            parts.append(sep)
            length += len(sep)
        pending = ""
        for i in range(len(token)):
            mapping[match.start() + i] = length + i
        parts.append(token)
        length += len(token)
    return "".join(parts), mapping


def _remap_span(mapping: Dict[int, int], start: int, end_inclusive: int) -> Optional[Tuple[int, int]]:
    """Map an MRQA span (inclusive end) to a [start, end) span in the cleaned text."""
    kept = [mapping[i] for i in range(start, end_inclusive + 1) if i in mapping]
    if not kept:
        return None  # the span was nothing but tags
    return kept[0], kept[-1] + 1


def build_nq_dataset(examples: Iterable[Dict]) -> EvalDataset:
    """
    Convert MRQA examples (header line already dropped) into documents and queries.

    NQ dev is annotated by up to five people per question, and MRQA keeps each
    annotation as its own example, so 12,836 examples hold 4,177 questions.
    Copies of a question are merged into one query whose gold answers are the
    union of the annotators' spans, across whichever passages they chose;
    otherwise questions with more annotators would count several times.
    """
    documents: List[EvalDocument] = []
    doc_ids: Dict[str, str] = {}
    # question text -> (first qid, [(doc_id, start, end), ...], [answer text, ...])
    merged: Dict[str, Tuple[str, List[Tuple[str, int, int]], List[str]]] = {}

    for example in examples:
        text, mapping = strip_tags(example["context"])
        if text not in doc_ids:
            doc_id = f"nq-{len(documents):04d}"
            doc_ids[text] = doc_id
            # NQ passages carry no page title; the id doubles as the upload's file name.
            documents.append(EvalDocument(doc_id=doc_id, title=doc_id, text=text))
        doc_id = doc_ids[text]

        for qa in example["qas"]:
            question = qa["question"].strip()
            _, locations, answers = merged.setdefault(question, (qa["qid"], [], []))
            for detected in qa["detected_answers"]:
                for start, end in detected["char_spans"]:
                    span = _remap_span(mapping, start, end)
                    if span is not None:
                        locations.append((doc_id, *span))
                        # The cleaned span text, so answer metrics compare like with like.
                        answers.append(text[span[0] : span[1]])

    queries: List[EvalQuery] = []
    for question, (query_id, locations, answers) in merged.items():
        locations = list(dict.fromkeys(locations))
        if not locations:
            continue
        doc_id = locations[0][0]
        queries.append(
            EvalQuery(
                query_id=query_id,
                doc_id=doc_id,
                question=question,
                answers=tuple(dict.fromkeys(answers)),
                answer_spans=tuple((s, e) for d, s, e in locations if d == doc_id),
                other_spans=tuple(loc for loc in locations if loc[0] != doc_id),
            )
        )
    return EvalDataset(documents=documents, queries=queries)


def read_mrqa(path: Path) -> List[Dict]:
    """Read an MRQA .jsonl.gz file, skipping its header line."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [row for row in rows if "header" not in row]


def load_nq(
    cache_path: Path = DEFAULT_NQ_CACHE,
    max_documents: Optional[int] = None,
    max_questions: Optional[int] = None,
    seed: int = 13,
) -> EvalDataset:
    """
    Load MRQA Natural Questions dev as an EvalDataset.

    Args:
        cache_path: Where the raw .jsonl.gz is cached.
        max_documents: Keep only the first N distinct passages and their questions.
        max_questions: Randomly sample N questions (seeded) from the kept passages.
        seed: Sampling seed so runs are reproducible.
    """
    dataset = build_nq_dataset(read_mrqa(download(NQ_DEV_URL, cache_path)))
    documents, queries = dataset.documents, dataset.queries
    if max_documents is not None:
        documents = documents[:max_documents]
        kept = {d.doc_id for d in documents}
        # Keep a question only if all its gold passages are kept, so recall stays comparable.
        queries = [q for q in queries if all(d in kept for d, _, _ in q.answer_locations)]
    if max_questions is not None and max_questions < len(queries):
        queries = random.Random(seed).sample(queries, max_questions)
    return EvalDataset(documents=documents, queries=queries)
