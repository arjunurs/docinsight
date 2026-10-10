"""Offline tests for the evaluation harness (no network, no API key)."""
import hashlib
import math
from types import SimpleNamespace

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding

from evaluation.chunking import chunk_documents, label_relevant_chunks
from evaluation.dataset import build_dataset
from evaluation.faithfulness import judge_faithfulness, parse_judge_output
from evaluation.metrics import answer_recall, reciprocal_rank, recall_at_k, token_f1
from evaluation.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    reciprocal_rank_fusion,
)

RAW = {
    "data": [
        {
            "title": "Solar_System",
            "paragraphs": [
                {
                    "context": "Jupiter is the largest planet in the Solar System.",
                    "qas": [
                        {
                            "id": "q1",
                            "question": "Which is the largest planet?",
                            "answers": [{"text": "Jupiter", "answer_start": 0}],
                        }
                    ],
                },
                {
                    "context": "Mercury is the planet closest to the Sun.",
                    "qas": [
                        {
                            "id": "q2",
                            "question": "Which planet is closest to the Sun?",
                            "answers": [
                                {"text": "Mercury", "answer_start": 0},
                                {"text": "Mercury", "answer_start": 0},
                            ],
                        }
                    ],
                },
            ],
        },
        {
            "title": "Chemistry",
            "paragraphs": [
                {
                    "context": "Water is made of hydrogen and oxygen.",
                    "qas": [
                        {
                            "id": "q3",
                            "question": "What elements make up water?",
                            "answers": [{"text": "hydrogen and oxygen", "answer_start": 17}],
                        }
                    ],
                }
            ],
        },
    ]
}


class HashingEmbedding(BaseEmbedding):
    """Deterministic bag-of-words embedding so dense retrieval runs offline."""

    dim: int = 64

    def _vector(self, text: str):
        vec = [0.0] * self.dim
        for token in text.lower().replace("?", " ").replace(".", " ").split():
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _get_query_embedding(self, query):
        return self._vector(query)

    async def _aget_query_embedding(self, query):
        return self._vector(query)

    def _get_text_embedding(self, text):
        return self._vector(text)


@pytest.fixture
def dataset():
    return build_dataset(RAW)


def test_answer_spans_are_global_document_offsets(dataset):
    docs = {d.doc_id: d for d in dataset.documents}
    for query in dataset.queries:
        text = docs[query.doc_id].text
        for start, end in query.answer_spans:
            assert text[start:end] in query.answers
    # Duplicate annotator answers collapse to one span.
    assert len(next(q for q in dataset.queries if q.query_id == "q2").answer_spans) == 1


def test_relevance_labels_follow_chunk_offsets(dataset):
    chunks = chunk_documents(dataset.documents, chunk_size=96, chunk_overlap=0)
    relevant = label_relevant_chunks(dataset.queries, chunks)
    by_id = {c.chunk_id: c for c in chunks}
    for query in dataset.queries:
        assert relevant[query.query_id], query.query_id
        for chunk_id in relevant[query.query_id]:
            assert query.answers[0] in by_id[chunk_id].text


def test_ranking_metrics():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"b"}, 1) == 0.0
    assert recall_at_k(ranked, {"b"}, 2) == 1.0
    assert reciprocal_rank(ranked, {"c"}, 3) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"c"}, 2) == 0.0
    assert reciprocal_rank(ranked, {"z"}, 3) == 0.0


def test_answer_metrics():
    assert token_f1("The Jupiter", ["Jupiter"]) == 1.0
    assert token_f1("Saturn", ["Jupiter"]) == 0.0
    assert answer_recall("The largest planet is Jupiter.", ["Jupiter"]) == 1.0
    assert answer_recall("Jupiterian moons", ["Jupiter"]) == 0.0
    assert answer_recall("anything", ["the"]) == 0.0


def test_rrf_prefers_documents_ranked_well_by_both():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=4)
    assert set(fused[:2]) == {"a", "b"}
    assert fused[2:] == ["c", "d"]


def test_bm25_dense_and_hybrid_retrieve_the_answer_chunk(dataset):
    chunks = chunk_documents(dataset.documents, chunk_size=96, chunk_overlap=0)
    relevant = label_relevant_chunks(dataset.queries, chunks)
    questions = [q.question for q in dataset.queries]

    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, embed_model=HashingEmbedding())
    hybrid = HybridRetriever([bm25, dense])
    for retriever in (bm25, dense, hybrid):
        rankings = retriever.retrieve_batch(questions, k=2)
        hits = [recall_at_k(r, relevant[q.query_id], 2) for r, q in zip(rankings, dataset.queries)]
        assert sum(hits) == len(hits), retriever.name


def test_parse_judge_output_handles_code_fences():
    text = '```json\n{"claims": [{"claim": "Jupiter is largest", "supported": true}]}\n```'
    assert parse_judge_output(text) == [{"claim": "Jupiter is largest", "supported": True}]


def test_judge_faithfulness_scores_supported_fraction():
    reply = (
        '{"claims": [{"claim": "Jupiter is the largest planet", "supported": true},'
        ' {"claim": "Jupiter has 3 moons", "supported": false}]}'
    )
    llm = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=reply))
    result = judge_faithfulness(llm, "q", "a", ["ctx"])
    assert result.score == 0.5

    empty = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text='{"claims": []}'))
    assert judge_faithfulness(empty, "q", "I don't know", ["ctx"]).score is None


@pytest.mark.parametrize(
    "reply",
    [
        '{"claims": [{"claim": "Jupiter has 3 moons", "supported": "false"}]}',
        '{"verdict": "fine"}',
        "no json here",
    ],
)
def test_malformed_judge_replies_are_rejected_not_scored(reply):
    with pytest.raises(ValueError):
        parse_judge_output(reply)
    llm = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=reply))
    result = judge_faithfulness(llm, "q", "a", ["ctx"])
    assert result.score is None and result.invalid


def test_judge_retry_recovers_from_one_bad_reply():
    replies = iter(["not json", '{"claims": [{"claim": "x", "supported": true}]}'])
    llm = SimpleNamespace(complete=lambda prompt: SimpleNamespace(text=next(replies)))
    result = judge_faithfulness(llm, "q", "a", ["ctx"])
    assert result.score == 1.0 and not result.invalid


@pytest.mark.parametrize(
    "argv",
    [["--retrievers", ""], ["--ks", "0,3"], ["--max-questions", "0"], ["--max-articles", "-1"]],
)
def test_cli_rejects_invalid_values(argv):
    from evaluation.run import parse_args

    with pytest.raises(SystemExit):
        parse_args(argv)


def test_faithfulness_summary_is_strict_json_when_nothing_is_scored(dataset):
    import json
    from types import SimpleNamespace as NS

    from llama_index.core.llms import MockLLM

    from evaluation.run import evaluate_faithfulness

    chunks = chunk_documents(dataset.documents, chunk_size=96, chunk_overlap=0)
    relevant = label_relevant_chunks(dataset.queries, chunks)
    dense = DenseRetriever(chunks, embed_model=HashingEmbedding())
    args = NS(llm_model="m", temperature=0.0, judge_model="j", faithfulness=2, seed=13, retrieval_mode="hybrid")
    answer_llm = MockLLM()
    no_claims = NS(complete=lambda prompt: NS(text='{"claims": []}'))

    result = evaluate_faithfulness(dense, dataset.queries, relevant, args, answer_llm, no_claims)
    summary = result["summary"]
    assert summary["faithfulness"] is None and summary["fully_faithful_rate"] is None
    json.dumps(summary, allow_nan=False)
