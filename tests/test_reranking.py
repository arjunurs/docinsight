"""
Optional reranker tests. A fake scorer stands in for the cross-encoder, so no
model is downloaded and sentence-transformers does not need to be installed.
"""
import pytest
from llama_index.core import QueryBundle, Settings
from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeWithScore, TextNode

import utils.reranking as reranking
from evaluation.retrievers import RerankRetriever
from tests.test_retrieval import TOPICS, ingest, make_manager, mock_openai, write_files  # noqa: F401
from utils.config import AppConfig
from utils.query_engine import QueryEngine
from utils.reranking import CrossEncoderReranker, build_reranker, reranker_key


def keyword_scorer(word):
    """Fake cross-encoder: a text scores 1 if it mentions the word, else 0."""
    return lambda query, texts: [1.0 if word in text.lower() else 0.0 for text in texts]


def nodes(*texts):
    return [NodeWithScore(node=TextNode(id_=t, text=t), score=1.0) for t in texts]


def test_reranker_reorders_by_score_and_keeps_top_n():
    reranker = CrossEncoderReranker("fake", top_n=2, scorer=keyword_scorer("bee"))

    kept = reranker.postprocess_nodes(nodes("tides", "comets", "bees dance", "glaciers"), QueryBundle("bees"))

    # The bee chunk jumps to the top; the tie keeps the retriever's order
    assert [n.node.node_id for n in kept] == ["bees dance", "tides"]
    assert kept[0].score == 1.0


def test_query_engine_retrieves_candidates_and_sends_top_3_to_the_llm(tmp_path):
    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload", TOPICS))
    Settings.llm = MockLLM()
    reranker = CrossEncoderReranker("fake", scorer=keyword_scorer("glacier"))

    engine = QueryEngine(manager.get_index(), reranker=reranker, rerank_candidates=5)
    response = engine.query("which landform is carved by ice?")

    assert engine.query_engine.retriever.similarity_top_k == 5
    assert len(response.source_nodes) == 3
    assert response.source_nodes[0].node.metadata["file_name"] == "glaciers.txt"


def test_without_a_reranker_nothing_changes(tmp_path):
    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload", TOPICS))

    engine = QueryEngine(manager.get_index())

    assert engine.reranker is None
    assert engine.query_engine.retriever.similarity_top_k == 3
    assert engine.query_engine._node_postprocessors == []


@pytest.mark.parametrize("off", [None, "", "none", "OFF", " "])
def test_reranking_is_off_unless_named(off):
    assert reranker_key(off) == ""
    assert build_reranker(off) is None


def test_unknown_reranker_is_rejected():
    with pytest.raises(ValueError):
        build_reranker("bigmodel")


def test_missing_extra_fails_at_startup(monkeypatch):
    monkeypatch.setattr(reranking.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="requirements-rerank.txt"):
        build_reranker("minilm")


def test_config_reads_reranker_settings(monkeypatch):
    assert AppConfig().reranker == ""
    monkeypatch.setenv("RERANKER", "minilm")
    monkeypatch.setenv("RERANK_CANDIDATES", "20")
    config = AppConfig.from_env()
    assert (config.reranker, config.rerank_candidates) == ("minilm", 20)


def test_harness_rerank_retriever_reorders_only_the_candidates():
    class FixedBase:
        def retrieve_batch(self, questions, k):
            return [["a", "b", "c", "d"][:k] for _ in questions]

    class FakeChunk:
        def __init__(self, chunk_id, text):
            self.chunk_id = chunk_id
            self.node = TextNode(text=text)

    chunks = [FakeChunk(cid, text) for cid, text in [("a", "x"), ("b", "x"), ("c", "match"), ("d", "match")]]
    reranker = CrossEncoderReranker("fake", scorer=keyword_scorer("match"))

    ranked = RerankRetriever(FixedBase(), chunks, reranker, candidates=3).retrieve_batch(["q"], k=4)

    # "c" is promoted within the 3 candidates; "d", past the cut, keeps its place
    assert ranked == [["c", "a", "b", "d"]]
