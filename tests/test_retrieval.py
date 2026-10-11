"""
Hybrid retrieval tests: fused ranking, BM25 built from the persisted collection, restarts and new uploads.

Embeddings are mocked (every chunk gets the same vector), so dense search alone
cannot tell chunks apart and any correct ranking has to come from BM25.
"""
import math

import pytest
from llama_index.core import QueryBundle
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode

import utils.index_manager as index_manager_module
from utils.config import AppConfig
from utils.document_processor import DocumentProcessor
from utils.index_manager import IndexManager
from utils.query_engine import QueryEngine
from utils.retrieval import BM25Index, HybridRetriever, reciprocal_rank_fusion

TOPICS = {
    "volcanoes.txt": "Basalt lava flows from shield volcanoes such as Mauna Loa.",
    "tides.txt": "The Moon's gravity raises ocean tides twice a day.",
    "bees.txt": "Honeybees communicate the direction of nectar with a waggle dance.",
    "glaciers.txt": "Glaciers carve U-shaped valleys as the ice slowly advances.",
    "comets.txt": "A comet's tail always points away from the Sun because of the solar wind.",
}


@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        index_manager_module, "OpenAIEmbedding", lambda model: MockEmbedding(embed_dim=8)
    )


def write_files(directory, topics):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in topics.items():
        path = directory / name
        path.write_text(text)
        paths.append(str(path))
    return paths


def ingest(manager, paths):
    processor = DocumentProcessor(chunk_size=128, chunk_overlap=16)
    return manager.add_nodes(processor.process_documents(processor.load_documents(paths)))


def make_manager(tmp_path):
    return IndexManager(persist_dir=str(tmp_path / "chroma"), collection_name="test")


def file_names(results):
    return [r.node.metadata["file_name"] for r in results]


def test_rrf_ranks_agreement_above_a_single_first_place():
    # "b" is second in both lists; "a" and "c" are each first in only one
    assert reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=3)[0] == "b"


def fake_dense(monkeypatch, retriever, order):
    """Make dense search return the stored chunks in a fixed order of file names."""
    stored = {n.metadata["file_name"]: n for n in retriever._bm25_index().nodes}

    class FakeDense:
        def retrieve(self, query_bundle):
            # Scores fall with rank, in Chroma's exp(-distance) form
            return [NodeWithScore(node=stored[name], score=math.exp(-0.1 * i)) for i, name in enumerate(order)]

    monkeypatch.setattr(retriever.index, "as_retriever", lambda similarity_top_k: FakeDense())


def bm25_top(retriever, query):
    """File name of BM25's best chunk for a query."""
    bm25 = retriever._bm25_index()
    best = bm25.rank(query, 1)[0]
    return next(n.metadata["file_name"] for n in bm25.nodes if n.node_id == best)


def test_hybrid_fuses_dense_and_bm25_rankings(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload", TOPICS))
    retriever = HybridRetriever(manager.get_index(), similarity_top_k=3)
    query = "Which way does the tail of a comet point from the Sun?"
    assert bm25_top(retriever, query) == "comets.txt"

    # Dense puts tides first and comets second; BM25 puts comets first
    fake_dense(monkeypatch, retriever, ["tides.txt", "comets.txt", "bees.txt", "glaciers.txt", "volcanoes.txt"])
    results = retriever.retrieve(query)

    # comets (2nd and 1st) beats tides (1st by dense only)
    assert file_names(results)[0] == "comets.txt"
    assert len(results) == 3
    assert results[0].score > results[1].score


def test_hybrid_rescues_a_keyword_match_dense_ranks_too_low(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload", TOPICS))
    retriever = HybridRetriever(manager.get_index(), similarity_top_k=3)
    query = "How do honeybees signal where nectar is?"
    assert bm25_top(retriever, query) == "bees.txt"

    # Dense alone would leave bees out of the top 3 sent to the LLM
    dense_order = ["tides.txt", "comets.txt", "glaciers.txt", "bees.txt", "volcanoes.txt"]
    fake_dense(monkeypatch, retriever, dense_order)
    results = retriever.retrieve(query)

    assert "bees.txt" not in dense_order[:3]
    assert "bees.txt" in file_names(results)
    # Retrieved nodes still hide bookkeeping from the LLM
    bees = next(r.node for r in results if r.node.metadata["file_name"] == "bees.txt")
    assert bees.get_metadata_str(MetadataMode.LLM) == "file_name: bees.txt"


def test_bm25_side_is_rebuilt_from_the_persisted_collection_after_restart(tmp_path):
    ingest(make_manager(tmp_path), write_files(tmp_path / "upload", TOPICS))

    # A fresh manager is what the app builds on restart; nothing but Chroma is persisted
    restarted = make_manager(tmp_path)
    retriever = HybridRetriever(restarted.get_index(), similarity_top_k=3)

    assert len(retriever._bm25_index().nodes) == restarted.chroma_collection.count() == len(TOPICS)
    assert bm25_top(retriever, "glaciers carve valleys") == "glaciers.txt"
    assert len(retriever.retrieve("glaciers carve valleys")) == 3


def test_bm25_side_picks_up_new_uploads(tmp_path):
    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload_a", TOPICS))
    retriever = HybridRetriever(manager.get_index(), similarity_top_k=3)
    retriever.retrieve("lava")

    ingest(manager, write_files(tmp_path / "upload_b", {"tea.txt": "Oolong tea is partly oxidised."}))

    assert bm25_top(retriever, "oolong oxidised") == "tea.txt"


def test_query_engine_uses_hybrid_by_default(tmp_path):
    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload", TOPICS))

    engine = QueryEngine(manager.get_index())

    assert engine.retrieval_mode == AppConfig().retrieval_mode == "hybrid"
    assert isinstance(engine.query_engine.retriever, HybridRetriever)
    assert len(engine.query_engine.retriever.retrieve(QueryBundle("tides and the Moon"))) == 3


def test_query_engine_rejects_unknown_mode():
    with pytest.raises(ValueError):
        QueryEngine(retrieval_mode="sparse")


def test_config_reads_retrieval_mode(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_MODE", "dense")
    assert AppConfig.from_env().retrieval_mode == "dense"


def test_empty_collection_returns_nothing(tmp_path):
    retriever = HybridRetriever(make_manager(tmp_path).get_index(), similarity_top_k=3)
    assert retriever.retrieve("anything") == []


def test_bm25_with_no_matching_words_leaves_the_dense_ranking_alone(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload", TOPICS))
    retriever = HybridRetriever(manager.get_index(), similarity_top_k=3)
    query = "xylophone quokka"  # shares no word with any chunk
    assert retriever._bm25_index().rank(query, 5) == []

    dense_order = ["tides.txt", "comets.txt", "bees.txt", "glaciers.txt", "volcanoes.txt"]
    fake_dense(monkeypatch, retriever, dense_order)
    assert file_names(retriever.retrieve(query)) == dense_order[:3]


def test_bm25_keeps_zero_score_matches_that_sort_after_the_cutoff():
    # "zeta" is in exactly half the chunks, so Okapi IDF gives it a score of 0
    # and the stable sort leaves the non-matching chunks first.
    nodes = [TextNode(text=t, id_=f"n{i}") for i, t in enumerate(["alpha", "beta", "zeta one", "zeta two"])]
    assert BM25Index(nodes).rank("zeta", 2) == ["n2", "n3"]


def test_answer_prompt_requires_grounding_and_a_fixed_no_answer_reply(tmp_path):
    from llama_index.core import Settings
    from llama_index.core.llms import MockLLM

    from utils.prompts import NO_ANSWER

    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload", TOPICS))
    Settings.llm = MockLLM()  # echoes the prompt it receives

    for mode in ("hybrid", "dense"):
        prompt_sent = QueryEngine(manager.get_index(), retrieval_mode=mode).query("tides and the Moon").response
        assert f'reply exactly: "{NO_ANSWER}"' in prompt_sent, mode
        assert "only the context above, not prior knowledge" in prompt_sent, mode
        assert "Query: tides and the Moon" in prompt_sent, mode


def test_no_answer_reply_is_detected():
    from utils.prompts import NO_ANSWER, is_no_answer

    assert is_no_answer(NO_ANSWER)
    assert is_no_answer("I don't know based on the provided documents")
    assert not is_no_answer("The Moon raises tides twice a day.")
    assert is_no_answer('  "I don\u2019t know based on the provided documents."\n')
    assert not is_no_answer(NO_ANSWER + " But the Moon probably raises tides.")
    assert not is_no_answer("I don't know based on the provided documents, but it is likely the Moon.")


def test_refine_prompt_keeps_the_grounding_rule_after_a_no_answer(tmp_path):
    from typing import Any, List

    from llama_index.core import Settings
    from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
    from llama_index.core.llms.callbacks import llm_completion_callback

    from llama_index.core.utils import get_tokenizer

    from utils.prompts import NO_ANSWER, REFINE_PROMPT

    # Room for the refine prompt plus about one chunk, so each chunk gets its own call.
    # LlamaIndex also reserves 500 tokens for the answer when packing chunks.
    empty_refine = REFINE_PROMPT.format(context_msg="", query_str="tides and the Moon", existing_answer=NO_ANSWER)
    window = len(get_tokenizer()(empty_refine)) + 500 + 16 + 40

    class RecordingLLM(CustomLLM):
        """Answers NO_ANSWER every time; a small window forces one call per chunk."""

        prompts: List[str] = []

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(context_window=window, num_output=16)

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
            self.prompts.append(prompt)
            return CompletionResponse(text=NO_ANSWER)

        @llm_completion_callback()
        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
            raise NotImplementedError

    # Three chunks, so the top 3 retrieved are all of them whatever the mock
    # embedding's ties, and the tides chunk always reaches the LLM.
    topics = {name: TOPICS[name] for name in ("volcanoes.txt", "tides.txt", "bees.txt")}
    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload", topics))

    for mode in ("hybrid", "dense"):
        llm = RecordingLLM()
        Settings.llm = llm
        QueryEngine(manager.get_index(), retrieval_mode=mode).query("tides and the Moon")

        refines = [p for p in llm.prompts if "existing answer" in p]
        assert refines, mode
        for prompt in refines:
            assert f"We have provided an existing answer: {NO_ANSWER}" in prompt, mode
            assert "only the context above, not prior knowledge" in prompt, mode
            assert f'reply exactly: "{NO_ANSWER}"' in prompt, mode
        # The chunk that answers the query reaches the LLM in one of the calls
        assert any("raises ocean tides" in p for p in llm.prompts), mode


def test_score_fusion_lets_a_strong_keyword_match_win_and_ignores_weak_ones():
    from utils.retrieval import score_fusion

    dense = {"a": 0.90, "b": 0.88, "c": 0.50}
    # A confident keyword match on "b" lifts it over "a"...
    assert score_fusion(dense, {"a": 0.0, "b": 9.0, "c": 0.0}, k=3, dense_weight=0.7)[0] == "b"
    # ...but small keyword differences leave the dense order alone
    assert score_fusion(dense, {"a": 0.0, "b": 0.1, "c": 1.0}, k=3, dense_weight=0.7) == ["a", "b", "c"]


def test_score_fusion_weights_and_missing_scores():
    from utils.retrieval import fused_scores, score_fusion

    dense, keyword = {"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}
    assert score_fusion(dense, keyword, k=2, dense_weight=1.0) == ["a", "b"]
    assert score_fusion(dense, keyword, k=2, dense_weight=0.0) == ["b", "a"]
    # A keyword-only candidate takes dense's lowest score, since it fell below dense's cutoff
    fused = fused_scores({"a": 1.0, "b": 0.5}, {"a": 0.0, "b": 0.0, "c": 2.0}, dense_weight=0.7)
    assert fused["c"] == pytest.approx(0.3)
    assert fused_scores({}, {}) == {}


def test_chroma_scores_are_converted_back_to_a_linear_scale():
    from utils.retrieval import chroma_score_to_dense

    # LlamaIndex reports exp(-distance); the conversion gives minus the distance back
    assert chroma_score_to_dense(math.exp(-0.4)) == pytest.approx(-0.4)
    assert chroma_score_to_dense(0.0) == -math.inf


def test_hybrid_rejects_bad_fusion_settings(tmp_path):
    index = make_manager(tmp_path).get_index()
    with pytest.raises(ValueError):
        HybridRetriever(index, fusion="max")
    with pytest.raises(ValueError):
        HybridRetriever(index, dense_weight=1.5)


def test_rrf_fusion_is_still_available(tmp_path, monkeypatch):
    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload", TOPICS))
    retriever = HybridRetriever(manager.get_index(), similarity_top_k=3, fusion="rrf")
    fake_dense(monkeypatch, retriever, ["tides.txt", "comets.txt", "bees.txt", "glaciers.txt", "volcanoes.txt"])

    results = retriever.retrieve("Which way does the tail of a comet point from the Sun?")

    # comets is 2nd by dense and 1st by BM25, so it beats tides (1st by dense only)
    assert file_names(results)[0] == "comets.txt"
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_config_reads_fusion_settings(monkeypatch):
    assert (AppConfig().hybrid_fusion, AppConfig().hybrid_dense_weight) == ("score", 0.7)
    monkeypatch.setenv("HYBRID_FUSION", "rrf")
    monkeypatch.setenv("HYBRID_DENSE_WEIGHT", "0.5")
    config = AppConfig.from_env()
    assert (config.hybrid_fusion, config.hybrid_dense_weight) == ("rrf", 0.5)


def test_stored_dense_scores_match_chroma_distances():
    import chromadb
    import chromadb.config

    from utils.retrieval import stored_dense_scores

    collection = chromadb.EphemeralClient(
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    ).create_collection("stored-scores")
    collection.add(ids=["a", "b", "c"], embeddings=[[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    query = [1.0, 0.0]

    scores = stored_dense_scores(collection, query, ["c", "a"])
    hits = collection.query(query_embeddings=[query], n_results=3)

    # Same scale as search hits: minus Chroma's own squared L2 distance
    by_id = dict(zip(hits["ids"][0], hits["distances"][0]))
    assert scores == {"a": pytest.approx(-by_id["a"]), "c": pytest.approx(-by_id["c"])}
    assert stored_dense_scores(collection, query, []) == {}


def test_bm25_only_candidates_get_exact_dense_scores_from_the_query_embedding(tmp_path, monkeypatch):
    import utils.retrieval as retrieval

    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload", TOPICS))
    retriever = HybridRetriever(manager.get_index(), similarity_top_k=3)
    stored = {n.metadata["file_name"]: n for n in retriever._bm25_index().nodes}

    class OneHitDense:
        def retrieve(self, query_bundle):
            query_bundle.embedding = [0.1] * 8  # dense search embeds the query into the bundle
            return [NodeWithScore(node=stored["tides.txt"], score=math.exp(-0.5))]

    monkeypatch.setattr(retriever.index, "as_retriever", lambda similarity_top_k: OneHitDense())
    calls = []
    real = retrieval.stored_dense_scores
    monkeypatch.setattr(
        retrieval, "stored_dense_scores", lambda c, emb, ids: calls.append((emb, list(ids))) or real(c, emb, ids)
    )

    retriever.retrieve("glaciers carve valleys")

    # Scored once, from the bundle's embedding (no second embedding call), for BM25-only chunks
    assert len(calls) == 1
    embedding, ids = calls[0]
    assert embedding == [0.1] * 8
    assert stored["glaciers.txt"].node_id in ids and stored["tides.txt"].node_id not in ids
