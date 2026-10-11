"""
Per-query tracing tests: the span tree a query produces, its token and cost
attributes, and the QueryTrace the app shows. Spans go to an in-memory exporter.
"""
import threading
from typing import Any

import pytest
from llama_index.core import Settings
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.test_retrieval import TOPICS, ingest, make_manager, mock_openai, write_files  # noqa: F401
from utils.llm import UsageRecorder
from utils.query_engine import QueryEngine
from utils.reranking import CrossEncoderReranker
from utils.tracing import ROOT_SPAN, configure_tracing, tracing_mode

EXPORTER = InMemorySpanExporter()


class UsageLLM(CustomLLM):
    """Answers every prompt with a fixed reply and reports Anthropic-style usage."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="claude-haiku-5-5")

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text="Twice a day.", raw={"usage": {"input_tokens": 1000, "output_tokens": 20}})

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        raise NotImplementedError


@pytest.fixture
def spans(tmp_path):
    # One provider per process (OpenTelemetry allows setting it once); clear its spans per test
    configure_tracing(exporter=EXPORTER)
    EXPORTER.clear()

    def finished():
        trace.get_tracer_provider().force_flush()
        return {span.name: span for span in EXPORTER.get_finished_spans()}

    return finished


@pytest.fixture
def manager(tmp_path):
    manager = make_manager(tmp_path)
    ingest(manager, write_files(tmp_path / "upload", TOPICS))
    Settings.llm = UsageLLM()
    return manager


def test_hybrid_query_records_a_span_per_stage_with_tokens_and_cost(manager, spans):
    response, query = QueryEngine(manager.get_index()).query_with_trace("tides and the Moon")
    finished = spans()

    assert response.response == "Twice a day."
    stages = ["retrieval", "retrieval.embed_query", "retrieval.dense", "retrieval.bm25", "retrieval.fusion",
              "llm.answer"]
    assert set(finished) == {ROOT_SPAN, *stages}
    assert list(query.stages_ms) == stages
    root = finished[ROOT_SPAN]
    # One trace, with the stages nested under retrieval or the root
    assert {span.context.trace_id for span in finished.values()} == {root.context.trace_id}
    assert query.trace_id == format(root.context.trace_id, "032x")
    assert finished["retrieval.dense"].parent.span_id == finished["retrieval"].context.span_id
    assert finished["llm.answer"].parent.span_id == root.context.span_id

    expected_cost = (1000 * 0.10 + 20 * 0.50) / 1e6
    assert (query.input_tokens, query.output_tokens, len(query.llm_calls)) == (1000, 20, 1)
    assert query.cost_usd == pytest.approx(expected_cost)
    for span in (root, finished["llm.answer"]):
        assert span.attributes["gen_ai.usage.input_tokens"] == 1000
        assert span.attributes["gen_ai.usage.output_tokens"] == 20
        assert span.attributes["docinsight.cost_usd"] == pytest.approx(expected_cost)
    assert finished["llm.answer"].attributes["gen_ai.request.model"] == "claude-haiku-5-5"
    assert finished["llm.answer"].attributes["gen_ai.provider.name"] == "anthropic"
    assert finished["retrieval.fusion"].attributes["docinsight.fusion.mode"] == "score"
    assert finished["retrieval.bm25"].attributes["docinsight.bm25.rebuilt"] is True
    assert query.total_ms >= sum(query.stages_ms[s] for s in ("retrieval", "llm.answer"))
    # Question text stays out of spans unless content capture is turned on
    assert "docinsight.query.text" not in root.attributes


def test_dense_query_with_reranker_has_no_bm25_or_fusion_spans(manager, spans, monkeypatch):
    monkeypatch.setenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true")
    reranker = CrossEncoderReranker("fake", scorer=lambda q, texts: [float("tides" in t) for t in texts])
    engine = QueryEngine(manager.get_index(), retrieval_mode="dense", reranker=reranker, rerank_candidates=5)

    _, query = engine.query_with_trace("tides and the Moon")
    finished = spans()

    assert set(finished) == {ROOT_SPAN, "retrieval", "retrieval.embed_query", "retrieval.dense", "rerank",
                             "llm.answer"}
    assert finished["rerank"].attributes["docinsight.rerank.candidates"] == 5
    assert finished["rerank"].attributes["docinsight.rerank.top_n"] == 3
    assert finished["llm.answer"].attributes["docinsight.llm.chunks"] == 3
    assert finished[ROOT_SPAN].attributes["docinsight.query.text"] == "tides and the Moon"


def test_query_still_returns_the_plain_response(manager, spans):
    assert QueryEngine(manager.get_index()).query("tides and the Moon").response == "Twice a day."


def test_usage_recordings_in_different_threads_do_not_mix():
    recorder = UsageRecorder().attach()
    llm = UsageLLM()
    results = {}
    started = threading.Barrier(2)

    def run(name, n):
        with recorder.recording() as calls:
            started.wait()
            for _ in range(n):
                llm.complete("question")
        results[name] = len(calls)

    threads = [threading.Thread(target=run, args=(name, n)) for name, n in (("a", 1), ("b", 3))]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        recorder.detach()
    assert results == {"a": 1, "b": 3}


def test_tracing_mode_defaults_to_otlp_only_when_an_endpoint_is_set():
    assert tracing_mode({}) == "off"
    assert tracing_mode({"OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317"}) == "otlp"
    assert tracing_mode({"TRACING": "Console", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://x:4317"}) == "console"
    assert tracing_mode({"TRACING": "off", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://x:4317"}) == "off"
    with pytest.raises(ValueError):
        tracing_mode({"TRACING": "jaeger"})
