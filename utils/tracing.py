"""
Per-query tracing: an OpenTelemetry span for each stage of answering a question
(query embedding, dense search, BM25, fusion, reranking, the LLM answer) with its
latency, plus token counts and estimated cost for the answer.

Spans reach an exporter only when one is configured (TRACING):
  console  print each finished span to stdout
  otlp     send spans over OTLP/gRPC to OTEL_EXPORTER_OTLP_ENDPOINT (default localhost:4317)
  unset    otlp when OTEL_EXPORTER_OTLP_ENDPOINT is set, otherwise off
Off, the spans are OpenTelemetry's no-op spans and cost next to nothing. Either way
every query also builds a QueryTrace (stage timings, tokens, $) that the app shows
under the answer and the evaluation harness aggregates.

Attribute names follow the OpenTelemetry GenAI semantic conventions where one
exists (gen_ai.*); the rest are prefixed docinsight.*. The question text is only
recorded when OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true, as those
conventions ask.
"""
from __future__ import annotations

import contextlib
import contextvars
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from opentelemetry import trace

from utils.llm import CallUsage, UsageRecorder, cost_usd

TRACING_MODES = ("off", "console", "otlp")
ROOT_SPAN = "docinsight.query"

_tracer = trace.get_tracer("docinsight")
_current: contextvars.ContextVar[Optional["QueryTrace"]] = contextvars.ContextVar("query_trace", default=None)
_configure_lock = threading.Lock()
_configured_mode: Optional[str] = None
_usage = UsageRecorder()


@dataclass
class QueryTrace:
    """What one query spent: wall clock per stage, LLM calls, tokens and cost."""

    model: Optional[str] = None
    # Stage name to milliseconds, in the order stages started; nested stages are
    # listed separately (retrieval includes retrieval.dense, for example)
    stages_ms: Dict[str, float] = field(default_factory=dict)
    llm_calls: List[CallUsage] = field(default_factory=list)
    total_ms: float = 0.0
    trace_id: Optional[str] = None

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @property
    def cost_usd(self) -> Optional[float]:
        """
        Answer-model cost at list prices; None when the model has no price in utils/llm.py
        or no call reported token usage.
        """
        if not self.llm_calls:
            return None
        costs = [cost_usd(self.model, c) if self.model else None for c in self.llm_calls]
        return None if None in costs else sum(costs)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "total_ms": self.total_ms,
            "stages_ms": dict(self.stages_ms),
            "llm_calls": len(self.llm_calls),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "trace_id": self.trace_id,
        }


def tracing_mode(env: Optional[Dict[str, str]] = None) -> str:
    """The exporter TRACING asks for, defaulting to otlp when an OTLP endpoint is configured."""
    env = os.environ if env is None else env
    mode = (env.get("TRACING") or "").strip().lower()
    if not mode:
        return "otlp" if env.get("OTEL_EXPORTER_OTLP_ENDPOINT") or env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") else "off"
    if mode in ("none", "false", "0"):
        return "off"
    if mode not in TRACING_MODES:
        raise ValueError(f"TRACING must be one of {TRACING_MODES}, got {mode!r}")
    return mode


def configure_tracing(mode: Optional[str] = None, exporter: Any = None) -> str:
    """
    Install a tracer provider that exports spans, once per process. Returns the mode in use.

    exporter replaces the one the mode picks (tests pass an in-memory exporter).
    Spans are exported from a background thread (BatchSpanProcessor), so a slow or
    absent collector never adds latency to a query.
    """
    global _configured_mode
    with _configure_lock:
        if _configured_mode is not None:
            return _configured_mode
        mode = tracing_mode() if mode is None else mode
        if mode == "off" and exporter is None:
            _configured_mode = "off"
            return _configured_mode

        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        if exporter is None and mode == "otlp":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter()  # endpoint, headers and TLS from the OTEL_EXPORTER_OTLP_* variables
        elif exporter is None:
            exporter = ConsoleSpanExporter()
        provider = TracerProvider(resource=Resource.create({SERVICE_NAME: os.getenv("OTEL_SERVICE_NAME") or "docinsight"}))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _configured_mode = mode
        return _configured_mode


def _set_attributes(span: trace.Span, attributes: Dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, value)


@contextlib.contextmanager
def stage(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[trace.Span]:
    """
    A child span for one stage of the current query, timed into its QueryTrace.

    Outside a query the span still goes to the exporter (when one is configured)
    but nothing is timed. None-valued attributes are skipped.
    """
    with _tracer.start_as_current_span(name) as span:
        _set_attributes(span, attributes or {})
        query = _current.get()
        if query is not None:
            # Listed in the order stages start, so a parent comes before its children
            query.stages_ms.setdefault(name, 0.0)
        started = time.perf_counter()
        try:
            yield span
        finally:
            if query is not None:
                query.stages_ms[name] += 1000 * (time.perf_counter() - started)


@contextlib.contextmanager
def query_trace(question: str, model: Optional[str] = None) -> Iterator[QueryTrace]:
    """The root span of one query; stages and LLM calls inside it are recorded on the yielded QueryTrace."""
    _usage.attach()
    query = QueryTrace(model=model)
    token = _current.set(query)
    started = time.perf_counter()
    try:
        with _tracer.start_as_current_span(ROOT_SPAN) as span, _usage.recording() as calls:
            # The recorder appends to this list as calls finish, so stages can read it mid-query
            query.llm_calls = calls
            context = span.get_span_context()
            if context.is_valid:
                query.trace_id = format(context.trace_id, "032x")
            _set_attributes(span, {
                "gen_ai.request.model": model,
                "docinsight.query.length": len(question),
                "docinsight.query.text": question if _capture_content() else None,
            })
            try:
                yield query
            finally:
                query.total_ms = 1000 * (time.perf_counter() - started)
                _set_attributes(span, {
                    "gen_ai.usage.input_tokens": query.input_tokens,
                    "gen_ai.usage.output_tokens": query.output_tokens,
                    "docinsight.llm.calls": len(query.llm_calls),
                    "docinsight.cost_usd": query.cost_usd,
                })
    finally:
        _current.reset(token)


def _capture_content() -> bool:
    return os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "").lower() == "true"
