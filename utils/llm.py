"""
Answer-model factory for OpenAI and Claude models, plus token usage and cost tracking.

The model name picks the provider: names starting with "claude-" go to Anthropic
(needs ANTHROPIC_API_KEY), everything else to OpenAI. Embeddings stay on OpenAI
either way, so switching the answer model never changes retrieval.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llama_index.core.base.llms.types import LLMMetadata
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent, LLMCompletionEndEvent
from llama_index.core.llms import LLM

# USD per 1M tokens (input, output), standard tier, checked 2026-10-11 against
# developers.openai.com/api/docs/models and platform.claude.com/docs/en/about-claude/pricing.
# Claude Haiku 5.5 is listed at its rate for prompts up to 100K tokens; ours are a few thousand.
PRICES: Dict[str, tuple] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
}

# Every current model offers at least a 1M-token window; only used when the installed
# LlamaIndex integration predates the model and can't look its window up.
FALLBACK_CONTEXT_WINDOW = 1_000_000
# Room for Claude's adaptive thinking plus a short grounded answer
CLAUDE_MAX_TOKENS = 4096


def is_claude(model: str) -> bool:
    return model.startswith("claude-")


def is_openai_reasoning(model: str) -> bool:
    """GPT-5 and later, and the o-series, take reasoning_effort and run at temperature 1."""
    return bool(re.match(r"^(gpt-([5-9]|\d{2})|o\d)", model))


def build_llm(model: str, temperature: float = 0.0, reasoning_effort: str = "none") -> LLM:
    """
    Build the answer LLM.

    reasoning_effort "none" asks for a direct answer with no thinking, as far as the model
    allows: OpenAI GPT-6 Luna takes it as is; Claude Haiku 5.5 disables thinking, Claude
    Sonnet 5.5 thinks only between tool calls (it has none here), and models where thinking
    can't be turned off (Claude Opus 5.5, GPT-6.1 Sol) should be given "low" instead.
    Temperature only reaches older models; current OpenAI and Claude models reject it.
    """
    if is_claude(model):
        return _claude(model, reasoning_effort)
    return _OpenAI(model=model, temperature=temperature, reasoning_effort=reasoning_effort or None)


def _claude(model: str, reasoning_effort: str) -> LLM:
    try:
        from llama_index.llms.anthropic import Anthropic
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("Claude models need llama-index-llms-anthropic (see pyproject.toml)") from err

    class Claude(Anthropic):
        @property
        def metadata(self) -> LLMMetadata:
            try:
                return super().metadata
            except ValueError:  # a model newer than the installed integration
                return LLMMetadata(
                    context_window=FALLBACK_CONTEXT_WINDOW,
                    num_output=self.max_tokens,
                    is_chat_model=True,
                    model_name=self.model,
                    is_function_calling_model=True,
                )

        @property
        def _model_kwargs(self) -> Dict[str, Any]:
            kwargs = dict(super()._model_kwargs)
            # Current Claude models reject non-default sampling parameters
            kwargs.pop("temperature", None)
            return kwargs

    effort = reasoning_effort if reasoning_effort and reasoning_effort != "none" else "low"
    thinking = None
    if reasoning_effort in ("", "none"):
        if model.startswith("claude-haiku-5-5"):
            thinking = {"type": "disabled"}
        elif model.startswith("claude-sonnet-5-5"):
            thinking = {"type": "between_tools"}
    return Claude(
        model=model,
        max_tokens=CLAUDE_MAX_TOKENS,
        thinking_dict=thinking,
        additional_kwargs={"output_config": {"effort": effort}},
    )


def _openai_class():
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.openai.utils import O1_MODELS

    class OpenAIAnyModel(OpenAI):
        """OpenAI, tolerant of models newer than the installed integration."""

        @property
        def metadata(self) -> LLMMetadata:
            try:
                return super().metadata
            except ValueError:
                return LLMMetadata(
                    context_window=FALLBACK_CONTEXT_WINDOW,
                    num_output=self.max_tokens or -1,
                    is_chat_model=True,
                    is_function_calling_model=True,
                    model_name=self.model,
                )

        def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
            all_kwargs = super()._get_model_kwargs(**kwargs)
            # The integration only sends reasoning_effort to models it already knows
            if self.model not in O1_MODELS and is_openai_reasoning(self.model):
                all_kwargs["temperature"] = 1.0
                if self.reasoning_effort:
                    all_kwargs["reasoning_effort"] = self.reasoning_effort
            return all_kwargs

    return OpenAIAnyModel


def _OpenAI(**kwargs: Any) -> LLM:
    return _openai_class()(**kwargs)


@dataclass
class CallUsage:
    input_tokens: int = 0
    output_tokens: int = 0


def _field(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def usage_from_raw(raw: Any) -> Optional[CallUsage]:
    """
    Token usage from a provider response (OpenAI ChatCompletion or Anthropic Message).
    Output tokens include reasoning or thinking tokens, which both providers bill as output.
    Cached input tokens are counted as plain input, so cost_usd is an uncached list-price
    estimate (the app sets no cache breakpoints).
    """
    usage = _field(raw, "usage")
    if usage is None:
        return None
    if _field(usage, "prompt_tokens") is not None:  # OpenAI
        return CallUsage(_field(usage, "prompt_tokens") or 0, _field(usage, "completion_tokens") or 0)
    input_tokens = sum(
        _field(usage, key) or 0
        for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    return CallUsage(input_tokens, _field(usage, "output_tokens") or 0)


def cost_usd(model: str, usage: CallUsage) -> Optional[float]:
    price = PRICES.get(model)
    if price is None:
        return None
    return (usage.input_tokens * price[0] + usage.output_tokens * price[1]) / 1_000_000


class UsageRecorder(BaseEventHandler):
    """
    Records token usage of every LLM call made while active.

        with recorder.recording() as calls:
            engine.query(question)
        # calls: one CallUsage per LLM call
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._lock = threading.Lock()
        self._calls: Optional[List[CallUsage]] = None
        self._seen: List[Any] = []

    @classmethod
    def class_name(cls) -> str:
        return "UsageRecorder"

    def handle(self, event: Any, **kwargs: Any) -> None:
        if not isinstance(event, (LLMChatEndEvent, LLMCompletionEndEvent)):
            return
        raw = _field(event.response, "raw")
        with self._lock:
            # A completion call on a chat model emits both events for the same response
            if self._calls is None or raw is None or any(raw is seen for seen in self._seen):
                return
            usage = usage_from_raw(raw)
            if usage is not None:
                self._seen.append(raw)
                self._calls.append(usage)

    def recording(self):
        recorder = self

        class _Scope:
            def __enter__(self) -> List[CallUsage]:
                with recorder._lock:
                    recorder._calls, recorder._seen = [], []
                return recorder._calls

            def __exit__(self, *exc: Any) -> None:
                with recorder._lock:
                    recorder._calls = None

        return _Scope()

    def attach(self) -> "UsageRecorder":
        dispatcher = get_dispatcher()
        if self not in dispatcher.event_handlers:
            dispatcher.add_event_handler(self)
        return self

    def detach(self) -> None:
        dispatcher = get_dispatcher()
        if self in dispatcher.event_handlers:
            dispatcher.event_handlers.remove(self)
