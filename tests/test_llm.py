"""
Answer-model factory: the request each provider receives, usage and cost tracking.
"""
import json

import anthropic
import httpx
import httpx2
import pytest
from llama_index.core.llms import ChatMessage

from utils.llm import UsageRecorder, build_llm, cost_usd, usage_from_raw


@pytest.fixture(autouse=True)
def dummy_api_keys(monkeypatch):
    # The wrappers don't need keys until the first request, but set them so no test
    # depends on that or picks up a developer's real keys
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")


def openai_request(model, effort):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Paris"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020},
        })

    llm = build_llm(model, 0.0, effort)
    llm.api_key = "test"
    llm._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    response = llm.chat([ChatMessage(role="user", content="Capital of France?")])
    return llm, response, bodies[0]


def claude_request(model, effort):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "m", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "Paris"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 900, "output_tokens": 20,
                      "cache_read_input_tokens": 100, "cache_creation_input_tokens": 0},
        })

    llm = build_llm(model, 0.0, effort)
    llm._client = anthropic.Anthropic(
        api_key="test", http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    )
    response = llm.chat([ChatMessage(role="user", content="Capital of France?")])
    return llm, response, bodies[0]


@pytest.mark.parametrize("model, effort", [("gpt-6-luna", "none"), ("gpt-6.1-sol", "low")])
def test_current_openai_models_get_reasoning_effort_and_no_temperature_zero(model, effort):
    llm, response, body = openai_request(model, effort)
    assert response.message.content == "Paris"
    assert body["reasoning_effort"] == effort
    # Reasoning models only take the default temperature
    assert body["temperature"] == 1.0
    # gpt-6.1-sol is newer than the installed integration; its window still resolves
    assert llm.metadata.context_window >= 1_000_000


def test_older_openai_models_keep_temperature_zero():
    _, _, body = openai_request("gpt-4o-mini", "none")
    assert body["temperature"] == 0.0
    assert "reasoning_effort" not in body


@pytest.mark.parametrize("model, effort, thinking", [
    ("claude-haiku-5-5", "none", {"type": "disabled"}),
    ("claude-sonnet-5-5", "none", {"type": "between_tools"}),
    ("claude-opus-5-5", "low", None),  # thinking can't be turned off; low effort keeps it short
])
def test_claude_requests(model, effort, thinking):
    llm, response, body = claude_request(model, effort)
    # Thinking blocks never leak into the answer
    assert response.message.content == "Paris"
    assert body["model"] == model
    assert body.get("thinking") == thinking
    assert body["output_config"] == {"effort": "low"}
    # Current Claude models reject non-default sampling parameters
    assert "temperature" not in body
    assert llm.metadata.context_window >= 1_000_000


def test_usage_and_cost_from_both_providers():
    _, openai_response, _ = openai_request("gpt-6-luna", "none")
    usage = usage_from_raw(openai_response.raw)
    assert (usage.input_tokens, usage.output_tokens) == (1000, 20)
    assert cost_usd("gpt-6-luna", usage) == pytest.approx((1000 * 0.10 + 20 * 0.50) / 1e6)

    _, claude_response, _ = claude_request("claude-haiku-5-5", "none")
    usage = usage_from_raw(claude_response.raw)
    # Cache reads count as input
    assert (usage.input_tokens, usage.output_tokens) == (1000, 20)
    assert cost_usd("unpriced-model", usage) is None


def test_usage_recorder_counts_each_call_once_and_only_while_recording():
    llm = build_llm("gpt-6-luna", 0.0, "none")
    llm.api_key = "test"
    llm._http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        "id": "c", "object": "chat.completion", "created": 0, "model": "gpt-6-luna",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Paris"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    })))
    recorder = UsageRecorder().attach()
    try:
        llm.complete("outside")  # not recorded
        with recorder.recording() as calls:
            llm.complete("one")  # a completion on a chat model emits two end events
            llm.chat([ChatMessage(role="user", content="two")])
        llm.complete("after")
    finally:
        recorder.detach()
    assert [(c.input_tokens, c.output_tokens) for c in calls] == [(10, 2), (10, 2)]
