import json

import httpx
import pytest

from llm_verifier.async_adapters import (
    AsyncVerifierAdapter,
    DeepSeekAsyncAdapter,
    GeminiAsyncAdapter,
    OpenAICompatibleAsyncAdapter,
)
from llm_verifier.fine_grained_reward import call_verifier_async


OPENAI_RESPONSE = {
    "choices": [{
        "message": {"content": "analysis <score_A> A </score_A>"},
        "logprobs": {"content": [{
            "token": " A",
            "logprob": -0.1,
            "top_logprobs": [
                {"token": " A", "logprob": -0.1},
                {"token": " B", "logprob": -2.0},
            ],
        }]},
    }],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 3},
    },
}


@pytest.mark.asyncio
async def test_openai_adapter_posts_contract_and_normalizes_logprobs():
    """Catches a wrong OpenAI URL, auth/payload, or logprob mapping."""
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=OPENAI_RESPONSE)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OpenAICompatibleAsyncAdapter(
        base_url="https://verifier.test/v1", api_key="test-key", client=client
    )
    try:
        result = await adapter.complete("rate this", model="test-model")
    finally:
        await client.aclose()

    assert result.text == "analysis <score_A> A </score_A>"
    assert result.tokens == [" A"]
    assert result.position_logprobs == [[(" A", -0.1), (" B", -2.0)]]
    assert result.usage == (12, 3, 5, 0)
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://verifier.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "rate this"}],
        "max_tokens": 4096,
        "temperature": 1.0,
        "logprobs": True,
        "top_logprobs": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


@pytest.mark.asyncio
async def test_deepseek_adapter_uses_reasoning_request_contract():
    """Catches omission of DeepSeek's thinking settings or output budget."""
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=OPENAI_RESPONSE)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = DeepSeekAsyncAdapter(
        api_key="deepseek-key",
        model="deepseek-test",
        reasoning_effort="low",
        max_tokens=777,
        client=client,
    )
    try:
        result = await adapter.complete("rate this", model="deepseek-test")
    finally:
        await client.aclose()

    assert result.text.startswith("analysis")
    request_body = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://api.deepseek.com/chat/completions"
    assert request_body["model"] == "deepseek-test"
    assert request_body["max_tokens"] == 777
    assert request_body["thinking"] == {"type": "enabled"}
    assert request_body["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_gemini_adapter_posts_vertex_contract_and_normalizes_logprobs():
    """Catches a wrong Vertex REST shape or Gemini camelCase response mapping."""
    requests = []
    gemini_response = {
        "candidates": [{
            "content": {"parts": [{"text": "analysis <score_A> A </score_A>"}]},
            "logprobsResult": {
                "topCandidates": [{"candidates": [
                    {"token": " A", "logProbability": -0.1},
                    {"token": " B", "logProbability": -2.0},
                ]}],
                "chosenCandidates": [{"token": " A"}],
            },
        }],
        "usageMetadata": {
            "promptTokenCount": 9,
            "cachedContentTokenCount": 2,
            "candidatesTokenCount": 4,
            "thoughtsTokenCount": 6,
        },
    }

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=gemini_response)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = GeminiAsyncAdapter(
        api_key="vertex-key", base_url="https://vertex.test/v1", client=client
    )
    try:
        result = await adapter.complete("rate this", model="gemini-test", top_logprobs=7)
    finally:
        await client.aclose()

    assert result.text == "analysis <score_A> A </score_A>"
    assert result.tokens == [" A"]
    assert result.position_logprobs == [[(" A", -0.1), (" B", -2.0)]]
    assert result.usage == (9, 2, 10, 6)
    request = requests[0]
    assert str(request.url) == (
        "https://vertex.test/v1/publishers/google/models/gemini-test:generateContent"
    )
    assert request.headers["x-goog-api-key"] == "vertex-key"
    assert json.loads(request.content) == {
        "contents": [{"role": "user", "parts": [{"text": "rate this"}]}],
        "generationConfig": {
            "maxOutputTokens": 4096,
            "temperature": 1.0,
            "responseLogprobs": True,
            "logprobs": 7,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }


class StaticAdapter(AsyncVerifierAdapter):
    async def complete(self, prompt, model, top_logprobs=20, images=None):
        assert (prompt, model, top_logprobs, images) == ("rate this", "test-model", 20, None)
        return self.response

    async def aclose(self):
        return None

    def __init__(self):
        from llm_verifier.async_adapters import AdapterResponse

        self.response = AdapterResponse(
            text="<score_A> A </score_A>",
            tokens=["<score_A>", " A"],
            position_logprobs=[[("<score_A>", 0.0)], [(" A", 0.0)]],
            usage=(1, 0, 1, 0),
        )


@pytest.mark.asyncio
async def test_async_dispatcher_preserves_extract_score_response_tuple():
    """Catches a dispatcher change that breaks the legacy three-item response format."""
    result = await call_verifier_async(StaticAdapter(), "rate this", model="test-model")

    assert result == (
        "<score_A> A </score_A>",
        ["<score_A>", " A"],
        [[("<score_A>", 0.0)], [(" A", 0.0)]],
    )
