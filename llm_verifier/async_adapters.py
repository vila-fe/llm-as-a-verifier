"""Provider-neutral asynchronous HTTP adapters for verifier calls.

The adapters intentionally normalize provider response JSON to the legacy
``(text, tokens, position_logprobs)`` data model.  They use only
``httpx.AsyncClient`` for HTTP so callers can compose verifier work with their
own event loop without thread-pool dispatch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass
import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import httpx


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEEPSEEK_MODEL = "deepseek-v4-flash"

ImageInput = Union[str, bytes, os.PathLike]
PositionLogprobs = Optional[List[List[Tuple[str, float]]]]


@dataclass(frozen=True)
class AdapterResponse:
    """Provider-independent verifier response.

    ``usage`` is ``(input, cached_input, output, reasoning)``.  This mirrors
    ``TokenUsage.record``'s accounting fields while leaving the existing score
    extraction tuple unchanged at the public dispatcher boundary.
    """

    text: str
    tokens: Optional[List[str]]
    position_logprobs: PositionLogprobs
    usage: Tuple[int, int, int, int] = (0, 0, 0, 0)


class AsyncVerifierAdapter(ABC):
    """Explicit asynchronous verifier transport contract.

    Implementations accept a prompt and return token-level alternatives in the
    format consumed by :func:`llm_verifier.fine_grained_reward.extract_score`.
    """

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        top_logprobs: int = 20,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> AdapterResponse:
        """Submit one verifier request and normalize its response."""

    async def aclose(self) -> None:
        """Release adapter-owned resources, if any."""


def _as_image_list(images: Optional[Sequence[ImageInput]]) -> List[ImageInput]:
    if images is None:
        return []
    if isinstance(images, (str, bytes, os.PathLike)):
        return [images]
    return list(images)


def _sniff_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _openai_usage(data: dict[str, Any]) -> Tuple[int, int, int, int]:
    usage = data.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    cached = _int(prompt_details.get("cached_tokens"))
    if not cached:
        cached = _int(usage.get("prompt_cache_hit_tokens"))
    completion_details = usage.get("completion_tokens_details") or {}
    return (
        _int(usage.get("prompt_tokens")),
        cached,
        _int(usage.get("completion_tokens")),
        _int(completion_details.get("reasoning_tokens")),
    )


def _gemini_usage(data: dict[str, Any]) -> Tuple[int, int, int, int]:
    usage = data.get("usageMetadata") or {}
    thoughts = _int(usage.get("thoughtsTokenCount"))
    return (
        _int(usage.get("promptTokenCount")),
        _int(usage.get("cachedContentTokenCount")),
        _int(usage.get("candidatesTokenCount")) + thoughts,
        thoughts,
    )


def _openai_response(data: dict[str, Any]) -> AdapterResponse:
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    text = message.get("content") or ""
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content") or []
    if not content:
        return AdapterResponse(text, None, None, _openai_usage(data))

    tokens: List[str] = []
    position_logprobs: List[List[Tuple[str, float]]] = []
    for position in content:
        token = position.get("token") or ""
        tokens.append(token)
        alternatives = position.get("top_logprobs") or []
        parsed = [
            (alternative.get("token") or "", float(alternative.get("logprob", 0.0)))
            for alternative in alternatives
        ]
        if not parsed:
            parsed = [(token, float(position.get("logprob", 0.0)))]
        position_logprobs.append(parsed)
    return AdapterResponse(text, tokens, position_logprobs, _openai_usage(data))


def _gemini_response(data: dict[str, Any]) -> AdapterResponse:
    candidates = data.get("candidates") or []
    candidate = candidates[0] if candidates else {}
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(part.get("text") or "" for part in parts)
    result = candidate.get("logprobsResult") or {}
    top_candidates = result.get("topCandidates") or []
    if not top_candidates:
        return AdapterResponse(text, None, None, _gemini_usage(data))

    position_logprobs = []
    for position in top_candidates:
        alternatives = position.get("candidates") or []
        position_logprobs.append([
            (alternative.get("token") or "", float(alternative.get("logProbability", 0.0)))
            for alternative in alternatives
        ])
    chosen = result.get("chosenCandidates") or []
    tokens = [candidate.get("token") or "" for candidate in chosen] or None
    return AdapterResponse(text, tokens, position_logprobs, _gemini_usage(data))


class _HttpxAdapter(AsyncVerifierAdapter):
    """Shared asynchronous transport and image handling."""

    def __init__(
        self,
        base_url: str,
        *,
        client: Optional[httpx.AsyncClient] = None,
        timeout: Optional[float] = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    async def _load_image(self, image: ImageInput) -> Tuple[bytes, str]:
        if isinstance(image, bytes):
            data = image
        elif isinstance(image, (str, os.PathLike)):
            path = os.fspath(image)
            if path.startswith(("http://", "https://")):
                response = await self._client.get(path)
                response.raise_for_status()
                data = response.content
            else:
                with open(path, "rb") as handle:
                    data = handle.read()
        else:
            raise TypeError(
                f"unsupported image type {type(image).__name__}: expected a "
                "file path, an http(s) URL, or raw bytes"
            )
        return data, _sniff_mime(data)

    async def _openai_content(
        self, prompt: str, images: Optional[Sequence[ImageInput]]
    ) -> Any:
        image_list = _as_image_list(images)
        if not image_list:
            return prompt
        content = [{"type": "text", "text": prompt}]
        for image in image_list:
            data, mime = await self._load_image(image)
            encoded = base64.b64encode(data).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })
        return content

    async def _gemini_parts(
        self, prompt: str, images: Optional[Sequence[ImageInput]]
    ) -> List[dict[str, Any]]:
        parts: List[dict[str, Any]] = [{"text": prompt}]
        for image in _as_image_list(images):
            data, mime = await self._load_image(image)
            parts.append({
                "inlineData": {
                    "mimeType": mime,
                    "data": base64.b64encode(data).decode("ascii"),
                }
            })
        return parts

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class OpenAICompatibleAsyncAdapter(_HttpxAdapter):
    """Async REST adapter for OpenAI-compatible logprob endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        *,
        client: Optional[httpx.AsyncClient] = None,
        timeout: Optional[float] = 60.0,
        model: Optional[str] = None,
    ) -> None:
        super().__init__(base_url, client=client, timeout=timeout)
        self.api_key = api_key
        self.model = model
        self._resolved_model: Optional[str] = None

    async def _resolve_model(self, requested_model: str) -> str:
        if requested_model != DEFAULT_GEMINI_MODEL:
            return requested_model
        if self.model:
            return self.model
        if self._resolved_model:
            return self._resolved_model
        response = await self._client.get(
            self._url("models"), headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()
        models = response.json().get("data") or []
        if not models or not models[0].get("id"):
            raise RuntimeError("OpenAI-compatible /models response contained no model id")
        self._resolved_model = models[0]["id"]
        return self._resolved_model

    async def _payload(
        self,
        prompt: str,
        model: str,
        top_logprobs: int,
        images: Optional[Sequence[ImageInput]],
    ) -> dict[str, Any]:
        return {
            "model": await self._resolve_model(model),
            "messages": [{"role": "user", "content": await self._openai_content(prompt, images)}],
            "max_tokens": 4096,
            "temperature": 1.0,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            # vLLM/SGLang can use this to skip hybrid thinking before score tags.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    async def complete(
        self,
        prompt: str,
        model: str,
        top_logprobs: int = 20,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> AdapterResponse:
        payload = await self._payload(prompt, model, top_logprobs, images)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = await self._client.post(self._url("chat/completions"), json=payload, headers=headers)
        # Some OpenAI-compatible servers reject the vLLM/SGLang-only extension.
        # Retry exactly once with the portable body, preserving legacy behavior.
        if response.is_error and "chat_template_kwargs" in payload:
            fallback_payload = dict(payload)
            fallback_payload.pop("chat_template_kwargs")
            response = await self._client.post(
                self._url("chat/completions"), json=fallback_payload, headers=headers
            )
        response.raise_for_status()
        return _openai_response(response.json())


class DeepSeekAsyncAdapter(OpenAICompatibleAsyncAdapter):
    """OpenAI-compatible async adapter with DeepSeek reasoning parameters."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.deepseek.com",
        model: str = DEEPSEEK_MODEL,
        reasoning_effort: str = "high",
        max_tokens: int = 32768,
        client: Optional[httpx.AsyncClient] = None,
        timeout: Optional[float] = 60.0,
    ) -> None:
        super().__init__(
            base_url, api_key, client=client, timeout=timeout, model=model
        )
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens

    async def _resolve_model(self, requested_model: str) -> str:
        return self.model if requested_model == DEFAULT_GEMINI_MODEL else requested_model

    async def _payload(
        self,
        prompt: str,
        model: str,
        top_logprobs: int,
        images: Optional[Sequence[ImageInput]],
    ) -> dict[str, Any]:
        content = await self._openai_content(prompt, images)
        thinking = {"type": "disabled"}
        if self.reasoning_effort not in ("off", "disabled", "none"):
            thinking = {"type": "enabled"}
        payload = {
            "model": await self._resolve_model(model),
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": 1.0,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "thinking": thinking,
        }
        if thinking["type"] == "enabled":
            payload["reasoning_effort"] = self.reasoning_effort
        return payload


class GeminiAsyncAdapter(_HttpxAdapter):
    """Async Vertex Gemini REST adapter with token-logprob normalization."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://aiplatform.googleapis.com/v1",
        client: Optional[httpx.AsyncClient] = None,
        timeout: Optional[float] = 60.0,
    ) -> None:
        super().__init__(base_url, client=client, timeout=timeout)
        self.api_key = api_key

    async def complete(
        self,
        prompt: str,
        model: str,
        top_logprobs: int = 20,
        images: Optional[Sequence[ImageInput]] = None,
    ) -> AdapterResponse:
        payload = {
            "contents": [{
                "role": "user",
                "parts": await self._gemini_parts(prompt, images),
            }],
            "generationConfig": {
                "maxOutputTokens": 4096,
                "temperature": 1.0,
                "responseLogprobs": True,
                "logprobs": top_logprobs,
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        response = await self._client.post(
            self._url(f"publishers/google/models/{model}:generateContent"),
            json=payload,
            headers={"x-goog-api-key": self.api_key},
        )
        response.raise_for_status()
        return _gemini_response(response.json())
