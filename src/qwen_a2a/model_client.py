"""Client abstraction for the Qwen Responses-compatible model endpoint."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

ModelDeltaKind = Literal["reasoning", "message"]


@dataclass(frozen=True, slots=True)
class ModelDelta:
    """One incremental model output fragment."""

    kind: ModelDeltaKind
    text: str


@dataclass(frozen=True, slots=True)
class ModelResult:
    """A complete model response."""

    text: str
    reasoning: str | None = None


class ModelClient(Protocol):
    """Minimal model interface consumed by the A2A executor."""

    async def complete(self, prompt: str) -> ModelResult: ...

    def stream(self, prompt: str) -> AsyncIterator[ModelDelta]: ...


class ModelUpstreamError(RuntimeError):
    """Raised when the upstream model request cannot produce a valid response."""


class QwenResponsesClient:
    """Async client for Qwen's OpenAI Responses-compatible endpoint."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        enable_thinking: bool = True,
        enable_web_search: bool = True,
        timeout_seconds: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_url = api_url
        self._model = model
        self._enable_thinking = enable_thinking
        self._enable_web_search = enable_web_search
        self._owns_http_client = http_client is None
        self._http_client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def complete(self, prompt: str) -> ModelResult:
        response = await self._http_client.post(
            self._api_url,
            json=self._request_payload(prompt, stream=False),
        )
        await self._ensure_success(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ModelUpstreamError("upstream returned malformed JSON") from exc
        return _parse_complete_response(payload)

    async def stream(self, prompt: str) -> AsyncIterator[ModelDelta]:
        seen_kinds: set[ModelDeltaKind] = set()
        async with self._http_client.stream(
            "POST",
            self._api_url,
            json=self._request_payload(prompt, stream=True),
        ) as response:
            await self._ensure_success(response)
            async for raw_data in _iter_sse_data(response.aiter_lines()):
                if raw_data == "[DONE]":
                    break
                try:
                    event = json.loads(raw_data)
                except json.JSONDecodeError as exc:
                    raise ModelUpstreamError(
                        "upstream returned malformed SSE data"
                    ) from exc

                event_type = event.get("type")
                if event_type in {"error", "response.failed", "response.incomplete"}:
                    raise ModelUpstreamError(_extract_error_message(event))

                delta = _parse_stream_delta(event)
                if delta and delta.text:
                    seen_kinds.add(delta.kind)
                    yield delta
                    continue

                if event_type == "response.completed":
                    final_payload = event.get("response")
                    if isinstance(final_payload, dict):
                        result = _parse_complete_response(final_payload)
                        if result.reasoning and "reasoning" not in seen_kinds:
                            yield ModelDelta("reasoning", result.reasoning)
                        if result.text and "message" not in seen_kinds:
                            yield ModelDelta("message", result.text)

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()

    def _request_payload(self, prompt: str, *, stream: bool) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model,
            "input": prompt,
            "enable_thinking": self._enable_thinking,
            "stream": stream,
        }
        if self._enable_web_search:
            payload["tools"] = [{"type": "web_search"}]
        return payload

    @staticmethod
    async def _ensure_success(response: httpx.Response) -> None:
        if not response.is_error:
            return
        await response.aread()
        raise ModelUpstreamError(f"upstream returned HTTP {response.status_code}")


async def _iter_sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    data_lines: list[str] = []
    async for line in lines:
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line.startswith("{"):
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            yield line
    if data_lines:
        yield "\n".join(data_lines)


def _parse_stream_delta(event: dict[str, object]) -> ModelDelta | None:
    event_type = event.get("type")
    delta = event.get("delta")
    if not isinstance(delta, str):
        choices = event.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice_delta = choices[0].get("delta")
            if isinstance(choice_delta, dict):
                delta = choice_delta.get("content")

    if not isinstance(delta, str):
        return None
    if event_type in {
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
    }:
        return ModelDelta("reasoning", delta)
    if event_type in {"response.output_text.delta", None}:
        return ModelDelta("message", delta)
    return None


def _parse_complete_response(payload: dict[str, object]) -> ModelResult:
    message_parts: list[str] = []
    reasoning_parts: list[str] = []

    output_text = payload.get("output_text")
    has_top_level_output_text = isinstance(output_text, str)
    if has_top_level_output_text:
        message_parts.append(output_text)

    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message" and not has_top_level_output_text:
                _collect_text(item.get("content"), message_parts)
            elif item_type == "reasoning":
                _collect_text(item.get("summary"), reasoning_parts)

    choices = payload.get("choices")
    if not message_parts and isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                message_parts.append(message["content"])

    text = "".join(message_parts)
    if not text:
        raise ModelUpstreamError("upstream response contained no output text")
    reasoning = "".join(reasoning_parts) or None
    return ModelResult(text=text, reasoning=reasoning)


def _collect_text(value: object, target: list[str]) -> None:
    if not isinstance(value, list):
        return
    for part in value:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            target.append(part["text"])


def _extract_error_message(event: dict[str, object]) -> str:
    error = event.get("error")
    if not isinstance(error, dict):
        response = event.get("response")
        if isinstance(response, dict):
            error = response.get("error")
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return f"upstream error: {error['message']}"
    return "upstream response failed"
