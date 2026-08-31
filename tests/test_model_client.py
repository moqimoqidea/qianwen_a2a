import json

import httpx
import pytest

from qwen_a2a.model_client import QwenResponsesClient

API_URL = "https://model.example/v1/responses"


@pytest.mark.asyncio
async def test_complete_uses_non_streaming_responses_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "先查天气。"}],
                    },
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "北京晴。"}],
                    },
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = QwenResponsesClient(
        api_url=API_URL,
        api_key="test-key",
        model="qwen-test",
        http_client=http_client,
    )
    result = await client.complete("北京天气")
    await http_client.aclose()

    assert captured["stream"] is False
    assert captured["tools"] == [{"type": "web_search"}]
    assert result.reasoning == "先查天气。"
    assert result.text == "北京晴。"


@pytest.mark.asyncio
async def test_stream_parses_reasoning_and_message_deltas() -> None:
    captured: dict[str, object] = {}
    sse = "\n\n".join(
        [
            'data: {"type":"response.reasoning_summary_text.delta","delta":"查询中"}',
            'data: {"type":"response.output_text.delta","delta":"北京"}',
            'data: {"type":"response.output_text.delta","delta":"晴"}',
            "data: [DONE]",
            "",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse.encode(),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = QwenResponsesClient(
        api_url=API_URL,
        api_key="test-key",
        model="qwen-test",
        http_client=http_client,
    )
    deltas = [delta async for delta in client.stream("北京天气")]
    await http_client.aclose()

    assert captured["stream"] is True
    assert [(delta.kind, delta.text) for delta in deltas] == [
        ("reasoning", "查询中"),
        ("message", "北京"),
        ("message", "晴"),
    ]
