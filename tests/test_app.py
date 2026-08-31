import hashlib
import json
import time
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import SecretStr

from qwen_a2a.app import create_app
from qwen_a2a.cancellation import CancellationRegistry, RunIdentifiers
from qwen_a2a.config import Settings
from qwen_a2a.model_client import ModelDelta, ModelResult


class FakeModelClient:
    def __init__(self) -> None:
        self.complete_prompts: list[str] = []
        self.stream_prompts: list[str] = []

    async def complete(self, prompt: str) -> ModelResult:
        self.complete_prompts.append(prompt)
        return ModelResult(text="北京当前晴，25℃。")

    async def stream(self, prompt: str) -> AsyncIterator[ModelDelta]:
        self.stream_prompts.append(prompt)
        yield ModelDelta("reasoning", "正在联网查询。")
        yield ModelDelta("message", "北京当前")
        yield ModelDelta("message", "晴，25℃。")


def make_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "qwen_api_url": "https://model.example/v1/responses",
        "qwen_api_key": SecretStr("test-key"),
        "a2a_public_url": "https://agent.example",
    }
    values.update(overrides)
    return Settings(**values)


def request_body(method: str) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": "answer-1",
        "method": method,
        "params": {
            "message": {
                "messageId": "question-1",
                "contextId": "session-1",
                "role": "user",
                "parts": [{"kind": "text", "text": "今天北京天气怎么样？"}],
                "metadata": {
                    "communication": {
                        "session_id": "session-1",
                        "question_id": "question-1",
                        "answer_id": "answer-1",
                    }
                },
            }
        },
    }


@pytest.mark.asyncio
async def test_agent_card_and_message_send() -> None:
    model = FakeModelClient()
    app = create_app(make_settings(), model_client=model)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        card_response = await client.get("/.well-known/agent-card.json")
        response = await client.post("/a2a", json=request_body("message/send"))

    assert card_response.status_code == 200
    card = card_response.json()
    assert card["protocolVersion"] == "0.3.0"
    assert card["preferredTransport"] == "JSONRPC"
    assert card["capabilities"]["streaming"] is True
    assert card["url"] == "https://agent.example/a2a"

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["id"] == "question-1"
    assert result["contextId"] == "session-1"
    assert result["status"]["state"] == "completed"
    assert result["artifacts"][0]["name"] == "message"
    assert result["artifacts"][0]["parts"][0]["text"] == "北京当前晴，25℃。"
    assert model.complete_prompts == ["今天北京天气怎么样？"]
    assert model.stream_prompts == []


@pytest.mark.asyncio
async def test_message_stream_emits_qwen_compatible_event_order() -> None:
    model = FakeModelClient()
    app = create_app(make_settings(), model_client=model)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/a2a", json=request_body("message/stream"))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    results = [event["result"] for event in events]

    assert results[0]["kind"] == "task"
    assert results[0]["status"]["state"] == "submitted"
    assert [result["kind"] for result in results[1:]] == [
        "artifact-update",
        "artifact-update",
        "artifact-update",
        "status-update",
    ]
    assert results[1]["artifact"]["name"] == "reasoning"
    assert results[1]["lastChunk"] is True
    assert results[2]["artifact"]["name"] == "message"
    assert results[2]["append"] is False
    assert results[2]["lastChunk"] is False
    assert results[3]["append"] is True
    assert results[3]["lastChunk"] is True
    assert results[4]["status"]["state"] == "completed"
    assert results[4]["final"] is True
    assert model.stream_prompts == ["今天北京天气怎么样？"]
    assert model.complete_prompts == []


@pytest.mark.asyncio
async def test_event_callback_marks_matching_run_cancelled() -> None:
    registry = CancellationRegistry()
    app = create_app(
        make_settings(),
        model_client=FakeModelClient(),
        cancellation_registry=registry,
    )
    identifiers = RunIdentifiers(
        task_id="question-1",
        context_id="session-1",
        question_id="question-1",
        answer_id="answer-1",
    )

    async with registry.track(identifiers) as cancel_event:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/agent/event/callback",
                json={
                    "session_id": "session-1",
                    "question_id": "question-1",
                    "answer_id": "answer-1",
                    "event_type": "conversation.terminated",
                    "event_data": {"source": "abort"},
                },
            )
        assert cancel_event.is_set()

    assert response.json() == {"code": 0, "message": "success", "data": {}}


@pytest.mark.asyncio
async def test_optional_qwen_signature_protects_rpc_endpoint() -> None:
    settings = make_settings(
        qwen_client_id="client-id",
        qwen_client_secret=SecretStr("client-secret"),
    )
    app = create_app(settings, model_client=FakeModelClient())
    timestamp = str(int(time.time() * 1000))
    nonce = "nonce-1"
    source = f"POST&/a2a&{timestamp}&client-id&client-secret&{nonce}"
    token = hashlib.sha256(source.encode()).hexdigest()
    headers = {
        "X-Client-Id": "client-id",
        "X-Tm": timestamp,
        "X-Nonce": nonce,
        "X-Token": token,
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        unauthorized = await client.post("/a2a", json=request_body("message/send"))
        authorized = await client.post(
            "/a2a", json=request_body("message/send"), headers=headers
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
