"""FastAPI composition root for the Qwen A2A weather service."""

from contextlib import asynccontextmanager

from a2a.server.apps.jsonrpc.fastapi_app import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from fastapi import FastAPI

from qwen_a2a import __version__
from qwen_a2a.cancellation import CancellationRegistry
from qwen_a2a.config import Settings, get_settings
from qwen_a2a.executor import QwenWeatherAgentExecutor
from qwen_a2a.model_client import ModelClient, QwenResponsesClient
from qwen_a2a.request_context import QwenRequestContextBuilder
from qwen_a2a.schemas import QwenCallbackResponse, QwenEventCallback
from qwen_a2a.security import QwenSignatureMiddleware


def create_app(
    settings: Settings | None = None,
    *,
    model_client: ModelClient | None = None,
    cancellation_registry: CancellationRegistry | None = None,
) -> FastAPI:
    """Build an injectable application for production and protocol tests."""

    settings = settings or get_settings()
    registry = cancellation_registry or CancellationRegistry()
    owns_model_client = model_client is None
    client = model_client or QwenResponsesClient(
        api_url=str(settings.qwen_api_url),
        api_key=settings.qwen_api_key.get_secret_value(),
        model=settings.qwen_model,
        enable_thinking=settings.qwen_enable_thinking,
        enable_web_search=settings.qwen_enable_web_search,
        timeout_seconds=settings.qwen_request_timeout_seconds,
    )

    executor = QwenWeatherAgentExecutor(client, registry)
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        request_context_builder=QwenRequestContextBuilder(task_store=task_store),
    )
    agent_card = _build_agent_card(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owns_model_client:
            close = getattr(client, "aclose", None)
            if close:
                await close()

    a2a_application = A2AFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    app = a2a_application.build(
        agent_card_url="/.well-known/agent-card.json",
        rpc_url=settings.a2a_rpc_path,
        title="Qwen A2A Weather Agent",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        settings.a2a_event_callback_path,
        response_model=QwenCallbackResponse,
    )
    async def handle_event(payload: QwenEventCallback) -> QwenCallbackResponse:
        await registry.cancel(
            question_id=payload.question_id,
            answer_id=payload.answer_id,
        )
        return QwenCallbackResponse()

    secret = (
        settings.qwen_client_secret.get_secret_value()
        if settings.qwen_client_secret
        else None
    )
    app.add_middleware(
        QwenSignatureMiddleware,
        client_id=settings.qwen_client_id,
        client_secret=secret,
        protected_paths=(
            settings.a2a_rpc_path,
            settings.a2a_event_callback_path,
        ),
        max_skew_seconds=settings.qwen_signature_max_skew_seconds,
    )
    app.state.agent_card = agent_card
    app.state.cancellation_registry = registry
    return app


def _build_agent_card(settings: Settings) -> AgentCard:
    skill = AgentSkill(
        id="get-weather",
        name="实时天气查询",
        description="通过联网搜索查询北京及其他城市的实时天气。",
        tags=["weather", "web-search", "realtime"],
        examples=["今天北京天气怎么样？", "北京明天会下雨吗？"],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    )
    return AgentCard(
        name="实时天气助手",
        description="基于千问模型和联网搜索提供实时天气信息。",
        url=settings.agent_endpoint,
        version=__version__,
        protocol_version="0.3.0",
        preferred_transport="JSONRPC",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[skill],
    )
