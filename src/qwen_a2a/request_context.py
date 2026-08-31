"""Qwen metadata adaptations for the official A2A request context."""

from a2a.server.agent_execution import RequestContext
from a2a.server.agent_execution.simple_request_context_builder import (
    SimpleRequestContextBuilder,
)
from a2a.server.context import ServerCallContext
from a2a.types import MessageSendParams, Task


class QwenRequestContextBuilder(SimpleRequestContextBuilder):
    """Reuse Qwen question/session IDs as A2A task/context IDs when available."""

    async def build(
        self,
        params: MessageSendParams | None = None,
        task_id: str | None = None,
        context_id: str | None = None,
        task: Task | None = None,
        context: ServerCallContext | None = None,
    ) -> RequestContext:
        if params:
            communication = _communication_metadata(params)
            task_id = (
                task_id
                or params.message.task_id
                or _string_value(communication, "question_id")
                or params.message.message_id
            )
            context_id = (
                context_id
                or params.message.context_id
                or _string_value(communication, "session_id")
            )
        return await super().build(
            params=params,
            task_id=task_id,
            context_id=context_id,
            task=task,
            context=context,
        )


def _communication_metadata(params: MessageSendParams) -> dict[str, object]:
    metadata = params.message.metadata or {}
    communication = metadata.get("communication")
    return communication if isinstance(communication, dict) else {}


def _string_value(mapping: dict[str, object], key: str) -> str | None:
    value = mapping.get(key)
    return value if isinstance(value, str) and value else None
