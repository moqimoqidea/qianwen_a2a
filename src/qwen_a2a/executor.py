"""A2A executor that translates Qwen model output into protocol events."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TaskState, TextPart
from a2a.utils import new_task

from qwen_a2a.cancellation import CancellationRegistry, RunIdentifiers
from qwen_a2a.model_client import ModelClient, ModelDelta, ModelUpstreamError

logger = logging.getLogger(__name__)


class QwenWeatherAgentExecutor(AgentExecutor):
    """Run the weather-capable model and emit Qwen-compatible A2A events."""

    def __init__(
        self,
        model_client: ModelClient,
        cancellation_registry: CancellationRegistry,
    ) -> None:
        self._model_client = model_client
        self._cancellations = cancellation_registry

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if (
            context.message is None
            or context.task_id is None
            or context.context_id is None
        ):
            raise ValueError("A2A message, task ID, and context ID are required")

        if context.current_task is None:
            await event_queue.enqueue_event(new_task(context.message))

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        query = context.get_user_input().strip()
        if not query:
            await updater.update_status(TaskState.rejected, final=True)
            return

        identifiers = _run_identifiers(context)
        try:
            async with self._cancellations.track(identifiers) as cancel_event:
                if _is_streaming_request(context):
                    await self._execute_streaming(query, updater, cancel_event)
                else:
                    await self._execute_complete(query, updater, cancel_event)
        except asyncio.CancelledError:
            raise
        except ModelUpstreamError:
            logger.exception("Qwen model request failed for task %s", context.task_id)
            await updater.failed()
        except Exception:
            logger.exception("Unexpected agent failure for task %s", context.task_id)
            await updater.failed()

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if context.task_id is None or context.context_id is None:
            raise ValueError("task ID and context ID are required for cancellation")
        await self._cancellations.cancel_task(context.task_id)
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()

    async def _execute_complete(
        self,
        query: str,
        updater: TaskUpdater,
        cancel_event: asyncio.Event,
    ) -> None:
        result = await self._model_client.complete(query)
        if cancel_event.is_set():
            await updater.cancel()
            return
        if result.reasoning:
            await _add_text_artifact(
                updater,
                text=result.reasoning,
                artifact_id=str(uuid.uuid4()),
                name="reasoning",
                append=False,
                last_chunk=True,
            )
        await _add_text_artifact(
            updater,
            text=result.text,
            artifact_id=str(uuid.uuid4()),
            name="message",
            append=False,
            last_chunk=True,
        )
        await updater.complete()

    async def _execute_streaming(
        self,
        query: str,
        updater: TaskUpdater,
        cancel_event: asyncio.Event,
    ) -> None:
        writer = _ArtifactStreamWriter(updater)
        async for delta in _until_cancelled(
            self._model_client.stream(query), cancel_event
        ):
            await writer.write(delta)

        if cancel_event.is_set():
            await updater.cancel()
            return
        await writer.finish()
        if not writer.has_message:
            raise ModelUpstreamError("upstream stream contained no output text")
        await updater.complete()


class _ArtifactStreamWriter:
    """Keep one delta pending so each artifact can mark its true last chunk."""

    def __init__(self, updater: TaskUpdater) -> None:
        self._updater = updater
        self._pending: ModelDelta | None = None
        self._current_kind: str | None = None
        self._artifact_id: str | None = None
        self._has_emitted = False
        self.has_message = False

    async def write(self, delta: ModelDelta) -> None:
        if not delta.text:
            return
        if delta.kind == "message":
            self.has_message = True
        if self._current_kind is not None and delta.kind != self._current_kind:
            await self._flush(last_chunk=True)
            self._reset(delta.kind)
        elif self._current_kind is None:
            self._reset(delta.kind)
        elif self._pending is not None:
            await self._flush(last_chunk=False)
        self._pending = delta

    async def finish(self) -> None:
        await self._flush(last_chunk=True)

    def _reset(self, kind: str) -> None:
        self._current_kind = kind
        self._artifact_id = str(uuid.uuid4())
        self._has_emitted = False
        self._pending = None

    async def _flush(self, *, last_chunk: bool) -> None:
        if self._pending is None or self._artifact_id is None:
            return
        await _add_text_artifact(
            self._updater,
            text=self._pending.text,
            artifact_id=self._artifact_id,
            name=self._pending.kind,
            append=self._has_emitted,
            last_chunk=last_chunk,
        )
        self._has_emitted = True
        self._pending = None


async def _until_cancelled(
    stream: AsyncIterator[ModelDelta],
    cancel_event: asyncio.Event,
) -> AsyncIterator[ModelDelta]:
    iterator = stream.__aiter__()
    while True:
        next_item = asyncio.create_task(anext(iterator))
        wait_for_cancel = asyncio.create_task(cancel_event.wait())
        done, _ = await asyncio.wait(
            {next_item, wait_for_cancel}, return_when=asyncio.FIRST_COMPLETED
        )
        if wait_for_cancel in done and cancel_event.is_set():
            next_item.cancel()
            await asyncio.gather(next_item, return_exceptions=True)
            aclose = getattr(iterator, "aclose", None)
            if aclose:
                await aclose()
            return

        wait_for_cancel.cancel()
        await asyncio.gather(wait_for_cancel, return_exceptions=True)
        try:
            yield next_item.result()
        except StopAsyncIteration:
            return


async def _add_text_artifact(
    updater: TaskUpdater,
    *,
    text: str,
    artifact_id: str,
    name: str,
    append: bool,
    last_chunk: bool,
) -> None:
    await updater.add_artifact(
        parts=[Part(root=TextPart(text=text))],
        artifact_id=artifact_id,
        name=name,
        append=append,
        last_chunk=last_chunk,
    )


def _is_streaming_request(context: RequestContext) -> bool:
    return bool(
        context.call_context
        and context.call_context.state.get("method") == "message/stream"
    )


def _run_identifiers(context: RequestContext) -> RunIdentifiers:
    metadata = context.message.metadata or {} if context.message else {}
    communication = metadata.get("communication")
    communication = communication if isinstance(communication, dict) else {}
    return RunIdentifiers(
        task_id=context.task_id or "",
        context_id=context.context_id or "",
        question_id=_optional_string(communication.get("question_id")),
        answer_id=_optional_string(communication.get("answer_id")),
        session_id=_optional_string(communication.get("session_id")),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
