"""In-memory coordination for A2A and Qwen callback cancellation."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunIdentifiers:
    task_id: str
    context_id: str
    question_id: str | None = None
    answer_id: str | None = None
    session_id: str | None = None


@dataclass(slots=True)
class _ActiveRun:
    identifiers: RunIdentifiers
    cancel_event: asyncio.Event


class CancellationRegistry:
    """Tracks active model calls without coupling callbacks to the executor."""

    def __init__(self) -> None:
        self._runs: dict[str, _ActiveRun] = {}
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def track(self, identifiers: RunIdentifiers):
        run = _ActiveRun(identifiers, asyncio.Event())
        async with self._lock:
            self._runs[identifiers.task_id] = run
        try:
            yield run.cancel_event
        finally:
            async with self._lock:
                if self._runs.get(identifiers.task_id) is run:
                    self._runs.pop(identifiers.task_id, None)

    async def cancel_task(self, task_id: str) -> int:
        return await self.cancel(task_id=task_id)

    async def cancel(
        self,
        *,
        task_id: str | None = None,
        question_id: str | None = None,
        answer_id: str | None = None,
    ) -> int:
        async with self._lock:
            matching = [
                run
                for run in self._runs.values()
                if _matches(run.identifiers, task_id, question_id, answer_id)
            ]
            for run in matching:
                run.cancel_event.set()
            return len(matching)


def _matches(
    identifiers: RunIdentifiers,
    task_id: str | None,
    question_id: str | None,
    answer_id: str | None,
) -> bool:
    return bool(
        (task_id and identifiers.task_id == task_id)
        or (
            question_id
            and question_id in {identifiers.question_id, identifiers.task_id}
        )
        or (answer_id and identifiers.answer_id == answer_id)
    )
