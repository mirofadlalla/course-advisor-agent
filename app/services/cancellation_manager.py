"""Asyncio task cancellation per conversation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CancellationManager:
    """
    Ensures only one active request per conversation.

    When a new request arrives, the previous asyncio.Task is cancelled immediately.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    async def register(self, conversation_id: str, task: asyncio.Task[Any]) -> None:
        async with self._lock:
            existing = self._tasks.get(conversation_id)
            if existing and not existing.done():
                logger.info(
                    "Cancelling previous task for conversation %s", conversation_id
                )
                existing.cancel()
                try:
                    await existing
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.debug("Previous task ended with: %s", exc)
            self._tasks[conversation_id] = task

    async def unregister(self, conversation_id: str, task: asyncio.Task[Any]) -> None:
        async with self._lock:
            current = self._tasks.get(conversation_id)
            if current is task:
                self._tasks.pop(conversation_id, None)

    def is_cancelled(self, conversation_id: str) -> bool:
        task = self._tasks.get(conversation_id)
        return task is not None and task.cancelled()
