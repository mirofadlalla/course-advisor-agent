"""In-memory conversation session store keyed by session_id."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart


class SessionStore:
    """Thread-safe store of pydantic-ai message history per session."""

    def __init__(self, max_messages: int = 40) -> None:
        self._sessions: dict[str, list[ModelMessage]] = defaultdict(list)
        self._lock = Lock()
        self._max_messages = max_messages

    def get_history(self, session_id: str | None) -> list[ModelMessage]:
        if not session_id:
            return []
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def get_user_messages(self, session_id: str | None) -> list[str]:
        """Extract prior user text turns for intent/lead analysis."""
        history = self.get_history(session_id)
        messages: list[str] = []
        for msg in history:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                        messages.append(part.content)
        return messages

    def append_turn(
        self,
        session_id: str | None,
        user_message: str,
        assistant_message: str,
    ) -> None:
        if not session_id:
            return
        with self._lock:
            session = self._sessions[session_id]
            session.append(
                ModelRequest(parts=[UserPromptPart(content=user_message)])
            )
            session.append(
                ModelResponse(parts=[TextPart(content=assistant_message)])
            )
            if len(session) > self._max_messages:
                self._sessions[session_id] = session[-self._max_messages :]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
