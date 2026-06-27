"""In-memory conversation session store keyed by session_id."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart


class SessionStore:
    """Thread-safe store of pydantic-ai message history per session."""

    def __init__(
        self,
        max_messages: int = 20,
        llm_turns: int = 3,
        analysis_user_messages: int = 6,
        assistant_history_chars: int = 500,
    ) -> None:
        self._sessions: dict[str, list[ModelMessage]] = defaultdict(list)
        self._lock = Lock()
        self._max_messages = max_messages
        self._llm_turns = llm_turns
        self._analysis_user_messages = analysis_user_messages
        self._assistant_history_chars = assistant_history_chars

    def get_history(self, session_id: str | None) -> list[ModelMessage]:
        if not session_id:
            return []
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def get_llm_history(self, session_id: str | None) -> list[ModelMessage]:
        """Last N turns only, with truncated assistant text — for the LLM."""
        history = self.get_history(session_id)
        if not history or self._llm_turns <= 0:
            return []
        cap = self._llm_turns * 2
        return self._truncate_assistant_messages(history[-cap:])

    def get_user_messages(self, session_id: str | None) -> list[str]:
        """All stored user text turns."""
        history = self.get_history(session_id)
        messages: list[str] = []
        for msg in history:
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                        messages.append(part.content)
        return messages

    def get_recent_user_messages(self, session_id: str | None) -> list[str]:
        """Recent user turns for intent/lead analysis (not full LLM history)."""
        messages = self.get_user_messages(session_id)
        if self._analysis_user_messages <= 0:
            return messages
        return messages[-self._analysis_user_messages :]

    def _truncate_assistant_messages(
        self, messages: list[ModelMessage]
    ) -> list[ModelMessage]:
        if self._assistant_history_chars <= 0:
            return messages

        trimmed: list[ModelMessage] = []
        for msg in messages:
            if not isinstance(msg, ModelResponse):
                trimmed.append(msg)
                continue
            parts = []
            for part in msg.parts:
                if isinstance(part, TextPart) and isinstance(part.content, str):
                    text = part.content
                    if len(text) > self._assistant_history_chars:
                        text = text[: self._assistant_history_chars].rstrip() + "…"
                    parts.append(TextPart(content=text))
                else:
                    parts.append(part)
            trimmed.append(ModelResponse(parts=parts))
        return trimmed

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

    def load_messages(
        self, session_id: str, messages: list
    ) -> None:
        """Hydrate in-memory history from persisted messages."""
        from app.schemas.message import MessageRole

        with self._lock:
            history: list[ModelMessage] = []
            for msg in messages:
                role = msg.role if hasattr(msg, "role") else msg.get("role")
                content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                if role == MessageRole.USER or role == "user":
                    history.append(
                        ModelRequest(parts=[UserPromptPart(content=content)])
                    )
                elif role == MessageRole.ASSISTANT or role == "assistant":
                    history.append(
                        ModelResponse(parts=[TextPart(content=content)])
                    )
            if history:
                self._sessions[session_id] = history[-self._max_messages :]
