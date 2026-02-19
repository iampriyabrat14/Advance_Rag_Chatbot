"""
Conversation Memory
===================
Maintains a sliding-window conversation history per session.
Supports summary compression for very long conversations.
"""

import logging
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    In-process conversation memory with per-session isolation.

    Features:
    - Sliding window with configurable max turns
    - Full history export for debugging / UI display
    - Formatted prompt context for LLM injection
    """

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        # session_id → deque of {"role": ..., "content": ..., "ts": ...}
        self._sessions: Dict[str, deque] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def add_turn(self, session_id: str, role: str, content: str):
        """Add a single turn (user or assistant) to session memory."""
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(maxlen=self.max_turns * 2)

        self._sessions[session_id].append({
            "role":    role,
            "content": content,
            "ts":      datetime.now().isoformat()
        })

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return full turn history for a session."""
        return list(self._sessions.get(session_id, []))

    def get_formatted_context(self, session_id: str, max_chars: int = 2000) -> str:
        """
        Return conversation history as a formatted string for LLM prompts.
        Truncates from the oldest end if exceeding max_chars.
        """
        history = self.get_history(session_id)
        if not history:
            return ""

        lines = []
        for turn in history:
            role    = "Human" if turn["role"] == "user" else "Assistant"
            content = turn["content"][:500]  # cap per-turn
            lines.append(f"{role}: {content}")

        full_context = "\n".join(lines)
        if len(full_context) > max_chars:
            full_context = "...[truncated]\n" + full_context[-max_chars:]

        return full_context

    def get_recent_messages(self, session_id: str, n: int = 4) -> List[Dict]:
        """Return the last N messages (for LLM API message arrays)."""
        history = self.get_history(session_id)
        return [{"role": m["role"], "content": m["content"]} for m in history[-n:]]

    def clear(self, session_id: str):
        """Clear memory for a session."""
        self._sessions.pop(session_id, None)
        logger.info(f"Memory cleared for session: {session_id}")

    def session_count(self) -> int:
        return len(self._sessions)
