"""Simple, bounded conversation memory used for contextual query rewriting."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ConversationMemory:
    max_turns: int = 12
    turns: list[Turn] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.turns.append(Turn("user", content))
        self._trim()

    def add_assistant(self, content: str) -> None:
        self.turns.append(Turn("assistant", content))
        self._trim()

    def as_dicts(self) -> list[dict]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def clear(self) -> None:
        self.turns = []

    def _trim(self) -> None:
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]
