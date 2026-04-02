from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TurnRecord:
    turn_index: int
    speaker: str
    model: str
    text: str
    latency_sec: float
    role: str = "assistant"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionRecord:
    session_id: str
    prompt_id: str
    input_text: str
    created_at: str
    meta: Dict[str, Any]
    turns: List[TurnRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prompt_id": self.prompt_id,
            "input": self.input_text,
            "created_at": self.created_at,
            "meta": self.meta,
            "turns": [t.to_dict() for t in self.turns],
        }


def session_to_messages(session: SessionRecord, include_system: Optional[str] = None) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if include_system:
        messages.append({"role": "system", "content": include_system})
    messages.append({"role": "user", "content": session.input_text})
    for turn in session.turns:
        messages.append(
            {
                "role": "assistant" if turn.role != "user" else "user",
                "content": turn.text,
            }
        )
    return messages
