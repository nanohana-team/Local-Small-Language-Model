from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping


@dataclass
class LLMCallResult:
    provider: str
    model: str
    mode: str
    raw_text: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    normalized_output: Dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float | None = None
    prompt_version: str | None = None
    teacher_name: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseLLMAdapter(ABC):
    """Thin provider adapter interface for external evaluator / teacher calls."""

    def __init__(self, model: str) -> None:
        self.model = str(model)

    @property
    @abstractmethod
    def provider(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        raise NotImplementedError


@dataclass
class TeacherProfile:
    mode: str
    system_prompt: str
    user_prompt_template: str
    max_output_tokens: int = 900
    temperature: float = 0.2
    prompt_version: str = "v1"

    @classmethod
    def from_mapping(cls, mode: str, payload: Mapping[str, Any]) -> "TeacherProfile":
        return cls(
            mode=str(mode),
            system_prompt=str(payload.get("system_prompt") or ""),
            user_prompt_template=str(payload.get("user_prompt_template") or "{payload_json}"),
            max_output_tokens=int(payload.get("max_output_tokens") or 900),
            temperature=float(payload.get("temperature") or 0.2),
            prompt_version=str(payload.get("prompt_version") or "v1"),
        )

    def format_user_prompt(self, payload: Mapping[str, Any]) -> str:
        import json

        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        return self.user_prompt_template.replace("{payload_json}", payload_json)
