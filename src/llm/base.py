from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
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
    payload_defaults: Dict[str, Any] = field(default_factory=dict)
    runtime_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mode: str, payload: Mapping[str, Any]) -> "TeacherProfile":
        defaults = payload.get("payload_defaults") if isinstance(payload.get("payload_defaults"), Mapping) else {}
        runtime_options = payload.get("runtime_options") if isinstance(payload.get("runtime_options"), Mapping) else {}
        return cls(
            mode=str(mode),
            system_prompt=str(payload.get("system_prompt") or ""),
            user_prompt_template=str(payload.get("user_prompt_template") or "{payload_json}"),
            max_output_tokens=int(payload.get("max_output_tokens") or 900),
            temperature=float(payload.get("temperature") or 0.2),
            prompt_version=str(payload.get("prompt_version") or "v1"),
            payload_defaults=_to_jsonable(defaults),
            runtime_options=_to_jsonable(runtime_options),
        )

    def build_payload(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return _deep_merge(self.payload_defaults, payload)

    def format_user_prompt(self, payload: Mapping[str, Any]) -> str:
        import json

        payload_json = json.dumps(self.build_payload(payload), ensure_ascii=False, indent=2)
        return self.user_prompt_template.replace("{payload_json}", payload_json)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = _to_jsonable(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(str(key)), Mapping):
            merged[str(key)] = _deep_merge(dict(merged[str(key)]), value)
        else:
            merged[str(key)] = _to_jsonable(value)
    return merged


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_to_jsonable(item) for item in sorted(value, key=str)]
    return deepcopy(value)
