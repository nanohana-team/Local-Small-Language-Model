from __future__ import annotations

from typing import Any, Mapping

from .base import LLMCallResult, TeacherProfile
from .openai_adapter import OpenAIAdapter
from .teacher_adapter_base import TeacherTurnRequest


class OpenAITeacherAdapter:
    """Small future-proof OpenAI-specific teacher wrapper.

    The current app still uses the provider-agnostic orchestrator, but this class
    gives the project a dedicated teacher adapter seam for later multi-teacher or
    provider-specific tuning.
    """

    def __init__(self, model: str, profile: TeacherProfile) -> None:
        self.model = str(model)
        self.profile = profile
        self._adapter = OpenAIAdapter(self.model)

    @property
    def teacher_name(self) -> str:
        return f"{self._adapter.provider}:{self.model}"

    def generate(self, request: TeacherTurnRequest | Mapping[str, Any]) -> LLMCallResult:
        payload = request.to_payload() if isinstance(request, TeacherTurnRequest) else dict(request)
        raw_text = self._adapter.generate_text(
            system_prompt=self.profile.system_prompt,
            user_prompt=self.profile.format_user_prompt(payload),
            max_output_tokens=self.profile.max_output_tokens,
            temperature=self.profile.temperature,
        )
        return LLMCallResult(
            provider=self._adapter.provider,
            model=self.model,
            mode=self.profile.mode,
            raw_text=raw_text,
            teacher_name=self.teacher_name,
            prompt_version=self.profile.prompt_version,
        )
