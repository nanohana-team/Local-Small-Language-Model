from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .base import LLMCallResult, TeacherProfile
from .config import load_environment, load_llm_order, load_teacher_profiles

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class ExternalTeacherOrchestrator:
    """Ordered evaluator / teacher runner with provider fallbacks."""

    def __init__(
        self,
        *,
        llm_order_path: str | Path = "settings/LLM_order.yaml",
        teacher_profile_path: str | Path = "settings/teacher_profiles.yaml",
    ) -> None:
        load_environment()
        self.model_order = load_llm_order(llm_order_path)
        self.profiles = load_teacher_profiles(teacher_profile_path)

    def evaluate_turn(self, payload: Mapping[str, Any]) -> LLMCallResult:
        return self._run("evaluator", payload)

    def teach_turn(self, payload: Mapping[str, Any]) -> LLMCallResult:
        return self._run("teacher", payload)

    def _run(self, mode: str, payload: Mapping[str, Any]) -> LLMCallResult:
        profile = self.profiles.get(mode)
        if profile is None:
            return LLMCallResult(
                provider="none",
                model="none",
                mode=mode,
                raw_text="",
                error=f"missing_profile:{mode}",
                prompt_version=None,
                teacher_name=None,
            )

        last_error: str | None = None
        for model_name in self.model_order:
            adapter = self._make_adapter(model_name)
            started = time.perf_counter()
            try:
                raw_text = adapter.generate_text(
                    system_prompt=profile.system_prompt,
                    user_prompt=profile.format_user_prompt(payload),
                    max_output_tokens=profile.max_output_tokens,
                    temperature=profile.temperature,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                parsed = _parse_json_payload(raw_text)
                return LLMCallResult(
                    provider=adapter.provider,
                    model=model_name,
                    mode=mode,
                    raw_text=raw_text,
                    parsed=parsed,
                    latency_ms=round(latency_ms, 3),
                    prompt_version=profile.prompt_version,
                    teacher_name=f"{adapter.provider}:{model_name}" if mode == "teacher" else None,
                )
            except Exception as exc:  # pragma: no cover - network / provider exceptions vary.
                last_error = f"{type(exc).__name__}: {exc}"
        return LLMCallResult(
            provider="fallback_exhausted",
            model=self.model_order[-1] if self.model_order else "none",
            mode=mode,
            raw_text="",
            error=last_error or "no_model_available",
            prompt_version=profile.prompt_version,
            teacher_name=(f"fallback_exhausted:{self.model_order[-1]}" if self.model_order and mode == "teacher" else None),
        )

    @staticmethod
    def _make_adapter(model_name: str):
        if model_name.startswith("gemini") or model_name.startswith("models/gemini"):
            from .gemini_adapter import GeminiAdapter

            return GeminiAdapter(model_name)
        if model_name.startswith("gpt") or model_name.startswith("o"):
            from .openai_adapter import OpenAIAdapter

            return OpenAIAdapter(model_name)
        raise ValueError(f"Unsupported provider model: {model_name}")



def _parse_json_payload(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {}
    match = JSON_BLOCK_RE.search(raw)
    if match:
        raw = match.group(1).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return {"raw_text": text}
        else:
            return {"raw_text": text}
    return parsed if isinstance(parsed, dict) else {"value": parsed}
