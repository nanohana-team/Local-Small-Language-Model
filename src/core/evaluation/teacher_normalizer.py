from __future__ import annotations

from typing import Any, Dict, Mapping

from src.llm.base import LLMCallResult


_DEF_EMPTY_HINTS = {
    "missing_slots": [],
    "recommended_relation_types": [],
    "issues": [],
}


def _as_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None



def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    normalized: list[str] = []
    for item in items:
        text = _as_string(item)
        if text:
            normalized.append(text)
    return normalized



def normalize_teacher_output(parsed: Mapping[str, Any] | None, raw_text: str = "") -> Dict[str, Any]:
    payload = dict(parsed or {})
    teacher_hints = payload.get("teacher_hints") if isinstance(payload.get("teacher_hints"), Mapping) else {}

    candidate_response = (
        _as_string(payload.get("candidate_response"))
        or _as_string(payload.get("teacher_target"))
        or _as_string(payload.get("improved_response"))
    )
    rationale_optional = (
        _as_string(payload.get("rationale_optional"))
        or _as_string(payload.get("feedback_text"))
        or _as_string(payload.get("rationale"))
    )
    safety_flags = _as_string_list(payload.get("safety_flags"))
    if payload.get("refusal") is True:
        safety_flags.append("teacher_refusal")
    if not payload:
        safety_flags.append("unparsed_output")
    if not candidate_response and raw_text.strip():
        safety_flags.append("missing_candidate_response")

    normalized = {
        "candidate_response": candidate_response,
        "rationale_optional": rationale_optional,
        "safety_flags": sorted(set(safety_flags)),
        "format_ok": bool(payload) and candidate_response is not None,
        "label": _as_string(payload.get("label")),
        "external_score": payload.get("external_score"),
        "teacher_hints": {
            "missing_slots": _as_string_list(teacher_hints.get("missing_slots")),
            "recommended_relation_types": _as_string_list(teacher_hints.get("recommended_relation_types")),
            "issues": _as_string_list(teacher_hints.get("issues")),
        }
        if teacher_hints
        else dict(_DEF_EMPTY_HINTS),
        "issues": _as_string_list(payload.get("issues")),
        "raw_fallback": None if payload else _as_string(raw_text),
    }
    return normalized



def summarize_teacher_result(result: LLMCallResult) -> Dict[str, Any]:
    normalized = normalize_teacher_output(result.parsed or {}, result.raw_text)
    result.normalized_output = normalized

    score = normalized.get("external_score")
    try:
        external_score = float(score) if score is not None else None
    except (TypeError, ValueError):
        external_score = None

    teacher_name = result.teacher_name or f"{result.provider}:{result.model}"
    return {
        "provider": result.provider,
        "model": result.model,
        "mode": result.mode,
        "teacher_name": teacher_name,
        "prompt_version": result.prompt_version,
        "external_score": external_score,
        "label": normalized.get("label"),
        "feedback_text": normalized.get("rationale_optional"),
        "raw_output": result.raw_text,
        "normalized_output": normalized,
        "error": result.error,
        "latency_ms": result.latency_ms,
    }
