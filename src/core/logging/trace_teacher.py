from __future__ import annotations

from typing import Any, Dict, Mapping


_ALLOWED_REQUEST_KEYS = {"input_text", "response", "plan_summary", "constraints", "scores", "reward"}
_ALLOWED_OUTPUT_KEYS = {
    "teacher_name",
    "prompt_version",
    "raw_output",
    "normalized_output",
    "latency_ms",
    "error",
    "label",
    "external_score",
}



def build_teacher_request_record(payload: Mapping[str, Any], *, mode: str = "teacher") -> Dict[str, Any]:
    compact = {str(key): payload[key] for key in payload.keys() if str(key) in _ALLOWED_REQUEST_KEYS}
    compact["mode"] = mode
    return compact



def build_teacher_output_record(summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): summary.get(key) for key in _ALLOWED_OUTPUT_KEYS}



def build_teacher_selection_record(summary: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not summary:
        return None
    teacher_name = summary.get("teacher_name")
    if not teacher_name:
        return None
    return {
        "selected_teacher": teacher_name,
        "selected_index": 0,
        "selection_reason": "single_teacher_first_available",
        "accepted_candidate_response": (summary.get("normalized_output") or {}).get("candidate_response"),
    }
