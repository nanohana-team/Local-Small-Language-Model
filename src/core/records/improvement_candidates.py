from __future__ import annotations

from typing import Any, Dict, Mapping


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    normalized: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized



def build_teacher_improvement_candidate(
    *,
    session_id: str,
    turn_id: str,
    user_input: str,
    response: str,
    plan: Mapping[str, Any] | None,
    teacher_summary: Mapping[str, Any] | None,
    external_signal: Mapping[str, Any] | None = None,
) -> Dict[str, Any] | None:
    if not isinstance(teacher_summary, Mapping):
        return None

    normalized = teacher_summary.get("normalized_output")
    if not isinstance(normalized, Mapping):
        return None

    teacher_hints = normalized.get("teacher_hints") if isinstance(normalized.get("teacher_hints"), Mapping) else {}
    missing_slots = _string_list(teacher_hints.get("missing_slots"))
    relation_types = _string_list(teacher_hints.get("recommended_relation_types"))
    issues = _string_list(teacher_hints.get("issues"))
    candidate_response = normalized.get("candidate_response")
    rationale = normalized.get("rationale_optional")

    if not missing_slots and not relation_types and not issues and not candidate_response:
        return None

    promote_targets: list[str] = []
    if missing_slots:
        promote_targets.append("slot")
    if relation_types:
        promote_targets.append("relation")
    if candidate_response or rationale or issues:
        promote_targets.append("surface")

    plan_obj = dict(plan or {})
    external_obj = dict(external_signal or {})

    candidate = {
        "candidate_id": f"{turn_id}:teacher_hint:0",
        "record_type": "teacher_improvement_candidate",
        "source_kind": "external_teacher",
        "session_id": session_id,
        "turn_id": turn_id,
        "input": user_input,
        "response": response,
        "plan_intent": plan_obj.get("intent"),
        "response_mode": plan_obj.get("response_mode"),
        "teacher": {
            "teacher_name": teacher_summary.get("teacher_name"),
            "provider": teacher_summary.get("provider"),
            "model": teacher_summary.get("model"),
            "prompt_version": teacher_summary.get("prompt_version"),
            "label": teacher_summary.get("label"),
            "external_score": teacher_summary.get("external_score"),
        },
        "external_signal": {
            "merge_policy": external_obj.get("merge_policy"),
            "used_sources": list(external_obj.get("used_sources") or []),
            "external_score": external_obj.get("external_score"),
        },
        "teacher_target": candidate_response,
        "teacher_feedback": rationale,
        "teacher_hints": {
            "missing_slots": missing_slots,
            "recommended_relation_types": relation_types,
            "issues": issues,
        },
        "candidate_summary": {
            "slot_candidate_count": len(missing_slots),
            "relation_type_candidate_count": len(relation_types),
            "issue_count": len(issues),
            "has_teacher_target": bool(candidate_response),
        },
        "review": {
            "status": "pending",
            "reason": "teacher_hint_extracted",
            "promote_targets": promote_targets,
            "auto_promote": False,
        },
    }
    return candidate
