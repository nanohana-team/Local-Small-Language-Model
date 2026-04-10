from __future__ import annotations

from typing import Any, Dict, Mapping

from src.core.scoring import load_scoring_config
from src.llm.base import LLMCallResult


DEFAULT_EXTERNAL_REWARD_CONFIG: Dict[str, float] = {
    "internal_weight": 0.8,
    "external_weight": 0.2,
    "evaluator_mix": 0.8,
    "teacher_mix": 0.2,
}


def _coerce_optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None



def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))



def summarize_external_result(result: LLMCallResult) -> Dict[str, Any]:
    parsed = dict(result.parsed or {})
    external_score = _coerce_optional_float(parsed.get("external_score"))
    label = parsed.get("label")
    feedback_text = parsed.get("feedback_text")
    summary = {
        "provider": result.provider,
        "model": result.model,
        "mode": result.mode,
        "external_score": external_score,
        "label": str(label) if label is not None else None,
        "feedback_text": str(feedback_text) if feedback_text is not None else None,
        "parsed": parsed,
        "error": result.error,
        "latency_ms": result.latency_ms,
    }
    return summary



def _load_external_reward_config(scoring_config_path: str | None = None) -> Dict[str, float]:
    config, _ = load_scoring_config(scoring_config_path)
    loaded = config.get("external_reward") if isinstance(config.get("external_reward"), Mapping) else {}
    merged = dict(DEFAULT_EXTERNAL_REWARD_CONFIG)
    for key, value in loaded.items():
        try:
            merged[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return merged



def build_external_signal(
    evaluator_summary: Mapping[str, Any] | None,
    teacher_summary: Mapping[str, Any] | None,
    *,
    scoring_config_path: str | None = None,
) -> Dict[str, Any]:
    config = _load_external_reward_config(scoring_config_path)
    evaluator_score = _coerce_optional_float((evaluator_summary or {}).get("external_score"))
    teacher_score = _coerce_optional_float((teacher_summary or {}).get("external_score"))

    evaluator_mix = max(0.0, float(config.get("evaluator_mix", 0.8)))
    teacher_mix = max(0.0, float(config.get("teacher_mix", 0.2)))

    used_sources: list[str] = []
    merge_policy = "none"
    external_score: float | None = None
    if evaluator_score is not None and teacher_score is not None:
        total_mix = evaluator_mix + teacher_mix
        if total_mix <= 0.0:
            external_score = evaluator_score
            merge_policy = "evaluator_only_invalid_mix"
            used_sources = ["evaluator"]
        else:
            external_score = _clamp((evaluator_mix * evaluator_score + teacher_mix * teacher_score) / total_mix)
            merge_policy = "blended"
            used_sources = ["evaluator", "teacher"]
    elif evaluator_score is not None:
        external_score = _clamp(evaluator_score)
        merge_policy = "evaluator_only"
        used_sources = ["evaluator"]
    elif teacher_score is not None:
        external_score = _clamp(teacher_score)
        merge_policy = "teacher_only"
        used_sources = ["teacher"]

    feedback_parts: list[str] = []
    for summary in (evaluator_summary, teacher_summary):
        if not isinstance(summary, Mapping):
            continue
        text = summary.get("feedback_text")
        if text is None:
            continue
        normalized = str(text).strip()
        if normalized and normalized not in feedback_parts:
            feedback_parts.append(normalized)

    label = None
    if isinstance(evaluator_summary, Mapping) and evaluator_summary.get("label") is not None:
        label = str(evaluator_summary.get("label"))
    elif isinstance(teacher_summary, Mapping) and teacher_summary.get("label") is not None:
        label = str(teacher_summary.get("label"))

    return {
        "external_score": external_score,
        "label": label,
        "feedback_text": " / ".join(feedback_parts) if feedback_parts else None,
        "used_sources": used_sources,
        "merge_policy": merge_policy,
        "source_scores": {
            "evaluator": evaluator_score,
            "teacher": teacher_score,
        },
        "weights": {
            "internal_weight": float(config.get("internal_weight", 0.8)),
            "external_weight": float(config.get("external_weight", 0.2)),
            "evaluator_mix": evaluator_mix,
            "teacher_mix": teacher_mix,
        },
    }



def apply_external_feedback(
    reward: Mapping[str, Any],
    evaluator_summary: Mapping[str, Any] | None = None,
    teacher_summary: Mapping[str, Any] | None = None,
    *,
    scoring_config_path: str | None = None,
) -> Dict[str, float | None]:
    config = _load_external_reward_config(scoring_config_path)
    internal_value = _coerce_optional_float(reward.get("internal")) or 0.0
    external_signal = build_external_signal(
        evaluator_summary,
        teacher_summary,
        scoring_config_path=scoring_config_path,
    )
    external_value = _coerce_optional_float(external_signal.get("external_score"))

    if external_value is None:
        total = round(_clamp(internal_value), 6)
    else:
        internal_weight = max(0.0, float(config.get("internal_weight", 0.8)))
        external_weight = max(0.0, float(config.get("external_weight", 0.2)))
        total = round(_clamp(internal_weight * internal_value + external_weight * external_value), 6)

    return {
        "internal": round(_clamp(internal_value), 6),
        "external": None if external_value is None else round(_clamp(external_value), 6),
        "total": total,
    }



def apply_external_reward(
    reward: Mapping[str, Any],
    external_summary: Mapping[str, Any],
    *,
    alpha: float = 0.8,
    beta: float = 0.2,
) -> Dict[str, float | None]:
    internal = _coerce_optional_float(reward.get("internal")) or 0.0
    external_value = _coerce_optional_float((external_summary or {}).get("external_score"))
    total = round(_clamp(internal), 6) if external_value is None else round(_clamp(alpha * internal + beta * external_value), 6)
    return {
        "internal": round(_clamp(internal), 6),
        "external": None if external_value is None else round(_clamp(external_value), 6),
        "total": total,
    }
