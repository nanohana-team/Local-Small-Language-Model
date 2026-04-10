from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _relation_type_counts(relations: list[Any]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for relation in relations:
        if isinstance(relation, Mapping):
            relation_type = str(relation.get("type") or "unknown")
            counter[relation_type] += 1
    return dict(counter)


def _concept_labels(candidates: list[Any], limit: int = 3) -> list[str]:
    labels: list[str] = []
    for candidate in candidates[:limit]:
        if not isinstance(candidate, Mapping):
            continue
        label = candidate.get("label") or candidate.get("concept_id")
        if label is None:
            continue
        labels.append(str(label))
    return labels


def _nonempty_dict(value: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    compact = {str(key): val for key, val in value.items() if val not in (None, [], {}, "")}
    return compact or None


def build_episode_v1(
    trace: Mapping[str, Any],
    *,
    episode_index: int,
    input_source: str = "loop_learning",
    unknown_word_enrichment: Mapping[str, Any] | None = None,
    auto_input_meta: Mapping[str, Any] | None = None,
    trace_runtime_path: str = "runtime/traces/latest.jsonl",
) -> Dict[str, Any]:
    plan = _dict(trace.get("plan"))
    feedback = _dict(trace.get("feedback"))
    scoring_details = _dict(trace.get("scoring_details"))
    diagnostics = _dict(scoring_details.get("diagnostics"))
    input_features = _dict(trace.get("input_features"))
    reward = _dict(trace.get("reward"))
    timing = _dict(trace.get("timing"))
    unknown_overlay = _dict(unknown_word_enrichment)

    weakest_axes = [str(axis) for axis in feedback.get("weakest_axes", []) if axis is not None]
    decision = str(scoring_details.get("decision") or "review")
    accepted_relations = _list(trace.get("accepted_relations"))
    convergence_candidates = _list(trace.get("convergence_candidates"))
    missing_slots = [str(slot) for slot in trace.get("missing_slots", []) if slot is not None]
    blocking_issues = _list(scoring_details.get("blocking_issues"))
    blocking_issue_codes = [str(item.get("code")) for item in blocking_issues if isinstance(item, Mapping) and item.get("code")]
    unknown_words = [str(word) for word in input_features.get("unknown_words", []) if word is not None][:8]

    session_id = str(trace.get("session_id") or "")
    episode_id = f"{session_id}_ep_{episode_index:04d}" if session_id else f"ep_{episode_index:04d}"

    learning_summary = {
        "blocking_issue_codes": blocking_issue_codes,
        "weakest_axes": weakest_axes,
        "needs_dictionary_work": any(axis in {"relation_coverage", "relation_precision", "dangling_rate"} for axis in weakest_axes),
        "needs_planning_work": any(axis in {"plan_fitness", "input_retention"} for axis in weakest_axes),
        "needs_surface_work": any(axis in {"grammar_fitness"} for axis in weakest_axes),
        "needs_slot_work": any(axis in {"slot_fitness", "convergence_fitness"} for axis in weakest_axes),
        "should_train_response": decision == "promote",
        "should_review": decision in {"review", "reject"},
    }

    structure_summary = {
        "seed_count": int(diagnostics.get("seed_count") or 0),
        "accepted_relation_count": len(accepted_relations),
        "accepted_relation_type_counts": _relation_type_counts(accepted_relations),
        "accepted_concept_labels": _concept_labels(convergence_candidates),
        "missing_slots": missing_slots,
    }

    episode = {
        "schema_version": "loop_learning_episode_v3",
        "record_type": "episode",
        "session_id": trace.get("session_id"),
        "episode_id": episode_id,
        "episode_index": int(episode_index),
        "turn_id": trace.get("turn_id"),
        "timestamp_jst": trace.get("timestamp_jst"),
        "input": trace.get("input"),
        "input_source": input_source,
        "trace_ref": {
            "turn_id": trace.get("turn_id"),
            "trace_mode": trace.get("trace_mode"),
            "runtime_path": trace_runtime_path,
        },
        "plan_summary": {
            "intent": plan.get("intent"),
            "response_mode": plan.get("response_mode"),
            "required_slots": list(plan.get("required_slots", [])),
            "unknown_focus": plan.get("unknown_focus"),
            "needs_clarification": bool(plan.get("needs_clarification", False)),
            "fallback_reason": plan.get("fallback_reason"),
        },
        "outcome": {
            "response": trace.get("response"),
            "decision": decision,
            "feedback_label": feedback.get("label"),
            "feedback_text": feedback.get("text"),
            "reward_total": reward.get("total"),
        },
        "learning_summary": learning_summary,
        "structure_summary": structure_summary,
        "timing": {
            "total_ms": timing.get("total_ms"),
        },
        "unknown_word_enrichment": unknown_overlay or None,
    }

    input_summary = _nonempty_dict(
        {
            "unknown_words": unknown_words,
            "detected_topics": [str(topic) for topic in input_features.get("detected_topics", []) if topic is not None][:4],
            "seed_concepts": [str(concept) for concept in input_features.get("seed_concepts", []) if concept is not None][:4],
        }
    )
    if input_summary is not None:
        episode["input_summary"] = input_summary

    auto_input_context: Dict[str, Any] | None = None
    if input_source == "auto_input" and isinstance(auto_input_meta, Mapping):
        auto_input_context = _nonempty_dict(
            {
                "source": auto_input_meta.get("source"),
                "used_llm": bool(auto_input_meta.get("used_llm", False)),
                "topic_hints": list(auto_input_meta.get("topic_hints", []))[:6],
            }
        )
    if auto_input_context is not None:
        episode["auto_input_context"] = auto_input_context

    return episode


__all__ = ["build_episode_v1"]
