from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


def _clamp_ms(value: float) -> float:
    try:
        return max(0.0, round(float(value), 3))
    except (TypeError, ValueError):
        return 0.0


def build_stage_metric(
    *,
    stage: str,
    elapsed_ms: float,
    candidate_count: int = 0,
    kept_count: int = 0,
    dropped_count: int = 0,
    branching_factor: float | None = None,
    expand_reason_codes: Sequence[str] | None = None,
    converge_reason_codes: Sequence[str] | None = None,
    rule_ids: Sequence[str] | None = None,
    dict_feature_ids: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_count = max(0, int(candidate_count))
    kept_count = max(0, int(kept_count))
    dropped_count = max(0, int(dropped_count))
    if branching_factor is None:
        branching_factor = float(candidate_count) / float(max(1, kept_count or 1)) if candidate_count else 0.0
    return {
        'stage': str(stage or 'unknown'),
        'elapsed_ms': _clamp_ms(elapsed_ms),
        'candidate_count': candidate_count,
        'kept_count': kept_count,
        'dropped_count': dropped_count,
        'branching_factor': round(float(branching_factor), 6),
        'expand_reason_codes': [str(item) for item in (expand_reason_codes or []) if str(item).strip()],
        'converge_reason_codes': [str(item) for item in (converge_reason_codes or []) if str(item).strip()],
        'rule_ids': [str(item) for item in (rule_ids or []) if str(item).strip()],
        'dict_feature_ids': [str(item) for item in (dict_feature_ids or []) if str(item).strip()],
        'metadata': dict(metadata or {}),
    }



def summarize_drop_reasons(actions: Sequence[Any], *, limit: int = 3) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for action in actions or []:
        for candidate in getattr(action, 'candidates', []) or []:
            reason = str(getattr(candidate, 'drop_reason', '') or '').strip()
            if reason:
                counts[reason] += 1
    return [
        {'reason': reason, 'count': count}
        for reason, count in counts.most_common(max(1, int(limit)))
    ]



def summarize_top_candidates(candidates: Sequence[Any], *, limit: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(
        list(candidates or []),
        key=lambda item: float(getattr(item, 'final_score', 0.0) or 0.0),
        reverse=True,
    )
    summary: list[dict[str, Any]] = []
    for index, candidate in enumerate(ranked[: max(1, int(limit))], start=1):
        summary.append(
            {
                'rank': index,
                'text': str(getattr(candidate, 'text', '') or ''),
                'template_id': str(getattr(candidate, 'template_id', '') or ''),
                'final_score': float(getattr(candidate, 'final_score', 0.0) or 0.0),
                'slot_coverage': float(getattr(candidate, 'slot_coverage', 0.0) or 0.0),
                'semantic_score': float(getattr(candidate, 'semantic_score', 0.0) or 0.0),
            }
        )
    return summary



def build_turn_audit_summary(
    *,
    input_text: str,
    response_text: str,
    stage_metrics: Sequence[Mapping[str, Any]],
    actions: Sequence[Any],
    scored_candidates: Sequence[Any],
    missing_required: Sequence[str] | None = None,
    unknown_word_learning: Mapping[str, Any] | None = None,
    reward_total: float = 0.0,
    reward_internal: float = 0.0,
    reward_external: float = 0.0,
) -> dict[str, Any]:
    total_elapsed_ms = round(sum(float(item.get('elapsed_ms', 0.0) or 0.0) for item in (stage_metrics or [])), 3)
    anomaly_flags: list[str] = []
    if missing_required:
        anomaly_flags.append('missing_required_slots')
    if float(reward_external) <= 0.05:
        anomaly_flags.append('external_reward_very_low')
    if float(reward_total) <= 0.35:
        anomaly_flags.append('reward_total_low')
    if total_elapsed_ms >= 500.0:
        anomaly_flags.append('latency_high')
    if unknown_word_learning and list(unknown_word_learning.get('quarantined_records', []) or []):
        anomaly_flags.append('dict_update_quarantined')
    if unknown_word_learning and list(unknown_word_learning.get('applied_words', []) or []):
        anomaly_flags.append('dict_updated')

    return {
        'input_text': str(input_text or ''),
        'final_output': str(response_text or ''),
        'total_elapsed_ms': total_elapsed_ms,
        'stage_count': len(list(stage_metrics or [])),
        'top_candidates': summarize_top_candidates(scored_candidates),
        'drop_reasons_top3': summarize_drop_reasons(actions),
        'anomaly_flags': anomaly_flags,
        'estimated_cause': ', '.join(anomaly_flags) if anomaly_flags else 'none',
        'missing_required': [str(item) for item in (missing_required or []) if str(item).strip()],
        'reward': {
            'total': float(reward_total),
            'internal': float(reward_internal),
            'external': float(reward_external),
        },
    }



def build_dict_update_event(
    *,
    update_type: str,
    entry_id: str,
    source_turn_id: str,
    reason: str,
    pollution_risk: float,
    status: str,
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
    evaluation_score: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        'dict_update_id': f'dictupd_{source_turn_id}_{entry_id}_{status}',
        'entry_id': str(entry_id or ''),
        'update_type': str(update_type or 'unknown'),
        'before': dict(before or {}),
        'after': dict(after or {}),
        'reason': str(reason or ''),
        'source_turn_id': str(source_turn_id or ''),
        'evaluation_score': None if evaluation_score is None else float(evaluation_score),
        'pollution_risk': float(pollution_risk),
        'status': str(status or ''),
        'metadata': dict(metadata or {}),
    }
