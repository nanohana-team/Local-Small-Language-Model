from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from src.core.convergence.convergence_v1 import ConvergenceResult
from src.core.divergence.divergence_v1 import DivergenceResult
from src.core.planning.plan_v1 import PlanV1
from src.core.slotting.slot_v1 import SlotResult

DEFAULT_SCORING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "latency": {
        "target_ms": 500.0,
        "hard_limit_ms": 3000.0,
    },
    "thresholds": {
        "strong_seed_score": 0.25,
        "accepted_relation_target": 4,
        "accepted_concept_target": 4,
        "divergence_candidate_target": 6,
    },
    "weights": {
        "plan_fitness": 0.15,
        "relation_coverage": 0.15,
        "divergence_relevance": 0.10,
        "convergence_fitness": 0.15,
        "relation_precision": 0.10,
        "slot_fitness": 0.15,
        "grammar_fitness": 0.08,
        "input_retention": 0.07,
        "latency_fitness": 0.03,
        "dangling_penalty": 0.02,
    },
}

AXIS_REMEDIES = {
    "plan_fitness": "Plan の fallback 条件や intent 判定を見直す",
    "relation_coverage": "relation 量と優先度、seed から届く接続幅を増やす",
    "divergence_relevance": "seed ノイズ除去と relation priority を調整する",
    "convergence_fitness": "採用数、plan bonus、取りこぼし条件を見直す",
    "relation_precision": "不要 relation の採用を減らし、priority hit を増やす",
    "slot_fitness": "slotting と predicate-slot の接続を補強する",
    "grammar_fitness": "surface template と文末処理を整える",
    "input_retention": "topic / support / unknown focus を応答へ残す",
    "latency_fitness": "depth / branching budget と startup cache を調整する",
    "dangling_rate": "relation validation と import review を強化する",
}


@dataclass
class TurnScoreV1:
    scores: Dict[str, float]
    reward: Dict[str, float | None]
    feedback: Dict[str, Any]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _ScoreContext:
    plan: PlanV1
    divergence: DivergenceResult
    convergence: ConvergenceResult
    slots: SlotResult
    response_text: str
    total_ms: float
    validation_report: Mapping[str, Any] | None
    config: Dict[str, Any]
    config_path: str | None


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _safe_ratio(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    if denominator <= 0:
        return float(default)
    return float(numerator) / float(denominator)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=8)
def _load_scoring_config_cached(path_str: str | None) -> tuple[Dict[str, Any], str | None]:
    config = dict(DEFAULT_SCORING_CONFIG)
    resolved_path: str | None = None

    candidate_paths = []
    if path_str:
        candidate_paths.append(Path(path_str))
    else:
        repo_root = Path(__file__).resolve().parents[3]
        candidate_paths.append(repo_root / "settings" / "scoring_v1.yaml")

    for candidate in candidate_paths:
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if isinstance(loaded, Mapping):
            config = _deep_merge(config, loaded)
            resolved_path = str(candidate)
            break
    return config, resolved_path


def load_scoring_config(config_path: str | None = None) -> tuple[Dict[str, Any], str | None]:
    return _load_scoring_config_cached(config_path)


def _slot_label(slot: Any) -> str | None:
    if isinstance(slot, Mapping):
        label = slot.get("label")
        if label:
            return str(label)
    return None


def _count_strong_seeds(divergence: DivergenceResult, threshold: float) -> int:
    return sum(1 for match in divergence.seed_matches if float(getattr(match, "score", 0.0)) >= threshold)


def _response_mentions(value: str | None, response: str) -> bool:
    if not value:
        return False
    return str(value) in response


def _plan_fitness(ctx: _ScoreContext) -> float:
    if not ctx.plan.needs_clarification:
        score = 1.0
        if ctx.plan.fallback_reason == "unknown_words_dominant":
            score -= 0.12
        if ctx.plan.intent == "define" and ctx.plan.unknown_focus:
            score += 0.02
        return _clamp(score)

    if ctx.plan.fallback_reason == "unknown_focus_term" and ctx.plan.unknown_focus:
        return 0.68
    if ctx.plan.fallback_reason == "unknown_words_dominant":
        return 0.62
    topic = ctx.slots.filled_slots.get("topic")
    if topic is not None:
        return 0.58
    return 0.45


def _relation_coverage(ctx: _ScoreContext) -> float:
    thresholds = ctx.config.get("thresholds", {}) if isinstance(ctx.config.get("thresholds"), Mapping) else {}
    relation_target = max(1.0, float(thresholds.get("accepted_relation_target", 4)))
    accepted_relations = list(ctx.convergence.accepted_relations or [])
    required_slots = list(ctx.plan.required_slots or [])
    missing_slots = list(ctx.slots.missing_slots or [])
    slot_fill_ratio = 1.0 if not required_slots else _clamp(1.0 - _safe_ratio(len(missing_slots), len(required_slots), default=1.0))
    relation_ratio = _clamp(_safe_ratio(len(accepted_relations), relation_target, default=0.0))
    support_relation_exists = any(
        ctx.slots.slot_evidence.get(key)
        for key in (
            "topic_support_relation",
            "inverse_topic_support_relation",
            "topic_support_hypernym_path",
            "inverse_topic_support_hypernym_path",
        )
    )
    bridge_bonus = 0.12 if support_relation_exists else 0.0
    score = 0.62 * relation_ratio + 0.26 * slot_fill_ratio + bridge_bonus
    if ctx.plan.needs_clarification and ctx.plan.unknown_focus and not accepted_relations:
        score = max(score, 0.55)
    return _clamp(score)


def _divergence_relevance(ctx: _ScoreContext) -> float:
    thresholds = ctx.config.get("thresholds", {}) if isinstance(ctx.config.get("thresholds"), Mapping) else {}
    strong_seed_threshold = float(thresholds.get("strong_seed_score", 0.25))
    candidate_target = max(1.0, float(thresholds.get("divergence_candidate_target", 6)))

    seed_count = len(ctx.divergence.seed_matches)
    strong_seed_count = _count_strong_seeds(ctx.divergence, strong_seed_threshold)
    candidate_count = len(ctx.divergence.candidate_concepts)
    accepted_count = len(ctx.convergence.accepted_concepts)
    seed_strength = _safe_ratio(strong_seed_count, max(1, min(seed_count, 3)), default=0.0)
    candidate_density = _safe_ratio(accepted_count, max(1, min(candidate_count, int(candidate_target))), default=0.0)
    top_score = float(ctx.divergence.candidate_concepts[0].score) if ctx.divergence.candidate_concepts else 0.0
    score = 0.42 * _clamp(seed_strength) + 0.33 * _clamp(candidate_density) + 0.25 * _clamp(top_score)
    if ctx.plan.unknown_focus and seed_count == 0:
        score = max(score, 0.56)
    return _clamp(score)


def _convergence_fitness(ctx: _ScoreContext) -> float:
    thresholds = ctx.config.get("thresholds", {}) if isinstance(ctx.config.get("thresholds"), Mapping) else {}
    concept_target = max(1.0, float(thresholds.get("accepted_concept_target", 4)))
    accepted_count = len(ctx.convergence.accepted_concepts)
    required_slots = list(ctx.plan.required_slots or [])
    missing_slots = list(ctx.slots.missing_slots or [])
    slot_fill_ratio = 1.0 if not required_slots else _clamp(1.0 - _safe_ratio(len(missing_slots), len(required_slots), default=1.0))
    concept_band = _clamp(_safe_ratio(accepted_count, concept_target, default=0.0))
    top_final_score = 0.0
    if ctx.convergence.accepted_concepts:
        top_final_score = float(ctx.convergence.accepted_concepts[0].get("final_score", 0.0))
        if top_final_score > 1.0:
            top_final_score = _clamp(top_final_score / 1.6)
    score = 0.44 * slot_fill_ratio + 0.36 * concept_band + 0.20 * top_final_score
    if ctx.plan.needs_clarification and ctx.plan.unknown_focus:
        score = max(score, 0.52)
    return _clamp(score)


def _relation_precision(ctx: _ScoreContext) -> float:
    accepted_relations = list(ctx.convergence.accepted_relations or [])
    if not accepted_relations:
        if ctx.plan.needs_clarification and ctx.plan.unknown_focus:
            return 0.58
        return 0.0 if ctx.plan.required_slots else 0.4

    priority_set = set(ctx.plan.relation_type_priority or [])
    priority_hits = sum(1 for relation in accepted_relations if str(relation.get("type")) in priority_set)
    precision = _safe_ratio(priority_hits, len(accepted_relations), default=0.0)
    support_relation_exists = any(
        ctx.slots.slot_evidence.get(key)
        for key in (
            "topic_support_relation",
            "inverse_topic_support_relation",
            "topic_support_hypernym_path",
            "inverse_topic_support_hypernym_path",
        )
    )
    score = 0.30 + 0.60 * _clamp(precision) + (0.10 if support_relation_exists else 0.0)
    return _clamp(score)


def _slot_fitness(ctx: _ScoreContext) -> float:
    required_slots = list(ctx.plan.required_slots or [])
    if not required_slots:
        return 1.0
    missing_slots = list(ctx.slots.missing_slots or [])
    fill_ratio = _clamp(1.0 - _safe_ratio(len(missing_slots), len(required_slots), default=1.0))
    evidence_bonus = 0.0
    if ctx.slots.selected_slot_frame:
        evidence_bonus += 0.05
    if ctx.slots.slot_evidence.get("accepted_concepts"):
        evidence_bonus += 0.05
    score = _clamp(fill_ratio + evidence_bonus)
    if ctx.plan.needs_clarification and ctx.plan.unknown_focus:
        score = max(score, 0.55)
    return score


def _grammar_fitness(response: str, plan: PlanV1) -> float:
    text = str(response or "").strip()
    if not text:
        return 0.0

    score = 1.0
    if "  " in text:
        score -= 0.08
    if any(token in text for token in ("..", "、、", "。。", "??", "！！")):
        score -= 0.12
    if not text.endswith(("。", "！", "？", ".", "!", "?")) and len(text) >= 8:
        score -= 0.06
    if len(text) > 180:
        score -= 0.08
    if plan.needs_clarification and "教えてください" not in text and len(text) < 8:
        score -= 0.08
    return _clamp(score)


def _input_retention(ctx: _ScoreContext) -> float:
    response_text = str(ctx.response_text or "")
    if not ctx.divergence.seed_matches and not ctx.plan.unknown_focus:
        return 0.25

    score = 0.40
    topic_label = _slot_label(ctx.slots.filled_slots.get("topic"))
    support_label = _slot_label(ctx.slots.filled_slots.get("support"))
    reason_label = _slot_label(ctx.slots.filled_slots.get("reason"))
    comparison_label = _slot_label(ctx.slots.filled_slots.get("comparison"))

    if _response_mentions(topic_label, response_text):
        score += 0.25
    if _response_mentions(support_label, response_text) and support_label != topic_label:
        score += 0.10
    if _response_mentions(reason_label, response_text) and reason_label != topic_label:
        score += 0.05
    if _response_mentions(comparison_label, response_text) and comparison_label != topic_label:
        score += 0.05
    if ctx.plan.unknown_focus and _response_mentions(ctx.plan.unknown_focus, response_text):
        score += 0.20
    if ctx.divergence.seed_matches:
        score += 0.05
    return _clamp(score)


def _latency_fitness(total_ms: float, config: Mapping[str, Any]) -> float:
    latency = config.get("latency", {}) if isinstance(config.get("latency"), Mapping) else {}
    target_ms = float(latency.get("target_ms", 500.0))
    hard_limit_ms = float(latency.get("hard_limit_ms", max(target_ms * 6.0, 3000.0)))
    if total_ms <= target_ms:
        return 1.0
    if total_ms >= hard_limit_ms:
        return 0.0
    return _clamp(1.0 - ((total_ms - target_ms) / max(1.0, hard_limit_ms - target_ms)))


def _dangling_rate(validation_report: Mapping[str, Any] | None) -> float:
    if not isinstance(validation_report, Mapping):
        return 0.0
    issues = [str(item) for item in list(validation_report.get("errors", [])) + list(validation_report.get("warnings", []))]
    dangling = sum(1 for issue in issues if "missing concept" in issue or "dangling" in issue)
    if dangling <= 0:
        return 0.0
    return _clamp(_safe_ratio(dangling, max(1, len(issues)), default=0.0))


def _axis_quality(axis_name: str, value: float) -> float:
    if axis_name == "dangling_rate":
        return _clamp(1.0 - float(value))
    return _clamp(float(value))


def _weakest_axes(scores: Mapping[str, float]) -> list[str]:
    ordered = sorted(scores.items(), key=lambda item: (_axis_quality(item[0], item[1]), item[0]))[:3]
    return [name for name, _ in ordered]


def _build_feedback(scores: Mapping[str, float]) -> Dict[str, Any]:
    weakest_axes = _weakest_axes(scores)
    suggestions = [AXIS_REMEDIES[name] for name in weakest_axes if name in AXIS_REMEDIES]
    reward_total = sum(_axis_quality(name, float(value)) for name, value in scores.items()) / max(1, len(scores))
    if reward_total >= 0.82:
        label = "good"
        text = "内部評価としてはかなり安定。弱い軸を局所改善すれば次に進める。"
    elif reward_total >= 0.65:
        label = "mixed"
        text = f"概ね通っているけれど、{', '.join(weakest_axes[:2])} がまだ弱い。"
    else:
        label = "needs_work"
        text = f"最小縦スライスは通るが、{', '.join(weakest_axes[:2])} を優先して詰めたい。"
    return {
        "label": label,
        "text": text,
        "weakest_axes": weakest_axes,
        "suggestions": suggestions,
    }


def _build_details(ctx: _ScoreContext, scores: Mapping[str, float]) -> Dict[str, Any]:
    thresholds = ctx.config.get("thresholds", {}) if isinstance(ctx.config.get("thresholds"), Mapping) else {}
    strong_seed_threshold = float(thresholds.get("strong_seed_score", 0.25))
    accepted_relations = list(ctx.convergence.accepted_relations or [])
    priority_set = set(ctx.plan.relation_type_priority or [])
    priority_hits = sum(1 for relation in accepted_relations if str(relation.get("type")) in priority_set)
    return {
        "score_version": "v1.1",
        "config_path": ctx.config_path,
        "config_version": ctx.config.get("version", 1),
        "diagnostics": {
            "seed_count": len(ctx.divergence.seed_matches),
            "strong_seed_count": _count_strong_seeds(ctx.divergence, strong_seed_threshold),
            "candidate_count": len(ctx.divergence.candidate_concepts),
            "accepted_concept_count": len(ctx.convergence.accepted_concepts),
            "rejected_concept_count": len(ctx.convergence.rejected_concepts),
            "accepted_relation_count": len(accepted_relations),
            "rejected_relation_count": len(ctx.convergence.rejected_relations),
            "required_slot_count": len(ctx.plan.required_slots),
            "missing_slot_count": len(ctx.slots.missing_slots),
            "filled_slot_count": sum(1 for value in ctx.slots.filled_slots.values() if value is not None),
            "relation_priority_hits": priority_hits,
            "relation_priority_ratio": round(_safe_ratio(priority_hits, len(accepted_relations), default=0.0), 6),
            "unknown_focus": ctx.plan.unknown_focus,
            "topic_label": _slot_label(ctx.slots.filled_slots.get("topic")),
            "support_label": _slot_label(ctx.slots.filled_slots.get("support")),
            "response_length": len(str(ctx.response_text or "")),
        },
        "weights": ctx.config.get("weights", {}),
        "thresholds": thresholds,
        "weakest_axes": _weakest_axes(scores),
    }


def _compute_scores_from_context(ctx: _ScoreContext) -> Dict[str, float]:
    scores = {
        "plan_fitness": round(_plan_fitness(ctx), 6),
        "relation_coverage": round(_relation_coverage(ctx), 6),
        "divergence_relevance": round(_divergence_relevance(ctx), 6),
        "convergence_fitness": round(_convergence_fitness(ctx), 6),
        "relation_precision": round(_relation_precision(ctx), 6),
        "slot_fitness": round(_slot_fitness(ctx), 6),
        "grammar_fitness": round(_grammar_fitness(ctx.response_text, ctx.plan), 6),
        "input_retention": round(_input_retention(ctx), 6),
        "latency_fitness": round(_latency_fitness(ctx.total_ms, ctx.config), 6),
        "dangling_rate": round(_dangling_rate(ctx.validation_report), 6),
    }
    return scores


def _compute_reward(scores: Mapping[str, float], config: Mapping[str, Any]) -> Dict[str, float | None]:
    weights = config.get("weights", {}) if isinstance(config.get("weights"), Mapping) else {}
    reward_internal = (
        float(weights.get("plan_fitness", 0.15)) * float(scores.get("plan_fitness", 0.0))
        + float(weights.get("relation_coverage", 0.15)) * float(scores.get("relation_coverage", 0.0))
        + float(weights.get("divergence_relevance", 0.10)) * float(scores.get("divergence_relevance", 0.0))
        + float(weights.get("convergence_fitness", 0.15)) * float(scores.get("convergence_fitness", 0.0))
        + float(weights.get("relation_precision", 0.10)) * float(scores.get("relation_precision", 0.0))
        + float(weights.get("slot_fitness", 0.15)) * float(scores.get("slot_fitness", 0.0))
        + float(weights.get("grammar_fitness", 0.08)) * float(scores.get("grammar_fitness", 0.0))
        + float(weights.get("input_retention", 0.07)) * float(scores.get("input_retention", 0.0))
        + float(weights.get("latency_fitness", 0.03)) * float(scores.get("latency_fitness", 0.0))
        + float(weights.get("dangling_penalty", 0.02)) * (1.0 - float(scores.get("dangling_rate", 0.0)))
    )
    reward_internal = round(_clamp(reward_internal), 6)
    return {"internal": reward_internal, "external": None, "total": reward_internal}


def score_turn_v1(
    *,
    plan: PlanV1,
    divergence: DivergenceResult,
    convergence: ConvergenceResult,
    slots: SlotResult,
    response_text: str,
    total_ms: float,
    validation_report: Mapping[str, Any] | None = None,
    scoring_config_path: str | None = None,
) -> TurnScoreV1:
    config, resolved_path = load_scoring_config(scoring_config_path)
    ctx = _ScoreContext(
        plan=plan,
        divergence=divergence,
        convergence=convergence,
        slots=slots,
        response_text=response_text,
        total_ms=total_ms,
        validation_report=validation_report,
        config=config,
        config_path=resolved_path,
    )
    scores = _compute_scores_from_context(ctx)
    reward = _compute_reward(scores, config)
    feedback = _build_feedback(scores)
    details = _build_details(ctx, scores)
    return TurnScoreV1(scores=scores, reward=reward, feedback=feedback, details=details)


def compute_internal_scores(
    *,
    plan: PlanV1,
    divergence: DivergenceResult,
    convergence: ConvergenceResult,
    slots: SlotResult,
    total_ms: float,
    response_text: str = "",
    validation_report: Mapping[str, Any] | None = None,
    scoring_config_path: str | None = None,
) -> Dict[str, float]:
    return score_turn_v1(
        plan=plan,
        divergence=divergence,
        convergence=convergence,
        slots=slots,
        response_text=response_text,
        total_ms=total_ms,
        validation_report=validation_report,
        scoring_config_path=scoring_config_path,
    ).scores


def compute_reward_v1(scores: Mapping[str, float], *, scoring_config_path: str | None = None) -> Dict[str, float | None]:
    config, _ = load_scoring_config(scoring_config_path)
    return _compute_reward(scores, config)


__all__ = [
    "TurnScoreV1",
    "compute_internal_scores",
    "compute_reward_v1",
    "load_scoring_config",
    "score_turn_v1",
]
