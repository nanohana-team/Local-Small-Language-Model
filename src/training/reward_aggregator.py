from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from src.core.schema import (
    EvaluationResult,
    ExternalRewardBreakdown,
    ExternalRewardComponent,
    InternalRewardBreakdown,
    RewardBreakdown,
    ResponseResult,
)


@dataclass(slots=True)
class RewardAggregatorConfig:
    alpha: float = 0.7
    beta: float = 0.3
    fallback_strategy: str = "neutral"
    neutral_external_score: float = 0.5
    use_power_transform: bool = False
    external_power: float = 1.5
    mismatch_internal_threshold: float = 0.76
    mismatch_external_threshold: float = 0.24
    mismatch_penalty: float = 0.24


class RewardAggregator:
    def __init__(self, config: RewardAggregatorConfig | None = None) -> None:
        self.config = config or RewardAggregatorConfig()

    def aggregate(
        self,
        *,
        response: ResponseResult,
        evaluation: Iterable[EvaluationResult] | None = None,
    ) -> RewardBreakdown:
        evaluation_list = list(evaluation or [])
        internal = self._build_internal(response)
        external = self._build_external(internal=internal, evaluation=evaluation_list)
        alpha, beta = self._normalized_weights(self.config.alpha, self.config.beta)
        raw_total = alpha * internal.total + beta * external.total
        mismatch_penalty_applied = 0.0
        reasons = list(internal.reasons)
        for component in external.components:
            suffix = f"{component.evaluator_name}:{component.label}" if component.label else component.evaluator_name
            reasons.append(f"external:{suffix}")

        if (
            internal.total >= float(self.config.mismatch_internal_threshold)
            and external.total <= float(self.config.mismatch_external_threshold)
        ):
            mismatch_penalty_applied = max(0.0, float(self.config.mismatch_penalty))
            raw_total -= mismatch_penalty_applied
            reasons.append('alignment_mismatch_high_internal_low_external')

        total = max(0.0, min(1.0, raw_total))

        return RewardBreakdown(
            internal=internal,
            external=external,
            total=total,
            reasons=reasons,
            metadata={
                "formula": "total = alpha * internal.total + beta * external.total - mismatch_penalty",
                "alpha": alpha,
                "beta": beta,
                "fallback_strategy": self.config.fallback_strategy,
                "use_power_transform": self.config.use_power_transform,
                "mismatch_internal_threshold": self.config.mismatch_internal_threshold,
                "mismatch_external_threshold": self.config.mismatch_external_threshold,
                "mismatch_penalty_applied": mismatch_penalty_applied,
                "response_text": response.text,
                "intent": response.intent,
                "policy": response.policy,
            },
        )

    def _build_internal(self, response: ResponseResult) -> InternalRewardBreakdown:
        return InternalRewardBreakdown(
            semantic=float(response.score.semantic_consistency),
            slot=float(response.score.slot_fitness),
            grammar=float(response.score.grammar_fitness),
            retention=float(response.score.input_retention),
            policy=float(response.score.policy_fitness),
            total=max(0.0, min(1.0, float(response.score.total))),
            reasons=list(response.score.reasons),
        )

    def _build_external(
        self,
        *,
        internal: InternalRewardBreakdown,
        evaluation: List[EvaluationResult],
    ) -> ExternalRewardBreakdown:
        components: List[ExternalRewardComponent] = []

        if evaluation:
            for item in evaluation:
                score = max(0.0, min(1.0, float(item.score)))
                weighted_score = self._transform_external_score(score)
                components.append(
                    ExternalRewardComponent(
                        evaluator_name=str(item.evaluator_name),
                        score=score,
                        weight=1.0,
                        weighted_score=weighted_score,
                        label=str(item.label),
                        feedback=str(item.feedback),
                        metadata=dict(item.metadata or {}),
                    )
                )
        else:
            fallback_score, evaluator_name, feedback = self._fallback_external(internal)
            components.append(
                ExternalRewardComponent(
                    evaluator_name=evaluator_name,
                    score=fallback_score,
                    weight=1.0,
                    weighted_score=fallback_score,
                    label="fallback",
                    feedback=feedback,
                    metadata={
                        "fallback_strategy": self.config.fallback_strategy,
                    },
                )
            )

        total = sum(component.weighted_score for component in components) / max(1, len(components))
        return ExternalRewardBreakdown(components=components, total=max(0.0, min(1.0, total)))

    def _fallback_external(self, internal: InternalRewardBreakdown) -> tuple[float, str, str]:
        strategy = str(self.config.fallback_strategy or "neutral").strip().lower()
        if strategy == "internal":
            score = max(0.0, min(1.0, float(internal.total)))
            return score, "fallback:internal", "external evaluation missing; internal total reused"

        score = max(0.0, min(1.0, float(self.config.neutral_external_score)))
        return score, "fallback:neutral", "external evaluation missing; neutral fallback applied"

    def _transform_external_score(self, score: float) -> float:
        if not self.config.use_power_transform:
            return score
        power = max(0.1, float(self.config.external_power))
        return max(0.0, min(1.0, score ** power))

    def _normalized_weights(self, alpha: float, beta: float) -> tuple[float, float]:
        alpha = max(0.0, float(alpha))
        beta = max(0.0, float(beta))
        total = alpha + beta
        if total <= 0.0:
            return 0.8, 0.2
        return alpha / total, beta / total
