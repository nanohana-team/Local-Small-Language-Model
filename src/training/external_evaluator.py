from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Protocol

from src.core.schema import EvaluationResult


class BaseExternalEvaluator(Protocol):
    def evaluate(
        self,
        *,
        user_input: str,
        final_response: str,
        context: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        ...


@dataclass(slots=True)
class NullExternalEvaluator:
    evaluator_name: str = "disabled"
    label: str = "skipped"
    feedback: str = "external evaluator disabled"

    def evaluate(
        self,
        *,
        user_input: str,
        final_response: str,
        context: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            evaluator_name=self.evaluator_name,
            score=0.0,
            label=self.label,
            feedback=self.feedback,
            metadata={
                "mode": "none",
                "input_length": len(user_input),
                "response_length": len(final_response),
            },
        )


@dataclass(slots=True)
class HeuristicExternalEvaluator:
    evaluator_name: str = "heuristic"

    def evaluate(
        self,
        *,
        user_input: str,
        final_response: str,
        context: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        context = dict(context or {})
        response = str(final_response or "").strip()
        user_text = str(user_input or "").strip()
        intent = str(context.get("intent", "unknown"))
        used_slots = dict(context.get("used_slots", {}) or {})

        non_empty = 1.0 if response else 0.0
        length_score = self._length_score(response)
        slot_score = self._slot_score(response, used_slots)
        intent_score = self._intent_score(response, intent)
        echo_penalty = self._echo_penalty(user_text, response)

        raw_total = (
            non_empty * 0.20
            + length_score * 0.20
            + slot_score * 0.35
            + intent_score * 0.25
            - echo_penalty
        )
        total = max(0.0, min(1.0, raw_total))

        label = self._label_from_score(total)
        feedback = (
            f"heuristic_evaluation label={label} total={total:.4f} "
            f"non_empty={non_empty:.4f} length={length_score:.4f} "
            f"slot={slot_score:.4f} intent={intent_score:.4f} echo_penalty={echo_penalty:.4f}"
        )

        return EvaluationResult(
            evaluator_name=self.evaluator_name,
            score=total,
            label=label,
            feedback=feedback,
            metadata={
                "mode": "heuristic",
                "intent": intent,
                "non_empty": non_empty,
                "length_score": length_score,
                "slot_score": slot_score,
                "intent_score": intent_score,
                "echo_penalty": echo_penalty,
                "used_slots": used_slots,
            },
        )

    def _length_score(self, response: str) -> float:
        length = len(response)
        if length == 0:
            return 0.0
        if 4 <= length <= 48:
            return 1.0
        if 1 <= length < 4:
            return max(0.0, length / 4.0)
        if 49 <= length <= 96:
            return max(0.0, 1.0 - ((length - 48) / 64.0))
        return 0.2

    def _slot_score(self, response: str, used_slots: Dict[str, str]) -> float:
        values = [str(v).strip() for v in used_slots.values() if str(v).strip()]
        if not values:
            return 0.5
        matched = sum(1 for value in values if value in response)
        return matched / max(1, len(values))

    def _intent_score(self, response: str, intent: str) -> float:
        if not response:
            return 0.0

        if intent == "question":
            return 1.0 if ("?" in response or "？" in response) else 0.4

        if intent == "empathy":
            empathy_markers = ("そう", "大変", "わか", "ですね", "だね", "よかった", "大丈夫")
            return 1.0 if any(marker in response for marker in empathy_markers) else 0.5

        if intent in {"respond", "confirm", "explain"}:
            if "?" in response or "？" in response:
                return 0.5
            return 0.9

        return 0.7

    def _echo_penalty(self, user_input: str, response: str) -> float:
        if not user_input or not response:
            return 0.0
        if user_input == response:
            return 0.25
        if response in user_input or user_input in response:
            return 0.10
        return 0.0

    def _label_from_score(self, score: float) -> str:
        if score >= 0.85:
            return "excellent"
        if score >= 0.70:
            return "good"
        if score >= 0.50:
            return "acceptable"
        if score >= 0.30:
            return "weak"
        return "poor"


def build_external_evaluator(mode: str = "heuristic") -> BaseExternalEvaluator:
    normalized = str(mode or "heuristic").strip().lower()
    if normalized in {"", "heuristic"}:
        return HeuristicExternalEvaluator()
    if normalized in {"none", "disabled", "off"}:
        return NullExternalEvaluator()
    raise ValueError(f"Unsupported external evaluator mode: {mode}")
