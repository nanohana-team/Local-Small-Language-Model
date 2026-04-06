from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Protocol

from src.core.schema import EvaluationResult
from src.training.llm_gateway import LLMGateway, extract_json_block

LOGGER = logging.getLogger(__name__)


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
    evaluator_name: str = 'disabled'
    label: str = 'skipped'
    feedback: str = 'external evaluator disabled'

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
                'mode': 'none',
                'input_length': len(user_input),
                'response_length': len(final_response),
            },
        )


@dataclass(slots=True)
class HeuristicExternalEvaluatorConfig:
    non_empty_weight: float = 0.15
    length_weight: float = 0.15
    slot_weight: float = 0.25
    intent_weight: float = 0.20
    target_weight: float = 0.25
    empty_slot_score: float = 0.5
    empty_target_score: float = 0.5
    empty_token_overlap_score: float = 0.5
    question_match_score: float = 1.0
    question_mismatch_score: float = 0.4
    empathy_match_score: float = 1.0
    empathy_mismatch_score: float = 0.5
    default_intent_score: float = 0.9
    punctuated_respond_penalty_score: float = 0.5
    unknown_intent_score: float = 0.7
    exact_echo_penalty: float = 0.25
    partial_echo_penalty: float = 0.10
    empty_length_score: float = 0.0
    short_length_divisor: float = 4.0
    short_length_floor: int = 4
    ideal_length_min: int = 4
    ideal_length_max: int = 64
    long_length_min: int = 65
    long_length_max: int = 128
    long_length_decay_span: float = 96.0
    long_length_base: int = 64
    too_long_score: float = 0.2
    excellent_threshold: float = 0.85
    good_threshold: float = 0.70
    acceptable_threshold: float = 0.50
    weak_threshold: float = 0.30


@dataclass(slots=True)
class HeuristicExternalEvaluator:
    evaluator_name: str = 'heuristic'
    config: HeuristicExternalEvaluatorConfig = field(default_factory=HeuristicExternalEvaluatorConfig)

    def evaluate(
        self,
        *,
        user_input: str,
        final_response: str,
        context: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        context = dict(context or {})
        response = str(final_response or '').strip()
        user_text = str(user_input or '').strip()
        intent = str(context.get('intent', 'unknown'))
        used_slots = dict(context.get('used_slots', {}) or {})
        target_text = str(context.get('target_text', '') or '').strip()

        non_empty = 1.0 if response else 0.0
        length_score = self._length_score(response)
        slot_score = self._slot_score(response, used_slots)
        intent_score = self._intent_score(response, intent)
        target_score = self._target_score(response, target_text)
        echo_penalty = self._echo_penalty(user_text, response)

        raw_total = (
            non_empty * float(self.config.non_empty_weight)
            + length_score * float(self.config.length_weight)
            + slot_score * float(self.config.slot_weight)
            + intent_score * float(self.config.intent_weight)
            + target_score * float(self.config.target_weight)
            - echo_penalty
        )
        total = max(0.0, min(1.0, raw_total))
        label = self._label_from_score(total)
        feedback = (
            f'heuristic_evaluation label={label} total={total:.4f} '
            f'non_empty={non_empty:.4f} length={length_score:.4f} '
            f'slot={slot_score:.4f} intent={intent_score:.4f} target={target_score:.4f} '
            f'echo_penalty={echo_penalty:.4f}'
        )

        return EvaluationResult(
            evaluator_name=self.evaluator_name,
            score=total,
            label=label,
            feedback=feedback,
            metadata={
                'mode': 'heuristic',
                'intent': intent,
                'non_empty': non_empty,
                'length_score': length_score,
                'slot_score': slot_score,
                'intent_score': intent_score,
                'target_score': target_score,
                'echo_penalty': echo_penalty,
                'used_slots': used_slots,
                'target_text': target_text,
            },
        )

    def _length_score(self, response: str) -> float:
        length = len(response)
        if length == 0:
            return float(self.config.empty_length_score)
        if int(self.config.ideal_length_min) <= length <= int(self.config.ideal_length_max):
            return 1.0
        if 1 <= length < int(self.config.short_length_floor):
            divisor = max(1.0, float(self.config.short_length_divisor))
            return max(0.0, length / divisor)
        if int(self.config.long_length_min) <= length <= int(self.config.long_length_max):
            span = max(1.0, float(self.config.long_length_decay_span))
            base = int(self.config.long_length_base)
            return max(0.0, 1.0 - ((length - base) / span))
        return float(self.config.too_long_score)

    def _slot_score(self, response: str, used_slots: Dict[str, str]) -> float:
        values = [str(v).strip() for v in used_slots.values() if str(v).strip()]
        if not values:
            return float(self.config.empty_slot_score)
        matched = sum(1 for value in values if value in response)
        return matched / max(1, len(values))

    def _intent_score(self, response: str, intent: str) -> float:
        if not response:
            return 0.0
        if intent == 'question':
            return float(self.config.question_match_score) if ('?' in response or '？' in response) else float(self.config.question_mismatch_score)
        if intent == 'empathy':
            empathy_markers = ('そう', '大変', 'わか', 'ですね', 'だね', 'よかった', '大丈夫')
            return float(self.config.empathy_match_score) if any(marker in response for marker in empathy_markers) else float(self.config.empathy_mismatch_score)
        if intent in {'respond', 'confirm', 'explain'}:
            if '?' in response or '？' in response:
                return float(self.config.punctuated_respond_penalty_score)
            return float(self.config.default_intent_score)
        return float(self.config.unknown_intent_score)

    def _target_score(self, response: str, target_text: str) -> float:
        if not response or not target_text:
            return float(self.config.empty_target_score)
        response_tokens = set(response.replace('。', ' ').replace('、', ' ').split())
        target_tokens = set(target_text.replace('。', ' ').replace('、', ' ').split())
        if not response_tokens or not target_tokens:
            return float(self.config.empty_token_overlap_score)
        overlap = len(response_tokens & target_tokens) / max(1, len(target_tokens))
        if response == target_text:
            return 1.0
        if target_text in response or response in target_text:
            return max(0.7, overlap)
        return max(0.0, min(1.0, overlap))

    def _echo_penalty(self, user_input: str, response: str) -> float:
        if not user_input or not response:
            return 0.0
        if user_input == response:
            return float(self.config.exact_echo_penalty)
        if response in user_input or user_input in response:
            return float(self.config.partial_echo_penalty)
        return 0.0

    def _label_from_score(self, score: float) -> str:
        if score >= float(self.config.excellent_threshold):
            return 'excellent'
        if score >= float(self.config.good_threshold):
            return 'good'
        if score >= float(self.config.acceptable_threshold):
            return 'acceptable'
        if score >= float(self.config.weak_threshold):
            return 'weak'
        return 'poor'


@dataclass(slots=True)
class LLMExternalEvaluatorConfig:
    temperature: float = 0.1
    max_output_tokens: int = 240


class LLMExternalEvaluator:
    def __init__(
        self,
        gateway: LLMGateway | None = None,
        *,
        config: LLMExternalEvaluatorConfig | None = None,
        heuristic_config: HeuristicExternalEvaluatorConfig | None = None,
    ) -> None:
        self.gateway = gateway or LLMGateway()
        self.config = config or LLMExternalEvaluatorConfig()
        self.heuristic_fallback = HeuristicExternalEvaluator(config=heuristic_config or HeuristicExternalEvaluatorConfig())

    def evaluate(
        self,
        *,
        user_input: str,
        final_response: str,
        context: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        ctx = dict(context or {})
        target_text = str(ctx.get('target_text', '') or '').strip()
        intent = str(ctx.get('intent', 'unknown'))
        used_slots = dict(ctx.get('used_slots', {}) or {})

        slot_lines = '\n'.join(f'- {k}: {v}' for k, v in used_slots.items()) or '- (none)'
        system_prompt = (
            'あなたは日本語対話応答の評価器です。'
            '理想応答 target とモデル応答 response を比較し、構造・自然さ・意図適合を採点してください。'
            '必ずJSONのみを返してください。'
        )
        user_prompt = (
            '次の応答を評価してください。\n\n'
            f'ユーザー入力:\n{user_input}\n\n'
            f'意図:\n{intent}\n\n'
            f'抽出スロット:\n{slot_lines}\n\n'
            f'理想応答 target:\n{target_text}\n\n'
            f'モデル応答 response:\n{final_response}\n\n'
            '返答形式(JSONのみ):\n'
            '{\n'
            '  "score": 0.0,\n'
            '  "label": "excellent|good|acceptable|weak|poor",\n'
            '  "feedback": "短い日本語フィードバック",\n'
            '  "naturalness": 0.0,\n'
            '  "intent_fit": 0.0,\n'
            '  "target_alignment": 0.0,\n'
            '  "slot_alignment": 0.0\n'
            '}\n'
            'score は 0.0〜1.0 の総合点です。'
        )

        try:
            response = self.gateway.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                purpose='external_evaluation',
                temperature=float(self.config.temperature),
                max_output_tokens=max(1, int(self.config.max_output_tokens)),
            )
            payload = self._parse_payload(response.text)
            score = max(0.0, min(1.0, float(payload.get('score', 0.0))))
            label = str(payload.get('label', '')) or self.heuristic_fallback._label_from_score(score)
            feedback = str(payload.get('feedback', '')).strip() or 'llm evaluator completed'
            metadata = {
                'mode': 'llm',
                'provider': response.provider,
                'model': response.model,
                'naturalness': self._safe_float(payload.get('naturalness')),
                'intent_fit': self._safe_float(payload.get('intent_fit')),
                'target_alignment': self._safe_float(payload.get('target_alignment')),
                'slot_alignment': self._safe_float(payload.get('slot_alignment')),
                'target_text': target_text,
                'used_slots': used_slots,
                'raw': response.raw,
            }
            LOGGER.info(
                'external_evaluator.llm.done model=%s score=%.4f label=%s',
                response.model,
                score,
                label,
            )
            return EvaluationResult(
                evaluator_name='llm',
                score=score,
                label=label,
                feedback=feedback,
                metadata=metadata,
            )
        except Exception as exc:
            LOGGER.warning('external_evaluator.llm.fallback error=%s', exc)
            fallback = self.heuristic_fallback.evaluate(
                user_input=user_input,
                final_response=final_response,
                context=context,
            )
            fallback.metadata = dict(fallback.metadata or {})
            fallback.metadata['fallback_from'] = 'llm'
            fallback.metadata['fallback_reason'] = str(exc)
            return fallback

    def _parse_payload(self, text: str) -> dict[str, Any]:
        block = extract_json_block(text)
        payload = json.loads(block)
        if not isinstance(payload, dict):
            raise ValueError('Evaluator output is not a JSON object.')
        return payload

    def _safe_float(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0


def build_external_evaluator(
    mode: str = 'llm',
    *,
    gateway: LLMGateway | None = None,
    llm_config: LLMExternalEvaluatorConfig | None = None,
    heuristic_config: HeuristicExternalEvaluatorConfig | None = None,
) -> BaseExternalEvaluator:
    normalized = str(mode or 'llm').strip().lower()
    if normalized in {'', 'llm', 'teacher'}:
        return LLMExternalEvaluator(
            gateway=gateway,
            config=llm_config,
            heuristic_config=heuristic_config,
        )
    if normalized in {'heuristic', 'rule'}:
        return HeuristicExternalEvaluator(config=heuristic_config or HeuristicExternalEvaluatorConfig())
    if normalized in {'none', 'disabled', 'off'}:
        return NullExternalEvaluator()
    raise ValueError(f'Unsupported external evaluator mode: {mode}')
