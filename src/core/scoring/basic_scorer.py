from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.schema import (
    FilledSlots,
    InputState,
    IntentPlan,
    RealizationCandidate,
    ResponseResult,
    ScoreBreakdown,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BasicScorerConfig:
    semantic_weight: float = 0.28
    slot_weight: float = 0.24
    grammar_weight: float = 0.18
    retention_weight: float = 0.16
    policy_weight: float = 0.14
    empty_candidate_penalty: float = 0.40
    missing_required_penalty: float = 0.08
    max_grammar_penalty: float = 0.45
    policy_memory_teacher_bonus: float = 0.22
    policy_memory_response_bonus: float = 0.10
    policy_memory_retention_floor: float = 0.75
    diversity_exact_match_penalty: float = 0.18
    diversity_similarity_penalty: float = 0.10
    diversity_similarity_threshold: float = 0.82
    diversity_frequency_bonus_penalty: float = 0.04
    diversity_frequency_cap: float = 0.24


class BasicScorer:
    """
    詳細 DEBUG ログ付き scorer。
    各候補の semantic / slot / grammar / retention / policy の内訳を全部出す。
    """

    def __init__(self, config: Optional[BasicScorerConfig] = None) -> None:
        self.config = config or BasicScorerConfig()

    def choose_best(
        self,
        input_state: InputState,
        intent_plan: IntentPlan,
        filled_slots: FilledSlots,
        candidates: Sequence[RealizationCandidate],
        recent_texts: Sequence[str] | None = None,
    ) -> Tuple[ResponseResult, List[RealizationCandidate]]:
        LOGGER.debug(
            "basic_scorer.start intent=%s policy=%s candidate_count=%s input_tokens=%s filled_slots=%s missing_required=%s config=%s",
            intent_plan.intent,
            intent_plan.response_policy_hint,
            len(candidates),
            input_state.normalized_tokens or input_state.tokens,
            {
                k: {
                    "value": v.value,
                    "confidence": v.confidence,
                    "source_candidate": v.source_candidate,
                    "inferred": v.inferred,
                    "note": v.note,
                }
                for k, v in filled_slots.values.items()
            },
            filled_slots.missing_required,
            self.config,
        )

        if not candidates:
            LOGGER.debug("basic_scorer.no_candidates_using_fallback")
            fallback = self._build_fallback_candidate(intent_plan=intent_plan)
            scored_fallback, breakdown = self._score_candidate(
                candidate=fallback,
                input_state=input_state,
                intent_plan=intent_plan,
                filled_slots=filled_slots,
                recent_texts=recent_texts,
            )
            response = self._build_response_result(
                intent_plan=intent_plan,
                filled_slots=filled_slots,
                candidate=scored_fallback,
                breakdown=breakdown,
            )
            LOGGER.debug("basic_scorer.fallback_response=%s", response)
            return response, [scored_fallback]

        scored_candidates: List[RealizationCandidate] = []
        best_candidate: Optional[RealizationCandidate] = None
        best_breakdown: Optional[ScoreBreakdown] = None
        best_total = -math.inf

        for index, candidate in enumerate(candidates):
            LOGGER.debug(
                "basic_scorer.evaluate_candidate index=%s candidate=%s",
                index,
                candidate,
            )
            scored_candidate, breakdown = self._score_candidate(
                candidate=candidate,
                input_state=input_state,
                intent_plan=intent_plan,
                filled_slots=filled_slots,
                recent_texts=recent_texts,
            )
            scored_candidates.append(scored_candidate)

            LOGGER.debug(
                "basic_scorer.candidate_result index=%s text=%s total=%.6f semantic=%.6f slot=%.6f grammar=%.6f retention=%.6f policy=%.6f reasons=%s",
                index,
                scored_candidate.text,
                breakdown.total,
                breakdown.semantic_consistency,
                breakdown.slot_fitness,
                breakdown.grammar_fitness,
                breakdown.input_retention,
                breakdown.policy_fitness,
                breakdown.reasons,
            )

            if breakdown.total > best_total:
                LOGGER.debug(
                    "basic_scorer.best_update index=%s old_best=%.6f new_best=%.6f text=%s",
                    index,
                    best_total,
                    breakdown.total,
                    scored_candidate.text,
                )
                best_total = breakdown.total
                best_candidate = scored_candidate
                best_breakdown = breakdown
            else:
                LOGGER.debug(
                    "basic_scorer.best_keep index=%s current_best=%.6f candidate_total=%.6f",
                    index,
                    best_total,
                    breakdown.total,
                )

        assert best_candidate is not None
        assert best_breakdown is not None

        response = self._build_response_result(
            intent_plan=intent_plan,
            filled_slots=filled_slots,
            candidate=best_candidate,
            breakdown=best_breakdown,
        )

        LOGGER.debug(
            "basic_scorer.result chosen_candidate=%s chosen_score=%s response=%s",
            best_candidate,
            best_breakdown,
            response,
        )
        return response, scored_candidates

    def _score_candidate(
        self,
        candidate: RealizationCandidate,
        input_state: InputState,
        intent_plan: IntentPlan,
        filled_slots: FilledSlots,
        recent_texts: Sequence[str] | None = None,
    ) -> Tuple[RealizationCandidate, ScoreBreakdown]:
        LOGGER.debug(
            "basic_scorer.score_candidate.start text=%s template_id=%s token_sequence=%s grammar_violations=%s slot_coverage=%.6f semantic_score=%.6f",
            candidate.text,
            candidate.template_id,
            candidate.token_sequence,
            candidate.grammar_violations,
            candidate.slot_coverage,
            candidate.semantic_score,
        )

        semantic_consistency = self._score_semantic_consistency(candidate, filled_slots)
        slot_fitness = self._score_slot_fitness(candidate, filled_slots)
        grammar_fitness = self._score_grammar_fitness(candidate, filled_slots)
        input_retention = self._score_input_retention(candidate, input_state)
        policy_fitness = self._score_policy_fitness(candidate, intent_plan)

        weighted_semantic = semantic_consistency * self.config.semantic_weight
        weighted_slot = slot_fitness * self.config.slot_weight
        weighted_grammar = grammar_fitness * self.config.grammar_weight
        weighted_retention = input_retention * self.config.retention_weight
        weighted_policy = policy_fitness * self.config.policy_weight

        total = (
            weighted_semantic
            + weighted_slot
            + weighted_grammar
            + weighted_retention
            + weighted_policy
        )
        diversity_penalty = self._score_diversity_penalty(candidate, recent_texts)
        total -= diversity_penalty

        LOGGER.debug(
            "basic_scorer.score_candidate.weighted text=%s weighted_semantic=%.6f weighted_slot=%.6f weighted_grammar=%.6f weighted_retention=%.6f weighted_policy=%.6f diversity_penalty=%.6f total_before_clamp=%.6f",
            candidate.text,
            weighted_semantic,
            weighted_slot,
            weighted_grammar,
            weighted_retention,
            weighted_policy,
            diversity_penalty,
            total,
        )

        reasons = self._build_reasons(
            candidate=candidate,
            filled_slots=filled_slots,
            semantic_consistency=semantic_consistency,
            slot_fitness=slot_fitness,
            grammar_fitness=grammar_fitness,
            input_retention=input_retention,
            policy_fitness=policy_fitness,
            diversity_penalty=diversity_penalty,
        )

        clamped_total = max(0.0, min(1.0, total))

        scored_candidate = RealizationCandidate(
            text=candidate.text,
            token_sequence=list(candidate.token_sequence),
            template_id=candidate.template_id,
            grammar_violations=list(candidate.grammar_violations),
            slot_coverage=candidate.slot_coverage,
            semantic_score=candidate.semantic_score,
            final_score=clamped_total,
        )

        breakdown = ScoreBreakdown(
            semantic_consistency=semantic_consistency,
            slot_fitness=slot_fitness,
            grammar_fitness=grammar_fitness,
            input_retention=input_retention,
            policy_fitness=policy_fitness,
            total=clamped_total,
            reasons=reasons,
        )

        LOGGER.debug(
            "basic_scorer.score_candidate.result text=%s scored_candidate=%s breakdown=%s",
            candidate.text,
            scored_candidate,
            breakdown,
        )
        return scored_candidate, breakdown

    def _score_semantic_consistency(
        self,
        candidate: RealizationCandidate,
        filled_slots: FilledSlots,
    ) -> float:
        score = float(candidate.semantic_score)
        matched_slots = 0
        total_slots = 0

        LOGGER.debug(
            "basic_scorer.score_semantic_consistency.start text=%s base_score=%.6f",
            candidate.text,
            score,
        )

        for slot in filled_slots.values.values():
            if not slot.value:
                LOGGER.debug(
                    "basic_scorer.score_semantic_consistency.skip slot=%s reason=empty_value",
                    slot.slot_name,
                )
                continue
            total_slots += 1
            if slot.value in candidate.text:
                matched_slots += 1
                LOGGER.debug(
                    "basic_scorer.score_semantic_consistency.match slot=%s value=%s",
                    slot.slot_name,
                    slot.value,
                )
            else:
                LOGGER.debug(
                    "basic_scorer.score_semantic_consistency.miss slot=%s value=%s",
                    slot.slot_name,
                    slot.value,
                )

        if total_slots > 0:
            score = (score * 0.70) + ((matched_slots / float(total_slots)) * 0.30)
            LOGGER.debug(
                "basic_scorer.score_semantic_consistency.blend matched_slots=%s total_slots=%s blended=%.6f",
                matched_slots,
                total_slots,
                score,
            )

        final = self._clamp01(score)
        LOGGER.debug(
            "basic_scorer.score_semantic_consistency.result=%.6f",
            final,
        )
        return final

    def _score_slot_fitness(
        self,
        candidate: RealizationCandidate,
        filled_slots: FilledSlots,
    ) -> float:
        base = float(candidate.slot_coverage)

        LOGGER.debug(
            "basic_scorer.score_slot_fitness.start text=%s base_slot_coverage=%.6f constraints=%s",
            candidate.text,
            base,
            [
                {
                    "name": c.name,
                    "required": c.required,
                }
                for c in filled_slots.frame.constraints
            ],
        )

        if filled_slots.frame.constraints:
            required_total = 0
            required_hit = 0

            for constraint in filled_slots.frame.constraints:
                if not constraint.required:
                    LOGGER.debug(
                        "basic_scorer.score_slot_fitness.skip_optional constraint=%s",
                        constraint.name,
                    )
                    continue
                required_total += 1

                slot_value = filled_slots.values.get(constraint.name)
                if slot_value and slot_value.value and slot_value.value in candidate.text:
                    required_hit += 1
                    LOGGER.debug(
                        "basic_scorer.score_slot_fitness.required_hit constraint=%s value=%s",
                        constraint.name,
                        slot_value.value,
                    )
                else:
                    LOGGER.debug(
                        "basic_scorer.score_slot_fitness.required_miss constraint=%s value=%s",
                        constraint.name,
                        slot_value.value if slot_value else None,
                    )

            if required_total > 0:
                required_ratio = required_hit / float(required_total)
                base = (base * 0.55) + (required_ratio * 0.45)
                LOGGER.debug(
                    "basic_scorer.score_slot_fitness.blend required_hit=%s required_total=%s required_ratio=%.6f blended=%.6f",
                    required_hit,
                    required_total,
                    required_ratio,
                    base,
                )

        penalty = min(
            self.config.missing_required_penalty * len(filled_slots.missing_required),
            0.30,
        )
        LOGGER.debug(
            "basic_scorer.score_slot_fitness.penalty missing_required=%s penalty=%.6f",
            filled_slots.missing_required,
            penalty,
        )

        final = self._clamp01(base - penalty)
        LOGGER.debug(
            "basic_scorer.score_slot_fitness.result=%.6f",
            final,
        )
        return final

    def _score_grammar_fitness(
        self,
        candidate: RealizationCandidate,
        filled_slots: FilledSlots,
    ) -> float:
        score = 1.0
        LOGGER.debug(
            "basic_scorer.score_grammar_fitness.start text=%s grammar_violations=%s",
            candidate.text,
            candidate.grammar_violations,
        )

        if not candidate.text.strip():
            score -= self.config.empty_candidate_penalty
            LOGGER.debug(
                "basic_scorer.score_grammar_fitness.sub reason=empty_text penalty=%.6f running=%.6f",
                self.config.empty_candidate_penalty,
                score,
            )

        penalty = min(
            len(candidate.grammar_violations) * 0.12,
            self.config.max_grammar_penalty,
        )
        score -= penalty
        LOGGER.debug(
            "basic_scorer.score_grammar_fitness.sub reason=grammar_violations count=%s penalty=%.6f running=%.6f",
            len(candidate.grammar_violations),
            penalty,
            score,
        )

        text = candidate.text
        if not text.endswith(("。", "？", "!", "！")):
            score -= 0.08
            LOGGER.debug(
                "basic_scorer.score_grammar_fitness.sub reason=no_terminal_punct penalty=0.08 running=%.6f",
                score,
            )

        if len(text) <= 1:
            score -= 0.20
            LOGGER.debug(
                "basic_scorer.score_grammar_fitness.sub reason=too_short penalty=0.20 running=%.6f",
                score,
            )

        if "  " in text:
            score -= 0.05
            LOGGER.debug(
                "basic_scorer.score_grammar_fitness.sub reason=double_space penalty=0.05 running=%.6f",
                score,
            )

        if filled_slots.frame.predicate and filled_slots.frame.predicate not in text:
            score -= 0.05
            LOGGER.debug(
                "basic_scorer.score_grammar_fitness.sub reason=predicate_not_realized predicate=%s penalty=0.05 running=%.6f",
                filled_slots.frame.predicate,
                score,
            )

        final = self._clamp01(score)
        LOGGER.debug(
            "basic_scorer.score_grammar_fitness.result=%.6f",
            final,
        )
        return final

    def _score_input_retention(
        self,
        candidate: RealizationCandidate,
        input_state: InputState,
    ) -> float:
        tokens = self._collect_input_tokens(input_state)
        LOGGER.debug(
            "basic_scorer.score_input_retention.start text=%s tokens=%s",
            candidate.text,
            tokens,
        )

        if not tokens:
            LOGGER.debug("basic_scorer.score_input_retention.no_tokens -> 0.50")
            return 0.50

        text = candidate.text
        matched = 0
        total = 0

        for token in tokens:
            if not token:
                LOGGER.debug("basic_scorer.score_input_retention.skip reason=empty_token")
                continue
            if self._is_trivial_token(token):
                LOGGER.debug(
                    "basic_scorer.score_input_retention.skip token=%s reason=trivial",
                    token,
                )
                continue
            total += 1
            if token in text:
                matched += 1
                LOGGER.debug(
                    "basic_scorer.score_input_retention.match token=%s",
                    token,
                )
            else:
                LOGGER.debug(
                    "basic_scorer.score_input_retention.miss token=%s",
                    token,
                )

        if total == 0:
            LOGGER.debug("basic_scorer.score_input_retention.no_nontrivial_tokens -> 0.50")
            return 0.50

        score = self._clamp01(matched / float(total))
        if self._is_policy_memory_candidate(candidate):
            if candidate.slot_coverage >= 0.80 and candidate.semantic_score >= 0.80:
                adjusted = max(score, float(self.config.policy_memory_retention_floor))
                if adjusted != score:
                    LOGGER.debug(
                        "basic_scorer.score_input_retention.policy_memory_floor old=%.6f new=%.6f",
                        score,
                        adjusted,
                    )
                score = adjusted
        LOGGER.debug(
            "basic_scorer.score_input_retention.result matched=%s total=%s score=%.6f",
            matched,
            total,
            score,
        )
        return score

    def _score_policy_fitness(
        self,
        candidate: RealizationCandidate,
        intent_plan: IntentPlan,
    ) -> float:
        text = candidate.text
        intent = intent_plan.intent
        policy = intent_plan.response_policy_hint
        score = 0.60

        LOGGER.debug(
            "basic_scorer.score_policy_fitness.start text=%s intent=%s policy=%s base=%.6f",
            text,
            intent,
            policy,
            score,
        )

        if intent == "empathy":
            if "ね" in text or "大変" in text or "つら" in text:
                score += 0.22
                LOGGER.debug(
                    "basic_scorer.score_policy_fitness.add reason=empathy_tone add=0.22 running=%.6f",
                    score,
                )
            if "?" in text or "？" in text:
                score -= 0.06
                LOGGER.debug(
                    "basic_scorer.score_policy_fitness.sub reason=empathy_question_mark penalty=0.06 running=%.6f",
                    score,
                )

        elif intent == "confirm":
            if "大丈夫" in text or "合って" in text or "問題ありません" in text:
                score += 0.22
                LOGGER.debug(
                    "basic_scorer.score_policy_fitness.add reason=confirm_tone add=0.22 running=%.6f",
                    score,
                )

        elif intent == "explain":
            if "説明" in text or "整理" in text or "関係" in text or "捉え" in text:
                score += 0.18
                LOGGER.debug(
                    "basic_scorer.score_policy_fitness.add reason=explain_tone add=0.18 running=%.6f",
                    score,
                )

        elif intent == "question":
            if text.endswith("。"):
                score += 0.08
                LOGGER.debug(
                    "basic_scorer.score_policy_fitness.add reason=question_answer_style add=0.08 running=%.6f",
                    score,
                )
            if "分かりません" in text or "不明" in text:
                score -= 0.10
                LOGGER.debug(
                    "basic_scorer.score_policy_fitness.sub reason=question_uncertain_penalty penalty=0.10 running=%.6f",
                    score,
                )

        elif intent == "respond":
            if "ですね" in text or "です" in text:
                score += 0.12
                LOGGER.debug(
                    "basic_scorer.score_policy_fitness.add reason=respond_tone add=0.12 running=%.6f",
                    score,
                )

        if self._is_policy_memory_teacher_candidate(candidate):
            score += float(self.config.policy_memory_teacher_bonus)
            LOGGER.debug(
                "basic_scorer.score_policy_fitness.add reason=policy_memory_teacher_bonus add=%.2f running=%.6f",
                self.config.policy_memory_teacher_bonus,
                score,
            )
        elif self._is_policy_memory_response_candidate(candidate):
            score += float(self.config.policy_memory_response_bonus)
            LOGGER.debug(
                "basic_scorer.score_policy_fitness.add reason=policy_memory_response_bonus add=%.2f running=%.6f",
                self.config.policy_memory_response_bonus,
                score,
            )

        if policy == "agree" and ("ね" in text or "大変" in text):
            score += 0.08
            LOGGER.debug(
                "basic_scorer.score_policy_fitness.add reason=policy_agree_match add=0.08 running=%.6f",
                score,
            )
        elif policy == "clarify" and ("確認" in text or "大丈夫" in text or "合って" in text):
            score += 0.08
            LOGGER.debug(
                "basic_scorer.score_policy_fitness.add reason=policy_clarify_match add=0.08 running=%.6f",
                score,
            )
        elif policy == "answer" and text.endswith(("。", "！")):
            score += 0.05
            LOGGER.debug(
                "basic_scorer.score_policy_fitness.add reason=policy_answer_match add=0.05 running=%.6f",
                score,
            )

        final = self._clamp01(score)
        LOGGER.debug(
            "basic_scorer.score_policy_fitness.result=%.6f",
            final,
        )
        return final


    def _is_policy_memory_candidate(self, candidate: RealizationCandidate) -> bool:
        return str(candidate.template_id or '').startswith('policy_memory_')

    def _is_policy_memory_teacher_candidate(self, candidate: RealizationCandidate) -> bool:
        return str(candidate.template_id or '').startswith('policy_memory_teacher_target')

    def _is_policy_memory_response_candidate(self, candidate: RealizationCandidate) -> bool:
        return str(candidate.template_id or '').startswith('policy_memory_selected_response')

    def _score_diversity_penalty(
        self,
        candidate: RealizationCandidate,
        recent_texts: Sequence[str] | None,
    ) -> float:
        normalized_candidate = self._normalize_text(candidate.text)
        if not normalized_candidate or not recent_texts:
            return 0.0

        normalized_recent = [self._normalize_text(text) for text in recent_texts if self._normalize_text(text)]
        if not normalized_recent:
            return 0.0

        exact_hits = sum(1 for text in normalized_recent if text == normalized_candidate)
        penalty = 0.0
        if exact_hits > 0:
            penalty = float(self.config.diversity_exact_match_penalty)
            penalty += min(
                float(self.config.diversity_frequency_cap),
                max(0, exact_hits - 1) * float(self.config.diversity_frequency_bonus_penalty),
            )

        similarity_threshold = float(self.config.diversity_similarity_threshold)
        for recent_text in normalized_recent:
            similarity = self._bigram_jaccard(normalized_candidate, recent_text)
            if similarity >= similarity_threshold:
                penalty = max(
                    penalty,
                    min(
                        float(self.config.diversity_frequency_cap),
                        float(self.config.diversity_similarity_penalty) * similarity,
                    ),
                )

        final_penalty = self._clamp01(penalty)
        LOGGER.debug(
            "basic_scorer.score_diversity_penalty text=%s exact_hits=%s penalty=%.6f recent=%s",
            candidate.text,
            exact_hits,
            final_penalty,
            normalized_recent,
        )
        return final_penalty

    def _normalize_text(self, text: str) -> str:
        return str(text or '').strip().rstrip('。？！!?')

    def _bigram_jaccard(self, left: str, right: str) -> float:
        left_bigrams = self._char_bigrams(left)
        right_bigrams = self._char_bigrams(right)
        if not left_bigrams or not right_bigrams:
            return 0.0
        intersection = len(left_bigrams & right_bigrams)
        union = len(left_bigrams | right_bigrams)
        if union <= 0:
            return 0.0
        return intersection / float(union)

    def _char_bigrams(self, text: str) -> set[str]:
        if len(text) < 2:
            return {text} if text else set()
        return {text[index:index + 2] for index in range(len(text) - 1)}

    def _build_reasons(
        self,
        candidate: RealizationCandidate,
        filled_slots: FilledSlots,
        semantic_consistency: float,
        slot_fitness: float,
        grammar_fitness: float,
        input_retention: float,
        policy_fitness: float,
        diversity_penalty: float = 0.0,
    ) -> List[str]:
        reasons: List[str] = []
        LOGGER.debug(
            "basic_scorer.build_reasons.start text=%s semantic=%.6f slot=%.6f grammar=%.6f retention=%.6f policy=%.6f",
            candidate.text,
            semantic_consistency,
            slot_fitness,
            grammar_fitness,
            input_retention,
            policy_fitness,
        )

        if semantic_consistency >= 0.75:
            reasons.append("semantic_consistency_high")
        elif semantic_consistency < 0.45:
            reasons.append("semantic_consistency_low")

        if slot_fitness >= 0.75:
            reasons.append("slot_fitness_high")
        elif filled_slots.missing_required:
            reasons.append("missing_required_slots")

        if grammar_fitness >= 0.85:
            reasons.append("grammar_clean")
        if candidate.grammar_violations:
            reasons.append(f"grammar_violations:{','.join(candidate.grammar_violations)}")

        if input_retention >= 0.60:
            reasons.append("input_retention_good")
        elif input_retention < 0.25:
            reasons.append("input_retention_low")

        if policy_fitness >= 0.75:
            reasons.append("policy_fit_good")
        if diversity_penalty >= self.config.diversity_exact_match_penalty:
            reasons.append("diversity_penalty_exact_repeat")
        elif diversity_penalty > 0.0:
            reasons.append("diversity_penalty_similar_recent")
        if self._is_policy_memory_candidate(candidate):
            reasons.append("policy_memory_candidate")

        LOGGER.debug(
            "basic_scorer.build_reasons.result reasons=%s",
            reasons,
        )
        return reasons

    def _build_response_result(
        self,
        intent_plan: IntentPlan,
        filled_slots: FilledSlots,
        candidate: RealizationCandidate,
        breakdown: ScoreBreakdown,
    ) -> ResponseResult:
        used_relations: List[str] = []
        used_slots: Dict[str, str] = {
            slot_name: slot_value.value
            for slot_name, slot_value in filled_slots.values.items()
            if slot_value.value and slot_value.value in candidate.text
        }

        response = ResponseResult(
            text=candidate.text,
            intent=intent_plan.intent,
            policy=intent_plan.response_policy_hint,
            chosen_candidate=candidate,
            score=breakdown,
            used_relations=used_relations,
            used_slots=used_slots,
        )
        LOGGER.debug(
            "basic_scorer.build_response_result candidate_text=%s used_slots=%s response=%s",
            candidate.text,
            used_slots,
            response,
        )
        return response

    def _build_fallback_candidate(self, intent_plan: IntentPlan) -> RealizationCandidate:
        if intent_plan.intent == "empathy":
            text = "大変でしたね。"
        elif intent_plan.intent == "confirm":
            text = "その理解で大丈夫です。"
        elif intent_plan.intent == "explain":
            text = "その内容について説明できます。"
        else:
            text = "そうですね。"

        candidate = RealizationCandidate(
            text=text,
            token_sequence=[text],
            template_id="basic_fallback",
            grammar_violations=[],
            slot_coverage=0.0,
            semantic_score=0.45,
            final_score=0.0,
        )
        LOGGER.debug("basic_scorer.build_fallback_candidate result=%s", candidate)
        return candidate

    def _collect_input_tokens(self, input_state: InputState) -> List[str]:
        tokens = [str(t).strip() for t in input_state.normalized_tokens if str(t).strip()]
        if tokens:
            LOGGER.debug("basic_scorer.collect_input_tokens source=normalized tokens=%s", tokens)
            return tokens

        fallback = [str(t).strip() for t in input_state.tokens if str(t).strip()]
        LOGGER.debug("basic_scorer.collect_input_tokens source=raw tokens=%s", fallback)
        return fallback

    def _is_trivial_token(self, token: str) -> bool:
        result = token in {
            "は",
            "が",
            "を",
            "に",
            "で",
            "へ",
            "と",
            "も",
            "の",
            "か",
            "？",
            "?",
            "。",
            "、",
        }
        LOGGER.debug("basic_scorer.is_trivial_token token=%s result=%s", token, result)
        return result

    def _clamp01(self, value: float) -> float:
        clamped = max(0.0, min(1.0, float(value)))
        LOGGER.debug("basic_scorer.clamp01 input=%.6f output=%.6f", value, clamped)
        return clamped


def choose_best_response(
    input_state: InputState,
    intent_plan: IntentPlan,
    filled_slots: FilledSlots,
    candidates: Sequence[RealizationCandidate],
    config: Optional[BasicScorerConfig] = None,
    recent_texts: Sequence[str] | None = None,
) -> Tuple[ResponseResult, List[RealizationCandidate]]:
    scorer = BasicScorer(config=config)
    return scorer.choose_best(
        input_state=input_state,
        intent_plan=intent_plan,
        filled_slots=filled_slots,
        candidates=candidates,
    )