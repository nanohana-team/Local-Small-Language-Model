from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.schema import DialogueState, InputState, IntentPlan


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class IntentRule:
    intent: str
    keywords: List[str] = field(default_factory=list)
    min_score: float = 1.0
    confidence_base: float = 0.55
    response_policy_hint: str = "answer"
    required_slots: List[str] = field(default_factory=list)
    optional_slots: List[str] = field(default_factory=list)
    note: str = ""


class IntentPlanner:
    """
    LSLM v3 最小縦スライス用の軽量 Intent Planner。
    今回は「なぜその intent になったか」を DEBUG で追えるように、
    ルール評価の全過程をかなり詳細にログ出力する。
    """

    QUESTION_MARKERS: Tuple[str, ...] = ("?", "？")
    QUESTION_WORDS: Tuple[str, ...] = (
        "なに",
        "何",
        "なん",
        "誰",
        "だれ",
        "どこ",
        "いつ",
        "なぜ",
        "どうして",
        "どう",
        "どんな",
        "どれ",
        "いくつ",
        "かな",
        "か",
    )

    EMPATHY_WORDS: Tuple[str, ...] = (
        "つらい",
        "辛い",
        "しんどい",
        "苦しい",
        "悲しい",
        "寂しい",
        "疲れた",
        "疲れ",
        "不安",
        "怖い",
        "むかつく",
        "イライラ",
        "落ち込む",
        "落ち込んだ",
        "泣きたい",
    )

    EXPLAIN_WORDS: Tuple[str, ...] = (
        "教えて",
        "説明",
        "とは",
        "意味",
        "理由",
        "なぜ",
        "どうして",
        "詳しく",
        "くわしく",
        "解説",
    )

    CONFIRM_WORDS: Tuple[str, ...] = (
        "確認",
        "合ってる",
        "あってる",
        "これでいい",
        "これで良い",
        "大丈夫",
        "ok",
        "OK",
        "問題ない",
        "正しい",
    )

    RESPOND_WORDS: Tuple[str, ...] = (
        "こんにちは",
        "こんばんは",
        "ありがとう",
        "よろしく",
        "了解",
        "うん",
        "はい",
    )

    def __init__(self) -> None:
        self._rules: List[IntentRule] = [
            IntentRule(
                intent="question",
                keywords=list(self.QUESTION_WORDS),
                min_score=1.0,
                confidence_base=0.72,
                response_policy_hint="answer",
                required_slots=[],
                optional_slots=["topic"],
                note="question_word_or_question_marker",
            ),
            IntentRule(
                intent="explain",
                keywords=list(self.EXPLAIN_WORDS),
                min_score=1.0,
                confidence_base=0.70,
                response_policy_hint="answer",
                required_slots=["topic"],
                optional_slots=[],
                note="explain_request_detected",
            ),
            IntentRule(
                intent="confirm",
                keywords=list(self.CONFIRM_WORDS),
                min_score=1.0,
                confidence_base=0.68,
                response_policy_hint="clarify",
                required_slots=[],
                optional_slots=["topic", "state"],
                note="confirmation_request_detected",
            ),
            IntentRule(
                intent="empathy",
                keywords=list(self.EMPATHY_WORDS),
                min_score=1.0,
                confidence_base=0.66,
                response_policy_hint="agree",
                required_slots=[],
                optional_slots=["state", "cause"],
                note="emotion_or_distress_detected",
            ),
            IntentRule(
                intent="respond",
                keywords=list(self.RESPOND_WORDS),
                min_score=1.0,
                confidence_base=0.58,
                response_policy_hint="answer",
                required_slots=[],
                optional_slots=["topic"],
                note="basic_conversation_detected",
            ),
        ]

    def plan(
        self,
        input_state: InputState,
        dialogue_state: Optional[DialogueState] = None,
    ) -> IntentPlan:
        dialogue_state = dialogue_state or DialogueState()

        tokens = self._collect_tokens(input_state)
        raw_text = input_state.raw_text.strip()

        LOGGER.debug(
            "intent_planner.start raw_text=%s tokens=%s normalized_tokens=%s dialogue_state=%s",
            raw_text,
            tokens,
            input_state.normalized_tokens,
            {
                "current_topic": dialogue_state.current_topic,
                "last_subject": dialogue_state.last_subject,
                "last_object": dialogue_state.last_object,
                "intent_history": dialogue_state.inferred_intent_history,
                "referents": dialogue_state.referents,
                "variables": dialogue_state.variables,
            },
        )

        if not raw_text and not tokens:
            plan = IntentPlan(
                intent="unknown",
                confidence=0.0,
                required_slots=[],
                optional_slots=[],
                response_policy_hint="hold",
                note="empty_input",
            )
            LOGGER.debug("intent_planner.empty_input result=%s", plan)
            return plan

        plan = self._plan_by_rules(tokens=tokens, raw_text=raw_text)
        LOGGER.debug("intent_planner.before_refine plan=%s", plan)

        plan = self._refine_with_dialogue_state(
            plan=plan,
            input_state=input_state,
            dialogue_state=dialogue_state,
        )

        LOGGER.debug("intent_planner.result=%s", plan)
        return plan

    def _collect_tokens(self, input_state: InputState) -> List[str]:
        merged: List[str] = []

        LOGGER.debug(
            "intent_planner.collect_tokens raw_tokens=%s raw_normalized_tokens=%s",
            input_state.tokens,
            input_state.normalized_tokens,
        )

        for token in input_state.normalized_tokens:
            token = str(token).strip()
            if token:
                merged.append(token)
                LOGGER.debug("intent_planner.collect_tokens accept_normalized token=%s", token)
            else:
                LOGGER.debug("intent_planner.collect_tokens skip_empty_normalized")

        if not merged:
            LOGGER.debug("intent_planner.collect_tokens normalized_empty_fallback_to_raw")
            for token in input_state.tokens:
                token = str(token).strip()
                if token:
                    merged.append(token)
                    LOGGER.debug("intent_planner.collect_tokens accept_raw token=%s", token)
                else:
                    LOGGER.debug("intent_planner.collect_tokens skip_empty_raw")

        LOGGER.debug("intent_planner.collect_tokens result=%s", merged)
        return merged

    def _plan_by_rules(self, tokens: Sequence[str], raw_text: str) -> IntentPlan:
        rule_scores: Dict[str, float] = {}
        matched_reasons: Dict[str, List[str]] = {}

        LOGGER.debug(
            "intent_planner.plan_by_rules begin raw_text=%s tokens=%s",
            raw_text,
            list(tokens),
        )

        for rule in self._rules:
            score, reasons = self._score_rule(rule=rule, tokens=tokens, raw_text=raw_text)
            rule_scores[rule.intent] = score
            matched_reasons[rule.intent] = reasons
            LOGGER.debug(
                "intent_planner.rule_evaluated intent=%s score=%.4f reasons=%s keywords=%s",
                rule.intent,
                score,
                reasons,
                rule.keywords,
            )

        LOGGER.debug("intent_planner.rule_scores=%s", rule_scores)

        if self._has_question_marker(raw_text):
            best_rule = self._find_rule("question")
            LOGGER.debug(
                "intent_planner.override reason=question_marker raw_text=%s chosen_intent=question",
                raw_text,
            )
            return self._build_plan_from_rule(
                rule=best_rule,
                score=max(rule_scores.get("question", 0.0), 1.5),
                reasons=["question_marker"],
            )

        best_intent, best_score = max(rule_scores.items(), key=lambda x: x[1])
        LOGGER.debug(
            "intent_planner.best_rule intent=%s score=%.4f reasons=%s",
            best_intent,
            best_score,
            matched_reasons.get(best_intent, []),
        )

        if best_score <= 0.0:
            LOGGER.debug(
                "intent_planner.fallback_default intent=respond reason=no_rule_scored_positive"
            )
            return IntentPlan(
                intent="respond",
                confidence=0.51,
                required_slots=[],
                optional_slots=["topic"],
                response_policy_hint="answer",
                note="fallback_default_respond",
            )

        rule = self._find_rule(best_intent)
        return self._build_plan_from_rule(
            rule=rule,
            score=best_score,
            reasons=matched_reasons.get(best_intent, []),
        )

    def _score_rule(
        self,
        rule: IntentRule,
        tokens: Sequence[str],
        raw_text: str,
    ) -> Tuple[float, List[str]]:
        score = 0.0
        reasons: List[str] = []
        token_set = set(tokens)

        LOGGER.debug(
            "intent_planner.score_rule.start intent=%s tokens=%s token_set=%s",
            rule.intent,
            list(tokens),
            sorted(token_set),
        )

        for keyword in rule.keywords:
            if keyword in token_set:
                score += 1.0
                reasons.append(f"token:{keyword}")
                LOGGER.debug(
                    "intent_planner.score_rule.hit intent=%s mode=token keyword=%s add=1.00 running_score=%.4f",
                    rule.intent,
                    keyword,
                    score,
                )
            elif keyword and keyword in raw_text:
                score += 0.75
                reasons.append(f"text:{keyword}")
                LOGGER.debug(
                    "intent_planner.score_rule.hit intent=%s mode=text keyword=%s add=0.75 running_score=%.4f",
                    rule.intent,
                    keyword,
                    score,
                )
            else:
                LOGGER.debug(
                    "intent_planner.score_rule.miss intent=%s keyword=%s",
                    rule.intent,
                    keyword,
                )

        if rule.intent == "question":
            if raw_text.endswith("か") or raw_text.endswith("か。"):
                score += 0.8
                reasons.append("question_like_ending")
                LOGGER.debug(
                    "intent_planner.score_rule.bonus intent=question bonus=0.80 reason=question_like_ending running_score=%.4f",
                    score,
                )
            if any(word in raw_text for word in ("教えて", "知りたい", "分かる", "わかる")):
                score += 0.5
                reasons.append("question_request_phrase")
                LOGGER.debug(
                    "intent_planner.score_rule.bonus intent=question bonus=0.50 reason=question_request_phrase running_score=%.4f",
                    score,
                )

        if rule.intent == "explain":
            if any(word in raw_text for word in ("教えて", "説明して", "解説して")):
                score += 0.7
                reasons.append("explicit_explain_request")
                LOGGER.debug(
                    "intent_planner.score_rule.bonus intent=explain bonus=0.70 reason=explicit_explain_request running_score=%.4f",
                    score,
                )

        if rule.intent == "empathy":
            empathy_hit_count = sum(1 for word in self.EMPATHY_WORDS if word in raw_text)
            LOGGER.debug(
                "intent_planner.score_rule.empathy_hit_count=%s raw_text=%s",
                empathy_hit_count,
                raw_text,
            )
            if empathy_hit_count >= 2:
                score += 0.6
                reasons.append("multiple_emotion_markers")
                LOGGER.debug(
                    "intent_planner.score_rule.bonus intent=empathy bonus=0.60 reason=multiple_emotion_markers running_score=%.4f",
                    score,
                )

        if rule.intent == "confirm":
            if any(word in raw_text for word in ("でいい", "で合ってる", "であってる")):
                score += 0.6
                reasons.append("yes_no_confirmation_like")
                LOGGER.debug(
                    "intent_planner.score_rule.bonus intent=confirm bonus=0.60 reason=yes_no_confirmation_like running_score=%.4f",
                    score,
                )

        LOGGER.debug(
            "intent_planner.score_rule.result intent=%s final_score=%.4f reasons=%s",
            rule.intent,
            score,
            reasons,
        )
        return score, reasons

    def _build_plan_from_rule(
        self,
        rule: IntentRule,
        score: float,
        reasons: List[str],
    ) -> IntentPlan:
        confidence = self._normalize_confidence(
            base=rule.confidence_base,
            score=score,
            min_score=rule.min_score,
        )
        note_parts = [rule.note] if rule.note else []
        note_parts.extend(reasons)

        plan = IntentPlan(
            intent=rule.intent,  # type: ignore[arg-type]
            confidence=confidence,
            required_slots=list(rule.required_slots),
            optional_slots=list(rule.optional_slots),
            response_policy_hint=rule.response_policy_hint,  # type: ignore[arg-type]
            note=" | ".join(note_parts),
        )

        LOGGER.debug(
            "intent_planner.build_plan intent=%s raw_score=%.4f confidence=%.4f required_slots=%s optional_slots=%s note=%s",
            plan.intent,
            score,
            confidence,
            plan.required_slots,
            plan.optional_slots,
            plan.note,
        )
        return plan

    def _refine_with_dialogue_state(
        self,
        plan: IntentPlan,
        input_state: InputState,
        dialogue_state: DialogueState,
    ) -> IntentPlan:
        raw_text = input_state.raw_text
        original_intent = plan.intent
        original_confidence = plan.confidence

        LOGGER.debug(
            "intent_planner.refine.start plan=%s raw_text=%s current_topic=%s intent_history=%s",
            plan,
            raw_text,
            dialogue_state.current_topic,
            dialogue_state.inferred_intent_history,
        )

        if (
            plan.intent == "respond"
            and dialogue_state.inferred_intent_history
            and dialogue_state.inferred_intent_history[-1] == "explain"
            and len(raw_text) <= 20
        ):
            plan.intent = "explain"
            plan.confidence = max(plan.confidence, 0.57)
            if "topic" not in plan.optional_slots:
                plan.optional_slots.append("topic")
            plan.note = f"{plan.note} | dialogue_context_shift_to_explain".strip(" |")
            LOGGER.debug(
                "intent_planner.refine.applied rule=dialogue_context_shift_to_explain old_intent=%s new_intent=%s old_confidence=%.4f new_confidence=%.4f",
                original_intent,
                plan.intent,
                original_confidence,
                plan.confidence,
            )

        if plan.intent == "confirm" and dialogue_state.current_topic:
            old_confidence = plan.confidence
            plan.confidence = min(0.98, plan.confidence + 0.08)
            plan.note = f"{plan.note} | topic_context_available".strip(" |")
            LOGGER.debug(
                "intent_planner.refine.applied rule=topic_context_available old_confidence=%.4f new_confidence=%.4f",
                old_confidence,
                plan.confidence,
            )

        LOGGER.debug("intent_planner.refine.result=%s", plan)
        return plan

    def _find_rule(self, intent: str) -> IntentRule:
        for rule in self._rules:
            if rule.intent == intent:
                return rule
        raise ValueError(f"unknown intent rule: {intent}")

    def _has_question_marker(self, raw_text: str) -> bool:
        has_marker = any(marker in raw_text for marker in self.QUESTION_MARKERS)
        LOGGER.debug(
            "intent_planner.has_question_marker raw_text=%s result=%s",
            raw_text,
            has_marker,
        )
        return has_marker

    def _normalize_confidence(self, base: float, score: float, min_score: float) -> float:
        if score <= 0.0:
            LOGGER.debug(
                "intent_planner.normalize_confidence score_non_positive score=%.4f -> 0.0",
                score,
            )
            return 0.0

        extra = max(0.0, score - min_score)
        confidence = base + (extra * 0.10)
        normalized = max(0.0, min(0.99, confidence))
        LOGGER.debug(
            "intent_planner.normalize_confidence base=%.4f score=%.4f min_score=%.4f extra=%.4f raw=%.4f normalized=%.4f",
            base,
            score,
            min_score,
            extra,
            confidence,
            normalized,
        )
        return normalized


def plan_intent(
    input_state: InputState,
    dialogue_state: Optional[DialogueState] = None,
) -> IntentPlan:
    planner = IntentPlanner()
    return planner.plan(input_state=input_state, dialogue_state=dialogue_state)