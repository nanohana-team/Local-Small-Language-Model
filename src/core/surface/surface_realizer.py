from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.core.schema import (
    FilledSlots,
    IntentPlan,
    RealizationCandidate,
    SurfacePlan,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SurfaceRealizerConfig:
    default_style: str = "neutral"
    default_politeness: str = "plain"
    max_candidates: int = 4
    include_question_variant: bool = True
    include_soft_variant: bool = True


class SurfaceRealizer:
    """
    詳細 DEBUG ログ付き Surface Realizer。
    どの slot からどのテンプレート候補を作ったかを追跡できる。
    """

    def __init__(self, config: Optional[SurfaceRealizerConfig] = None) -> None:
        self.config = config or SurfaceRealizerConfig()

    def realize(
        self,
        filled_slots: FilledSlots,
        intent_plan: Optional[IntentPlan] = None,
    ) -> Tuple[SurfacePlan, List[RealizationCandidate]]:
        intent_plan = intent_plan or IntentPlan(intent="unknown")

        LOGGER.debug(
            "surface_realizer.start intent=%s response_policy_hint=%s predicate=%s predicate_type=%s filled_slots=%s missing_required=%s optional_unfilled=%s config=%s",
            intent_plan.intent,
            intent_plan.response_policy_hint,
            filled_slots.frame.predicate,
            filled_slots.frame.predicate_type,
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
            filled_slots.optional_unfilled,
            self.config,
        )

        plan = self._build_surface_plan(
            filled_slots=filled_slots,
            intent_plan=intent_plan,
        )
        LOGGER.debug("surface_realizer.plan=%s", plan)

        candidates = self._build_candidates(
            filled_slots=filled_slots,
            intent_plan=intent_plan,
            plan=plan,
        )

        if not candidates:
            fallback_text = self._fallback_text(intent_plan=intent_plan, filled_slots=filled_slots)
            LOGGER.debug(
                "surface_realizer.no_candidates fallback_text=%s",
                fallback_text,
            )
            candidates = [
                RealizationCandidate(
                    text=fallback_text,
                    token_sequence=self._simple_tokenize(fallback_text),
                    template_id="fallback_minimal",
                    grammar_violations=[],
                    slot_coverage=self._slot_coverage(filled_slots),
                    semantic_score=0.40,
                    final_score=0.40,
                )
            ]

        trimmed = candidates[: self.config.max_candidates]
        LOGGER.debug(
            "surface_realizer.result template_id=%s candidate_count_before_trim=%s candidate_count_after_trim=%s candidates=%s",
            plan.template_id,
            len(candidates),
            len(trimmed),
            [
                {
                    "text": c.text,
                    "template_id": c.template_id,
                    "token_sequence": c.token_sequence,
                    "grammar_violations": c.grammar_violations,
                    "slot_coverage": c.slot_coverage,
                    "semantic_score": c.semantic_score,
                    "final_score": c.final_score,
                }
                for c in trimmed
            ],
        )

        return plan, trimmed

    def _build_surface_plan(
        self,
        filled_slots: FilledSlots,
        intent_plan: IntentPlan,
    ) -> SurfacePlan:
        order = self._default_order_for_intent(intent_plan.intent)
        template_id = self._template_id_for_intent(intent_plan.intent, filled_slots)
        style = self.config.default_style
        politeness = self.config.default_politeness

        LOGGER.debug(
            "surface_realizer.build_plan.begin intent=%s predicate_type=%s default_style=%s default_politeness=%s",
            intent_plan.intent,
            filled_slots.frame.predicate_type,
            style,
            politeness,
        )

        if intent_plan.intent == "empathy":
            style = "warm"
        elif intent_plan.intent == "explain":
            style = "informative"
        elif intent_plan.intent == "confirm":
            style = "checking"
        elif intent_plan.intent == "question":
            style = "answering"

        plan = SurfacePlan(
            template_id=template_id,
            style=style,
            politeness=politeness,
            sentence_count=1,
            order=order,
            auxiliaries=[],
            note=f"surface_plan_for:{intent_plan.intent}",
        )
        LOGGER.debug("surface_realizer.build_plan.result=%s", plan)
        return plan

    def _build_candidates(
        self,
        filled_slots: FilledSlots,
        intent_plan: IntentPlan,
        plan: SurfacePlan,
    ) -> List[RealizationCandidate]:
        LOGGER.debug(
            "surface_realizer.build_candidates.begin intent=%s template_id=%s",
            intent_plan.intent,
            plan.template_id,
        )

        if intent_plan.intent == "empathy":
            texts = self._realize_empathy(filled_slots)
        elif intent_plan.intent == "question":
            texts = self._realize_question_answer(filled_slots)
        elif intent_plan.intent == "explain":
            texts = self._realize_explain(filled_slots)
        elif intent_plan.intent == "confirm":
            texts = self._realize_confirm(filled_slots)
        else:
            texts = self._realize_respond(filled_slots)

        LOGGER.debug("surface_realizer.build_candidates.raw_texts=%s", texts)

        unique_texts = self._dedupe_keep_order(texts)
        LOGGER.debug("surface_realizer.build_candidates.unique_texts=%s", unique_texts)

        slot_coverage = self._slot_coverage(filled_slots)
        LOGGER.debug("surface_realizer.build_candidates.slot_coverage=%.6f", slot_coverage)

        candidates: List[RealizationCandidate] = []
        for index, text in enumerate(unique_texts):
            if not text:
                LOGGER.debug(
                    "surface_realizer.build_candidates.skip index=%s reason=empty_text",
                    index,
                )
                continue

            template_id = f"{plan.template_id}_v{index + 1}"
            grammar_violations = self._quick_grammar_checks(text)
            semantic_score = self._estimate_semantic_score(
                text=text,
                filled_slots=filled_slots,
                intent_plan=intent_plan,
            )
            token_sequence = self._simple_tokenize(text)

            candidate = RealizationCandidate(
                text=text,
                token_sequence=token_sequence,
                template_id=template_id,
                grammar_violations=grammar_violations,
                slot_coverage=slot_coverage,
                semantic_score=semantic_score,
                final_score=0.0,
            )
            LOGGER.debug(
                "surface_realizer.build_candidates.add index=%s candidate=%s",
                index,
                candidate,
            )
            candidates.append(candidate)

        return candidates

    def _realize_respond(self, filled_slots: FilledSlots) -> List[str]:
        topic = self._get_value(filled_slots, "topic")
        state = self._get_value(filled_slots, "state")
        actor = self._get_value(filled_slots, "actor")
        predicate = self._get_value(filled_slots, "predicate")
        target = self._get_value(filled_slots, "target")
        location = self._get_value(filled_slots, "location")
        time = self._get_value(filled_slots, "time")

        LOGGER.debug(
            "surface_realizer.realize_respond inputs topic=%s state=%s actor=%s predicate=%s target=%s location=%s time=%s",
            topic,
            state,
            actor,
            predicate,
            target,
            location,
            time,
        )

        texts: List[str] = []

        if topic and state:
            texts.append(f"{topic}は{state}です。")
            texts.append(f"{topic}は{state}ですね。")
            LOGGER.debug("surface_realizer.realize_respond.add reason=topic_and_state")

        if actor and target and predicate:
            texts.append(f"{actor}が{target}を{predicate}ます。")
            LOGGER.debug("surface_realizer.realize_respond.add reason=actor_target_predicate")

        if actor and predicate and location:
            texts.append(f"{actor}が{location}で{predicate}ます。")
            LOGGER.debug("surface_realizer.realize_respond.add reason=actor_predicate_location")

        if topic and predicate:
            texts.append(f"{topic}については{predicate}です。")
            LOGGER.debug("surface_realizer.realize_respond.add reason=topic_and_predicate")

        if topic:
            texts.append(f"{topic}ですね。")
            LOGGER.debug("surface_realizer.realize_respond.add reason=topic_only")

        if state:
            texts.append(f"{state}です。")
            LOGGER.debug("surface_realizer.realize_respond.add reason=state_only")

        if time and topic:
            texts.append(f"{time}の{topic}ですね。")
            LOGGER.debug("surface_realizer.realize_respond.add reason=time_and_topic")

        return texts

    def _realize_question_answer(self, filled_slots: FilledSlots) -> List[str]:
        topic = self._get_value(filled_slots, "topic")
        state = self._get_value(filled_slots, "state")
        predicate = self._get_value(filled_slots, "predicate")
        actor = self._get_value(filled_slots, "actor")
        target = self._get_value(filled_slots, "target")

        LOGGER.debug(
            "surface_realizer.realize_question_answer inputs topic=%s state=%s predicate=%s actor=%s target=%s",
            topic,
            state,
            predicate,
            actor,
            target,
        )

        texts: List[str] = []

        if topic and state:
            texts.append(f"{topic}は{state}です。")
            texts.append(f"{topic}は{state}だと思います。")
            LOGGER.debug("surface_realizer.realize_question_answer.add reason=topic_and_state")

        if actor and target and predicate:
            texts.append(f"{actor}が{target}を{predicate}ます。")
            LOGGER.debug("surface_realizer.realize_question_answer.add reason=actor_target_predicate")

        if topic and predicate:
            texts.append(f"{topic}については{predicate}と考えられます。")
            texts.append(f"{topic}については{predicate}です。")
            LOGGER.debug("surface_realizer.realize_question_answer.add reason=topic_and_predicate")

        if topic:
            texts.append(f"{topic}に関する話ですね。")
            LOGGER.debug("surface_realizer.realize_question_answer.add reason=topic_only")

        texts.append("確認できる範囲ではそのように見えます。")
        LOGGER.debug("surface_realizer.realize_question_answer.add reason=generic_tail")
        return texts

    def _realize_explain(self, filled_slots: FilledSlots) -> List[str]:
        topic = self._get_value(filled_slots, "topic")
        predicate = self._get_value(filled_slots, "predicate")
        state = self._get_value(filled_slots, "state")
        cause = self._get_value(filled_slots, "cause")
        location = self._get_value(filled_slots, "location")
        time = self._get_value(filled_slots, "time")

        LOGGER.debug(
            "surface_realizer.realize_explain inputs topic=%s predicate=%s state=%s cause=%s location=%s time=%s",
            topic,
            predicate,
            state,
            cause,
            location,
            time,
        )

        texts: List[str] = []

        if topic and state:
            texts.append(f"{topic}は{state}という状態です。")
            LOGGER.debug("surface_realizer.realize_explain.add reason=topic_and_state")

        if topic and predicate:
            texts.append(f"{topic}は{predicate}ことに関係します。")
            texts.append(f"{topic}については{predicate}と説明できます。")
            LOGGER.debug("surface_realizer.realize_explain.add reason=topic_and_predicate")

        if topic and cause:
            texts.append(f"{topic}は{cause}が理由として考えられます。")
            LOGGER.debug("surface_realizer.realize_explain.add reason=topic_and_cause")

        if topic and location:
            texts.append(f"{topic}は{location}に関係する話です。")
            LOGGER.debug("surface_realizer.realize_explain.add reason=topic_and_location")

        if topic and time:
            texts.append(f"{topic}は{time}に関わる内容です。")
            LOGGER.debug("surface_realizer.realize_explain.add reason=topic_and_time")

        if topic:
            texts.append(f"{topic}について整理すると、そのように捉えられます。")
            LOGGER.debug("surface_realizer.realize_explain.add reason=topic_only")

        return texts

    def _realize_confirm(self, filled_slots: FilledSlots) -> List[str]:
        topic = self._get_value(filled_slots, "topic")
        state = self._get_value(filled_slots, "state")
        predicate = self._get_value(filled_slots, "predicate")

        LOGGER.debug(
            "surface_realizer.realize_confirm inputs topic=%s state=%s predicate=%s",
            topic,
            state,
            predicate,
        )

        texts: List[str] = []

        if topic and state:
            texts.append(f"{topic}は{state}で合っています。")
            texts.append(f"{topic}は{state}で大丈夫です。")
            LOGGER.debug("surface_realizer.realize_confirm.add reason=topic_and_state")

        if topic and predicate:
            texts.append(f"{topic}については{predicate}で問題ありません。")
            LOGGER.debug("surface_realizer.realize_confirm.add reason=topic_and_predicate")

        if topic:
            texts.append(f"{topic}で認識しています。")
            LOGGER.debug("surface_realizer.realize_confirm.add reason=topic_only")

        texts.append("その理解で大丈夫です。")
        LOGGER.debug("surface_realizer.realize_confirm.add reason=generic_tail")
        return texts

    def _realize_empathy(self, filled_slots: FilledSlots) -> List[str]:
        state = self._get_value(filled_slots, "state")
        cause = self._get_value(filled_slots, "cause")
        actor = self._get_value(filled_slots, "actor")
        topic = self._get_value(filled_slots, "topic")

        LOGGER.debug(
            "surface_realizer.realize_empathy inputs state=%s cause=%s actor=%s topic=%s",
            state,
            cause,
            actor,
            topic,
        )

        texts: List[str] = []

        if state and cause:
            texts.append(f"{cause}で{state}のですね。")
            texts.append(f"{cause}があって{state}なんですね。")
            LOGGER.debug("surface_realizer.realize_empathy.add reason=state_and_cause")

        if actor and state:
            texts.append(f"{actor}が{state}のですね。")
            LOGGER.debug("surface_realizer.realize_empathy.add reason=actor_and_state")

        if topic and state:
            texts.append(f"{topic}のことで{state}のですね。")
            LOGGER.debug("surface_realizer.realize_empathy.add reason=topic_and_state")

        if state:
            texts.append(f"{state}のですね。")
            texts.append(f"{state}なんですね。")
            LOGGER.debug("surface_realizer.realize_empathy.add reason=state_only")

        texts.append("それは大変でしたね。")
        LOGGER.debug("surface_realizer.realize_empathy.add reason=generic_tail")
        return texts

    def _default_order_for_intent(self, intent: str) -> List[str]:
        if intent == "empathy":
            order = ["cause", "topic", "state", "predicate"]
        elif intent == "explain":
            order = ["topic", "predicate", "cause", "location", "time", "state"]
        elif intent == "confirm":
            order = ["topic", "state", "predicate"]
        elif intent == "question":
            order = ["topic", "state", "predicate", "target"]
        else:
            order = ["topic", "actor", "target", "predicate", "state"]

        LOGGER.debug(
            "surface_realizer.default_order intent=%s order=%s",
            intent,
            order,
        )
        return order

    def _template_id_for_intent(self, intent: str, filled_slots: FilledSlots) -> str:
        predicate_type = filled_slots.frame.predicate_type or "generic"
        template_id = f"{intent}_{predicate_type}"
        LOGGER.debug(
            "surface_realizer.template_id intent=%s predicate_type=%s template_id=%s",
            intent,
            predicate_type,
            template_id,
        )
        return template_id

    def _fallback_text(
        self,
        intent_plan: IntentPlan,
        filled_slots: FilledSlots,
    ) -> str:
        topic = self._get_value(filled_slots, "topic")
        if intent_plan.intent == "empathy":
            text = "つらそうですね。"
        elif intent_plan.intent == "confirm":
            text = "その理解で大丈夫です。"
        elif intent_plan.intent == "explain" and topic:
            text = f"{topic}についての話です。"
        elif topic:
            text = f"{topic}ですね。"
        else:
            text = "そうですね。"

        LOGGER.debug(
            "surface_realizer.fallback_text intent=%s topic=%s result=%s",
            intent_plan.intent,
            topic,
            text,
        )
        return text

    def _get_value(self, filled_slots: FilledSlots, slot_name: str) -> str:
        slot = filled_slots.values.get(slot_name)
        if slot is None:
            LOGGER.debug(
                "surface_realizer.get_value slot=%s result='' reason=missing",
                slot_name,
            )
            return ""
        value = str(slot.value).strip()
        LOGGER.debug(
            "surface_realizer.get_value slot=%s value=%s",
            slot_name,
            value,
        )
        return value

    def _slot_coverage(self, filled_slots: FilledSlots) -> float:
        if not filled_slots.frame.constraints:
            coverage = 1.0 if filled_slots.values else 0.0
            LOGGER.debug(
                "surface_realizer.slot_coverage no_constraints values_present=%s coverage=%.6f",
                bool(filled_slots.values),
                coverage,
            )
            return coverage

        filled = 0
        total = len(filled_slots.frame.constraints)

        for constraint in filled_slots.frame.constraints:
            if constraint.name in filled_slots.values:
                filled += 1
                LOGGER.debug(
                    "surface_realizer.slot_coverage.hit constraint=%s",
                    constraint.name,
                )
            else:
                LOGGER.debug(
                    "surface_realizer.slot_coverage.miss constraint=%s",
                    constraint.name,
                )

        coverage = max(0.0, min(1.0, filled / float(total)))
        LOGGER.debug(
            "surface_realizer.slot_coverage result filled=%s total=%s coverage=%.6f",
            filled,
            total,
            coverage,
        )
        return coverage

    def _estimate_semantic_score(
        self,
        text: str,
        filled_slots: FilledSlots,
        intent_plan: IntentPlan,
    ) -> float:
        score = 0.50
        LOGGER.debug(
            "surface_realizer.estimate_semantic_score.start text=%s base=%.6f intent=%s",
            text,
            score,
            intent_plan.intent,
        )

        for slot in filled_slots.values.values():
            if slot.value and slot.value in text:
                score += 0.08
                LOGGER.debug(
                    "surface_realizer.estimate_semantic_score.add reason=slot_value_match slot=%s value=%s add=0.08 running=%.6f",
                    slot.slot_name,
                    slot.value,
                    score,
                )

        if text.endswith("。"):
            score += 0.04
            LOGGER.debug(
                "surface_realizer.estimate_semantic_score.add reason=endswith_period add=0.04 running=%.6f",
                score,
            )

        if intent_plan.intent == "empathy" and ("ね" in text or "大変" in text):
            score += 0.06
            LOGGER.debug(
                "surface_realizer.estimate_semantic_score.add reason=empathy_tone add=0.06 running=%.6f",
                score,
            )

        if intent_plan.intent == "confirm" and ("大丈夫" in text or "合って" in text):
            score += 0.06
            LOGGER.debug(
                "surface_realizer.estimate_semantic_score.add reason=confirm_tone add=0.06 running=%.6f",
                score,
            )

        if intent_plan.intent == "explain" and ("説明" in text or "整理" in text or "関係" in text):
            score += 0.04
            LOGGER.debug(
                "surface_realizer.estimate_semantic_score.add reason=explain_tone add=0.04 running=%.6f",
                score,
            )

        final = max(0.0, min(1.0, score))
        LOGGER.debug(
            "surface_realizer.estimate_semantic_score.result text=%s score=%.6f",
            text,
            final,
        )
        return final

    def _quick_grammar_checks(self, text: str) -> List[str]:
        violations: List[str] = []
        LOGGER.debug("surface_realizer.quick_grammar_checks.start text=%s", text)

        if not text:
            violations.append("empty_text")
            LOGGER.debug("surface_realizer.quick_grammar_checks.add empty_text")
            return violations

        if "。。" in text:
            violations.append("double_period")
            LOGGER.debug("surface_realizer.quick_grammar_checks.add double_period")

        if "はは" in text or "がが" in text or "をを" in text:
            violations.append("duplicated_particle")
            LOGGER.debug("surface_realizer.quick_grammar_checks.add duplicated_particle")

        if text.endswith("ます。") and "が" not in text and "は" not in text and "を" not in text:
            violations.append("weak_structure")
            LOGGER.debug("surface_realizer.quick_grammar_checks.add weak_structure")

        LOGGER.debug(
            "surface_realizer.quick_grammar_checks.result text=%s violations=%s",
            text,
            violations,
        )
        return violations

    def _simple_tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        current = ""

        LOGGER.debug("surface_realizer.simple_tokenize.start text=%s", text)

        for ch in text:
            if ch in {"。", "、", "?", "？", "!", "！"}:
                if current:
                    tokens.append(current)
                    LOGGER.debug(
                        "surface_realizer.simple_tokenize.flush token=%s",
                        current,
                    )
                    current = ""
                tokens.append(ch)
                LOGGER.debug(
                    "surface_realizer.simple_tokenize.punct token=%s",
                    ch,
                )
                continue
            current += ch

        if current:
            tokens.append(current)
            LOGGER.debug(
                "surface_realizer.simple_tokenize.flush_last token=%s",
                current,
            )

        result = [token for token in tokens if token]
        LOGGER.debug("surface_realizer.simple_tokenize.result=%s", result)
        return result

    def _dedupe_keep_order(self, texts: Sequence[str]) -> List[str]:
        seen = set()
        result: List[str] = []

        LOGGER.debug("surface_realizer.dedupe_keep_order.start texts=%s", list(texts))

        for text in texts:
            normalized = self._normalize_text(text)
            LOGGER.debug(
                "surface_realizer.dedupe_keep_order.normalized original=%s normalized=%s",
                text,
                normalized,
            )
            if not normalized:
                LOGGER.debug(
                    "surface_realizer.dedupe_keep_order.skip reason=empty_after_normalize"
                )
                continue
            if normalized in seen:
                LOGGER.debug(
                    "surface_realizer.dedupe_keep_order.skip normalized=%s reason=duplicate",
                    normalized,
                )
                continue
            seen.add(normalized)
            result.append(normalized)
            LOGGER.debug(
                "surface_realizer.dedupe_keep_order.accept normalized=%s",
                normalized,
            )

        LOGGER.debug("surface_realizer.dedupe_keep_order.result=%s", result)
        return result

    def _normalize_text(self, text: str) -> str:
        text = str(text).strip()
        if not text:
            LOGGER.debug("surface_realizer.normalize_text input_empty -> ''")
            return ""
        if not text.endswith(("。", "？", "!", "！")):
            text += "。"
            LOGGER.debug(
                "surface_realizer.normalize_text.append_period result=%s",
                text,
            )
        else:
            LOGGER.debug("surface_realizer.normalize_text.keep=%s", text)
        return text


def realize_surface(
    filled_slots: FilledSlots,
    intent_plan: Optional[IntentPlan] = None,
    config: Optional[SurfaceRealizerConfig] = None,
) -> Tuple[SurfacePlan, List[RealizationCandidate]]:
    realizer = SurfaceRealizer(config=config)
    return realizer.realize(
        filled_slots=filled_slots,
        intent_plan=intent_plan,
    )