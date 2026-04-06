from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.core.schema import (
    FilledSlots,
    IntentPlan,
    LexiconContainer,
    LexiconEntry,
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
    def __init__(self, config: Optional[SurfaceRealizerConfig] = None) -> None:
        self.config = config or SurfaceRealizerConfig()

    def realize(
        self,
        filled_slots: FilledSlots,
        intent_plan: Optional[IntentPlan] = None,
        lexicon: Optional[LexiconContainer] = None,
    ) -> Tuple[SurfacePlan, List[RealizationCandidate]]:
        intent_plan = intent_plan or IntentPlan(intent="unknown")

        LOGGER.debug(
            "surface_realizer.start intent=%s predicate=%s values=%s missing_required=%s",
            intent_plan.intent,
            filled_slots.frame.predicate,
            {k: v.value for k, v in filled_slots.values.items()},
            filled_slots.missing_required,
        )

        plan = self._build_surface_plan(filled_slots=filled_slots, intent_plan=intent_plan)
        candidates = self._build_candidates(
            filled_slots=filled_slots,
            intent_plan=intent_plan,
            plan=plan,
            lexicon=lexicon,
        )

        if not candidates:
            fallback_text = self._fallback_text(intent_plan=intent_plan, filled_slots=filled_slots)
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
            "surface_realizer.result template_id=%s count=%s texts=%s",
            plan.template_id,
            len(trimmed),
            [c.text for c in trimmed],
        )
        return plan, trimmed

    def _build_surface_plan(
        self,
        filled_slots: FilledSlots,
        intent_plan: IntentPlan,
    ) -> SurfacePlan:
        style = self.config.default_style
        if intent_plan.intent == "empathy":
            style = "warm"
        elif intent_plan.intent == "explain":
            style = "informative"
        elif intent_plan.intent == "confirm":
            style = "checking"
        elif intent_plan.intent == "question":
            style = "answering"

        return SurfacePlan(
            template_id=f"{intent_plan.intent}_{filled_slots.frame.predicate_type or 'generic'}",
            style=style,
            politeness=self.config.default_politeness,
            sentence_count=1,
            order=self._default_order_for_intent(intent_plan.intent),
            auxiliaries=[],
            note=f"surface_plan_for:{intent_plan.intent}",
        )

    def _build_candidates(
        self,
        filled_slots: FilledSlots,
        intent_plan: IntentPlan,
        plan: SurfacePlan,
        lexicon: Optional[LexiconContainer],
    ) -> List[RealizationCandidate]:
        if intent_plan.intent == "empathy":
            texts = self._realize_empathy(filled_slots, lexicon)
        elif intent_plan.intent == "question":
            texts = self._realize_question_answer(filled_slots, lexicon)
        elif intent_plan.intent == "explain":
            texts = self._realize_explain(filled_slots, lexicon)
        elif intent_plan.intent == "confirm":
            texts = self._realize_confirm(filled_slots, lexicon)
        else:
            texts = self._realize_respond(filled_slots, lexicon)

        unique_texts = self._dedupe_keep_order(texts)
        slot_coverage = self._slot_coverage(filled_slots)

        candidates: List[RealizationCandidate] = []
        for index, text in enumerate(unique_texts):
            if not text:
                continue
            candidates.append(
                RealizationCandidate(
                    text=text,
                    token_sequence=self._simple_tokenize(text),
                    template_id=f"{plan.template_id}_v{index + 1}",
                    grammar_violations=self._quick_grammar_checks(text),
                    slot_coverage=slot_coverage,
                    semantic_score=self._estimate_semantic_score(
                        text=text,
                        filled_slots=filled_slots,
                        intent_plan=intent_plan,
                    ),
                    final_score=0.0,
                )
            )
        return candidates

    def _realize_respond(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        predicate_raw = self._get_value(filled_slots, "predicate")
        predicate_entry = lexicon.entries.get(predicate_raw) if lexicon and predicate_raw else None
        if predicate_entry is not None and predicate_entry.grammar.pos == "interjection":
            return [f"{predicate_entry.get_surface('plain')}。"]

        topic = self._slot_surface(filled_slots, "topic", lexicon)
        state = self._slot_surface(filled_slots, "state", lexicon)
        actor = self._slot_surface(filled_slots, "actor", lexicon)
        predicate_plain = self._slot_surface(filled_slots, "predicate", lexicon, preferred_form="plain")
        predicate_polite = self._slot_surface(filled_slots, "predicate", lexicon, preferred_form="polite")
        target = self._slot_surface(filled_slots, "target", lexicon)
        location = self._slot_surface(filled_slots, "location", lexicon)
        time = self._slot_surface(filled_slots, "time", lexicon)

        texts: List[str] = []

        if topic and state:
            texts.append(f"{topic}は{state}です。")
            texts.append(f"{topic}は{state}ですね。")

        if actor and target and predicate_polite:
            texts.append(f"{actor}が{target}を{predicate_polite}。")

        if actor and predicate_polite and location:
            texts.append(f"{actor}が{location}で{predicate_polite}。")

        if time and actor and predicate_polite:
            texts.append(f"{time}に{actor}が{predicate_polite}。")

        if topic and predicate_plain:
            texts.append(f"{topic}については{predicate_plain}ことですね。")
            texts.append(f"{topic}について{predicate_polite}。")

        if topic:
            texts.append(f"{topic}ですね。")

        if state:
            texts.append(f"{state}です。")

        if predicate_polite:
            texts.append(f"{predicate_polite}。")

        return texts

    def _realize_question_answer(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, "topic", lexicon)
        state = self._slot_surface(filled_slots, "state", lexicon)
        predicate_plain = self._slot_surface(filled_slots, "predicate", lexicon, preferred_form="plain")
        predicate_polite = self._slot_surface(filled_slots, "predicate", lexicon, preferred_form="polite")
        actor = self._slot_surface(filled_slots, "actor", lexicon)
        target = self._slot_surface(filled_slots, "target", lexicon)

        texts: List[str] = []
        if topic and state:
            texts.append(f"{topic}は{state}です。")
            texts.append(f"{topic}は{state}だと思います。")
        if actor and target and predicate_polite:
            texts.append(f"{actor}が{target}を{predicate_polite}。")
        if topic and predicate_plain:
            texts.append(f"{topic}については{predicate_plain}ことが考えられます。")
            texts.append(f"{topic}について{predicate_polite}と考えられます。")
        if topic:
            texts.append(f"{topic}に関する話ですね。")
        texts.append("確認できる範囲ではそのように見えます。")
        return texts

    def _realize_explain(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, "topic", lexicon)
        predicate_plain = self._slot_surface(filled_slots, "predicate", lexicon, preferred_form="plain")
        state = self._slot_surface(filled_slots, "state", lexicon)
        cause = self._slot_surface(filled_slots, "cause", lexicon)
        location = self._slot_surface(filled_slots, "location", lexicon)
        time = self._slot_surface(filled_slots, "time", lexicon)

        texts: List[str] = []
        if topic and state:
            texts.append(f"{topic}は{state}という状態です。")
        if topic and predicate_plain:
            texts.append(f"{topic}は{predicate_plain}ことに関係します。")
            texts.append(f"{topic}については{predicate_plain}と説明できます。")
        if topic and cause:
            texts.append(f"{topic}は{cause}が理由として考えられます。")
        if topic and location:
            texts.append(f"{topic}は{location}に関係する話です。")
        if topic and time:
            texts.append(f"{topic}は{time}に関わる内容です。")
        if topic:
            texts.append(f"{topic}について整理すると、そのように捉えられます。")
        return texts

    def _realize_confirm(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        topic = self._slot_surface(filled_slots, "topic", lexicon)
        state = self._slot_surface(filled_slots, "state", lexicon)
        predicate_plain = self._slot_surface(filled_slots, "predicate", lexicon, preferred_form="plain")

        texts: List[str] = []
        if topic and state:
            texts.append(f"{topic}は{state}で合っています。")
            texts.append(f"{topic}は{state}で大丈夫です。")
        if topic and predicate_plain:
            texts.append(f"{topic}については{predicate_plain}ことで問題ありません。")
        if topic:
            texts.append(f"{topic}で認識しています。")
        texts.append("その理解で大丈夫です。")
        return texts

    def _realize_empathy(self, filled_slots: FilledSlots, lexicon: Optional[LexiconContainer]) -> List[str]:
        state = self._slot_surface(filled_slots, "state", lexicon)
        cause = self._slot_surface(filled_slots, "cause", lexicon)
        actor = self._slot_surface(filled_slots, "actor", lexicon)
        topic = self._slot_surface(filled_slots, "topic", lexicon)

        texts: List[str] = []
        if state and cause:
            texts.append(f"{cause}で{state}なんですね。")
            texts.append(f"{cause}があって{state}なんですね。")
        if actor and state:
            texts.append(f"{actor}が{state}なんですね。")
        if topic and state:
            texts.append(f"{topic}のことで{state}なんですね。")
        if state:
            texts.append(f"{state}んですね。" if state.endswith("い") else f"{state}なんですね。")
        texts.append("それは大変でしたね。")
        return texts

    def _slot_surface(
        self,
        filled_slots: FilledSlots,
        slot_name: str,
        lexicon: Optional[LexiconContainer],
        preferred_form: str = "plain",
    ) -> str:
        slot = filled_slots.values.get(slot_name)
        if slot is None:
            return ""
        word = str(slot.value).strip()
        if not word:
            return ""
        if lexicon is None:
            return word
        entry = lexicon.entries.get(word)
        if entry is None:
            return word
        return self._realize_entry(entry, preferred_form=preferred_form, slot_name=slot_name)

    def _realize_entry(self, entry: LexiconEntry, preferred_form: str, slot_name: str) -> str:
        if entry.grammar.pos == "verb_stem" or entry.entry_type == "stem":
            form = preferred_form
            if preferred_form == "plain" and slot_name in {"state", "topic"}:
                form = "plain"
            return entry.get_surface(form)

        if entry.grammar.pos == "adjective_i":
            return entry.get_surface("plain")

        if entry.grammar.pos == "adjective_na":
            return entry.word

        if entry.grammar.pos in {"copula", "auxiliary"}:
            return entry.get_surface(preferred_form)

        return entry.get_surface(preferred_form)

    def _default_order_for_intent(self, intent: str) -> List[str]:
        if intent == "empathy":
            return ["cause", "topic", "state", "predicate"]
        if intent == "explain":
            return ["topic", "predicate", "cause", "location", "time", "state"]
        if intent == "confirm":
            return ["topic", "state", "predicate"]
        if intent == "question":
            return ["topic", "state", "predicate", "target"]
        return ["topic", "actor", "target", "predicate", "state"]

    def _fallback_text(self, intent_plan: IntentPlan, filled_slots: FilledSlots) -> str:
        topic = self._get_value(filled_slots, "topic")
        if intent_plan.intent == "empathy":
            return "つらそうですね。"
        if intent_plan.intent == "confirm":
            return "その理解で大丈夫です。"
        if intent_plan.intent == "explain" and topic:
            return f"{topic}についての話です。"
        if topic:
            return f"{topic}ですね。"
        return "そうですね。"

    def _get_value(self, filled_slots: FilledSlots, slot_name: str) -> str:
        slot = filled_slots.values.get(slot_name)
        if slot is None:
            return ""
        return str(slot.value).strip()

    def _slot_coverage(self, filled_slots: FilledSlots) -> float:
        if not filled_slots.frame.constraints:
            return 1.0 if filled_slots.values else 0.0
        total = len(filled_slots.frame.constraints)
        filled = sum(1 for constraint in filled_slots.frame.constraints if constraint.name in filled_slots.values)
        return filled / float(total) if total > 0 else 0.0

    def _quick_grammar_checks(self, text: str) -> List[str]:
        violations: List[str] = []
        if "。。" in text:
            violations.append("double_period")
        if "です。です" in text:
            violations.append("duplicated_copula")
        if "。。" in text or "、、" in text:
            violations.append("duplicated_punctuation")
        return violations

    def _estimate_semantic_score(
        self,
        text: str,
        filled_slots: FilledSlots,
        intent_plan: IntentPlan,
    ) -> float:
        score = 0.40
        for slot in filled_slots.values.values():
            if slot.value and slot.value in text:
                score += 0.12
        if intent_plan.intent == "empathy" and "大変" in text:
            score += 0.08
        if intent_plan.intent == "confirm" and "大丈夫" in text:
            score += 0.08
        if intent_plan.intent == "question" and "考えられます" in text:
            score += 0.06
        return max(0.0, min(1.0, score))

    def _simple_tokenize(self, text: str) -> List[str]:
        return [chunk for chunk in text.replace("。", " 。").replace("、", " 、").split() if chunk]

    def _dedupe_keep_order(self, texts: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for text in texts:
            normalized = self._normalize_text(text)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    def _normalize_text(self, text: str) -> str:
        text = str(text).strip()
        if not text:
            return ""
        if not text.endswith(("。", "？", "!", "！")):
            text += "。"
        return text


def realize_surface(
    filled_slots: FilledSlots,
    intent_plan: Optional[IntentPlan] = None,
    lexicon: Optional[LexiconContainer] = None,
    config: Optional[SurfaceRealizerConfig] = None,
) -> Tuple[SurfacePlan, List[RealizationCandidate]]:
    realizer = SurfaceRealizer(config=config)
    return realizer.realize(
        filled_slots=filled_slots,
        intent_plan=intent_plan,
        lexicon=lexicon,
    )
