from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.core.schema import (
    DialogueState,
    FilledSlots,
    InputState,
    IntentPlan,
    LexiconContainer,
    LexiconEntry,
    RecallResult,
    SlotConstraint,
    SlotFrame,
    SlotValue,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SlotFillerConfig:
    marker_confidence: float = 0.88
    inferred_confidence: float = 0.58
    fallback_confidence: float = 0.46
    use_dialogue_topic_fallback: bool = True
    use_recall_topic_fallback: bool = True


class SlotFiller:
    """
    詳細 DEBUG ログ付きの Slot Filler。
    どのトークンをどう見て predicate / slot を埋めたかを追跡できる。
    """

    TIME_WORDS: Set[str] = {
        "今日",
        "明日",
        "昨日",
        "今",
        "さっき",
        "朝",
        "昼",
        "夜",
        "今朝",
        "今夜",
        "週末",
        "来週",
        "先週",
        "今年",
        "来年",
        "去年",
    }

    LOCATION_HINTS: Set[str] = {
        "場所",
        "家",
        "部屋",
        "学校",
        "会社",
        "駅",
        "公園",
        "東京",
        "日本",
        "ここ",
        "そこ",
        "あそこ",
    }

    PERSON_HINTS: Set[str] = {
        "私",
        "僕",
        "俺",
        "あなた",
        "君",
        "きみ",
        "なのはさん",
        "ユナ",
        "彼",
        "彼女",
        "みんな",
        "先生",
        "友達",
    }

    PREDICATE_POS_PRIORITY: Tuple[str, ...] = (
        "interjection",
        "verb_stem",
        "verb",
        "adjective_i",
        "adjective_na",
        "copula",
    )

    NOMINAL_POS: Set[str] = {
        "noun",
        "pronoun",
        "proper_noun",
        "nominal",
    }

    STATE_LIKE_POS: Set[str] = {
        "adjective_i",
        "adjective_na",
        "copula",
    }

    def __init__(self, config: Optional[SlotFillerConfig] = None) -> None:
        self.config = config or SlotFillerConfig()

    def fill(
        self,
        input_state: InputState,
        recall_result: RecallResult,
        lexicon: LexiconContainer,
        intent_plan: Optional[IntentPlan] = None,
        dialogue_state: Optional[DialogueState] = None,
    ) -> FilledSlots:
        intent_plan = intent_plan or IntentPlan(intent="unknown")
        dialogue_state = dialogue_state or DialogueState()

        tokens = self._collect_tokens(input_state)
        LOGGER.debug(
            "slot_filler.start raw_text=%s intent=%s tokens=%s seeds=%s recall_candidates=%s dialogue_state=%s config=%s",
            input_state.raw_text,
            intent_plan.intent,
            tokens,
            recall_result.seeds,
            [
                {
                    "word": c.word,
                    "score": round(c.score, 6),
                    "source": c.source,
                    "note": c.note,
                }
                for c in recall_result.candidates
            ],
            {
                "current_topic": dialogue_state.current_topic,
                "last_subject": dialogue_state.last_subject,
                "last_object": dialogue_state.last_object,
                "referents": dialogue_state.referents,
                "variables": dialogue_state.variables,
            },
            self.config,
        )

        predicate_word, predicate_entry = self._select_predicate(
            tokens=tokens,
            recall_result=recall_result,
            lexicon=lexicon,
        )
        LOGGER.debug(
            "slot_filler.predicate_selected predicate=%s predicate_entry=%s",
            predicate_word,
            {
                "pos": predicate_entry.grammar.pos,
                "category": predicate_entry.category,
                "hierarchy": predicate_entry.hierarchy,
                "slots": [slot.name for slot in predicate_entry.slots],
            }
            if predicate_entry
            else None,
        )

        frame = self._build_slot_frame(
            predicate_word=predicate_word,
            predicate_entry=predicate_entry,
            intent_plan=intent_plan,
        )
        LOGGER.debug(
            "slot_filler.frame_built predicate=%s predicate_type=%s constraints=%s",
            frame.predicate,
            frame.predicate_type,
            [
                {
                    "name": c.name,
                    "required": c.required,
                    "allowed_pos": c.allowed_pos,
                    "semantic_hint": c.semantic_hint,
                    "note": c.note,
                }
                for c in frame.constraints
            ],
        )

        values: Dict[str, SlotValue] = {}

        if predicate_word:
            self._set_slot_if_better(
                values=values,
                slot_name="predicate",
                value=predicate_word,
                confidence=0.92,
                source_candidate=predicate_word,
                inferred=False,
                note="predicate_selected",
            )

        self._apply_particle_patterns(
            tokens=tokens,
            lexicon=lexicon,
            values=values,
        )

        self._apply_predicate_state_rule(
            predicate_word=predicate_word,
            predicate_entry=predicate_entry,
            values=values,
        )

        self._apply_intent_specific_fallbacks(
            tokens=tokens,
            recall_result=recall_result,
            lexicon=lexicon,
            values=values,
            intent_plan=intent_plan,
        )

        self._apply_dialogue_fallbacks(
            values=values,
            dialogue_state=dialogue_state,
        )

        missing_required, optional_unfilled = self._compute_missing_slots(
            frame=frame,
            values=values,
        )
        consistency_score = self._compute_consistency_score(
            frame=frame,
            values=values,
            missing_required=missing_required,
        )

        result = FilledSlots(
            frame=frame,
            values=values,
            missing_required=missing_required,
            optional_unfilled=optional_unfilled,
            consistency_score=consistency_score,
        )

        LOGGER.debug(
            "slot_filler.result predicate=%s filled=%s missing_required=%s optional_unfilled=%s consistency_score=%.6f",
            frame.predicate,
            {
                key: {
                    "value": slot_value.value,
                    "confidence": slot_value.confidence,
                    "source_candidate": slot_value.source_candidate,
                    "inferred": slot_value.inferred,
                    "note": slot_value.note,
                }
                for key, slot_value in values.items()
            },
            missing_required,
            optional_unfilled,
            consistency_score,
        )
        return result

    def _collect_tokens(self, input_state: InputState) -> List[str]:
        tokens = [str(t).strip() for t in input_state.normalized_tokens if str(t).strip()]
        if tokens:
            LOGGER.debug(
                "slot_filler.collect_tokens source=normalized tokens=%s",
                tokens,
            )
            return tokens

        fallback_tokens = [str(t).strip() for t in input_state.tokens if str(t).strip()]
        LOGGER.debug(
            "slot_filler.collect_tokens source=raw tokens=%s",
            fallback_tokens,
        )
        return fallback_tokens

    def _select_predicate(
        self,
        tokens: Sequence[str],
        recall_result: RecallResult,
        lexicon: LexiconContainer,
    ) -> Tuple[str, Optional[LexiconEntry]]:
        LOGGER.debug(
            "slot_filler.select_predicate.begin tokens=%s",
            list(tokens),
        )

        for i in range(len(tokens) - 1):
            left = tokens[i]
            right = tokens[i + 1]
            left_entry = lexicon.entries.get(left)
            right_entry = lexicon.entries.get(right)

            LOGGER.debug(
                "slot_filler.select_predicate.nominal_copula_scan index=%s left=%s right=%s left_entry=%s right_entry=%s",
                i,
                left,
                right,
                left_entry.grammar.pos if left_entry else None,
                right_entry.grammar.pos if right_entry else None,
            )

            if left_entry is None or right_entry is None:
                continue

            if right_entry.grammar.pos == "copula" and self._is_nominal(left_entry):
                LOGGER.debug(
                    "slot_filler.select_predicate.hit mode=nominal_plus_copula predicate=%s right=%s",
                    left,
                    right,
                )
                return left, left_entry

        for preferred_pos in self.PREDICATE_POS_PRIORITY:
            LOGGER.debug(
                "slot_filler.select_predicate.scan_preferred_pos pos=%s",
                preferred_pos,
            )
            for token in reversed(tokens):
                entry = lexicon.entries.get(token)
                LOGGER.debug(
                    "slot_filler.select_predicate.scan_token token=%s pos=%s",
                    token,
                    entry.grammar.pos if entry else None,
                )
                if entry is None:
                    continue
                if entry.grammar.pos == preferred_pos:
                    if preferred_pos == "copula":
                        LOGGER.debug(
                            "slot_filler.select_predicate.skip token=%s reason=bare_copula_not_selected",
                            token,
                        )
                        continue
                    LOGGER.debug(
                        "slot_filler.select_predicate.hit mode=input_token predicate=%s pos=%s",
                        token,
                        entry.grammar.pos,
                    )
                    return token, entry

        LOGGER.debug("slot_filler.select_predicate.fallback_to_recall")
        for candidate in recall_result.candidates:
            entry = lexicon.entries.get(candidate.word)
            LOGGER.debug(
                "slot_filler.select_predicate.recall_candidate word=%s score=%.6f source=%s pos=%s note=%s",
                candidate.word,
                candidate.score,
                candidate.source,
                entry.grammar.pos if entry else None,
                candidate.note,
            )
            if entry is None:
                continue
            if entry.grammar.pos in {"verb", "verb_stem", "adjective_i", "adjective_na"}:
                LOGGER.debug(
                    "slot_filler.select_predicate.hit mode=recall_candidate predicate=%s pos=%s",
                    candidate.word,
                    entry.grammar.pos,
                )
                return candidate.word, entry

        LOGGER.debug("slot_filler.select_predicate.result none")
        return "", None

    def _build_slot_frame(
        self,
        predicate_word: str,
        predicate_entry: Optional[LexiconEntry],
        intent_plan: IntentPlan,
    ) -> SlotFrame:
        constraints: List[SlotConstraint] = []
        seen: Set[str] = set()

        LOGGER.debug(
            "slot_filler.build_slot_frame.begin predicate=%s intent=%s",
            predicate_word,
            intent_plan.intent,
        )

        if predicate_entry is not None:
            LOGGER.debug(
                "slot_filler.build_slot_frame.from_predicate_entry slots=%s",
                [
                    {
                        "name": slot.name,
                        "required": slot.required,
                        "allowed_pos": slot.allowed_pos,
                        "semantic_hint": slot.semantic_hint,
                        "note": slot.note,
                    }
                    for slot in predicate_entry.slots
                ],
            )
            for constraint in predicate_entry.slots:
                if constraint.name and constraint.name not in seen:
                    constraints.append(constraint)
                    seen.add(constraint.name)
                    LOGGER.debug(
                        "slot_filler.build_slot_frame.add_from_predicate name=%s required=%s",
                        constraint.name,
                        constraint.required,
                    )

        for slot_name in intent_plan.required_slots:
            if slot_name not in seen:
                constraints.append(
                    SlotConstraint(
                        name=slot_name,
                        required=True,
                        note="from_intent_required",
                    )
                )
                seen.add(slot_name)
                LOGGER.debug(
                    "slot_filler.build_slot_frame.add_from_intent_required name=%s",
                    slot_name,
                )

        for slot_name in intent_plan.optional_slots:
            if slot_name not in seen:
                constraints.append(
                    SlotConstraint(
                        name=slot_name,
                        required=False,
                        note="from_intent_optional",
                    )
                )
                seen.add(slot_name)
                LOGGER.debug(
                    "slot_filler.build_slot_frame.add_from_intent_optional name=%s",
                    slot_name,
                )

        if not constraints:
            defaults = self._default_slots_for_intent(intent_plan.intent)
            LOGGER.debug(
                "slot_filler.build_slot_frame.use_defaults intent=%s defaults=%s",
                intent_plan.intent,
                defaults,
            )
            for slot_name, required in defaults:
                if slot_name not in seen:
                    constraints.append(
                        SlotConstraint(
                            name=slot_name,
                            required=required,
                            note="default_by_intent",
                        )
                    )
                    seen.add(slot_name)
                    LOGGER.debug(
                        "slot_filler.build_slot_frame.add_default name=%s required=%s",
                        slot_name,
                        required,
                    )

        frame = SlotFrame(
            predicate=predicate_word,
            predicate_type=self._infer_predicate_type(predicate_entry),
            constraints=constraints,
        )
        LOGGER.debug("slot_filler.build_slot_frame.result=%s", frame)
        return frame

    def _default_slots_for_intent(self, intent: str) -> List[Tuple[str, bool]]:
        if intent == "question":
            return [("topic", False)]
        if intent == "explain":
            return [("topic", True)]
        if intent == "confirm":
            return [("topic", False), ("state", False)]
        if intent == "empathy":
            return [("state", False), ("cause", False)]
        return [("topic", False)]

    def _infer_predicate_type(self, predicate_entry: Optional[LexiconEntry]) -> str:
        if predicate_entry is None:
            LOGGER.debug("slot_filler.infer_predicate_type predicate_entry=None -> ''")
            return ""
        pos = predicate_entry.grammar.pos
        if pos in {"verb", "verb_stem"}:
            result = "action"
        elif pos in {"adjective_i", "adjective_na", "copula"}:
            result = "state"
        elif self._is_nominal(predicate_entry):
            result = "entity_state"
        else:
            result = pos

        LOGGER.debug(
            "slot_filler.infer_predicate_type word=%s pos=%s result=%s",
            predicate_entry.word,
            pos,
            result,
        )
        return result

    def _apply_particle_patterns(
        self,
        tokens: Sequence[str],
        lexicon: LexiconContainer,
        values: Dict[str, SlotValue],
    ) -> None:
        LOGGER.debug("slot_filler.apply_particle_patterns.begin tokens=%s", list(tokens))

        for index, token in enumerate(tokens):
            prev_token, prev_entry = self._find_previous_content_token(tokens, index, lexicon)

            LOGGER.debug(
                "slot_filler.apply_particle_patterns.scan index=%s token=%s prev_token=%s prev_pos=%s",
                index,
                token,
                prev_token,
                prev_entry.grammar.pos if prev_entry else None,
            )

            if not prev_token or prev_entry is None:
                LOGGER.debug(
                    "slot_filler.apply_particle_patterns.skip index=%s token=%s reason=no_previous_content_token",
                    index,
                    token,
                )
                continue

            if token == "は":
                LOGGER.debug(
                    "slot_filler.apply_particle_patterns.hit particle=は prev_token=%s",
                    prev_token,
                )
                self._set_slot_if_better(
                    values=values,
                    slot_name="topic",
                    value=prev_token,
                    confidence=self.config.marker_confidence,
                    source_candidate=prev_token,
                    inferred=False,
                    note="topic_from_wa",
                )
                if "actor" not in values and self._is_nominal(prev_entry):
                    self._set_slot_if_better(
                        values=values,
                        slot_name="actor",
                        value=prev_token,
                        confidence=0.68,
                        source_candidate=prev_token,
                        inferred=True,
                        note="actor_proxy_from_wa",
                    )

            elif token == "が":
                LOGGER.debug(
                    "slot_filler.apply_particle_patterns.hit particle=が prev_token=%s",
                    prev_token,
                )
                self._set_slot_if_better(
                    values=values,
                    slot_name="actor",
                    value=prev_token,
                    confidence=self.config.marker_confidence,
                    source_candidate=prev_token,
                    inferred=False,
                    note="actor_from_ga",
                )

            elif token == "を":
                LOGGER.debug(
                    "slot_filler.apply_particle_patterns.hit particle=を prev_token=%s",
                    prev_token,
                )
                self._set_slot_if_better(
                    values=values,
                    slot_name="target",
                    value=prev_token,
                    confidence=self.config.marker_confidence,
                    source_candidate=prev_token,
                    inferred=False,
                    note="target_from_wo",
                )

            elif token == "に":
                role = self._classify_ni_role(prev_token, prev_entry)
                LOGGER.debug(
                    "slot_filler.apply_particle_patterns.hit particle=に prev_token=%s classified_role=%s",
                    prev_token,
                    role,
                )
                self._set_slot_if_better(
                    values=values,
                    slot_name=role,
                    value=prev_token,
                    confidence=self.config.marker_confidence,
                    source_candidate=prev_token,
                    inferred=False,
                    note=f"{role}_from_ni",
                )

            elif token == "へ":
                LOGGER.debug(
                    "slot_filler.apply_particle_patterns.hit particle=へ prev_token=%s",
                    prev_token,
                )
                self._set_slot_if_better(
                    values=values,
                    slot_name="location",
                    value=prev_token,
                    confidence=self.config.marker_confidence,
                    source_candidate=prev_token,
                    inferred=False,
                    note="location_from_he",
                )

            elif token == "で":
                role = "location" if self._looks_like_location(prev_token, prev_entry) else "manner"
                LOGGER.debug(
                    "slot_filler.apply_particle_patterns.hit particle=で prev_token=%s classified_role=%s",
                    prev_token,
                    role,
                )
                self._set_slot_if_better(
                    values=values,
                    slot_name=role,
                    value=prev_token,
                    confidence=self.config.marker_confidence,
                    source_candidate=prev_token,
                    inferred=False,
                    note=f"{role}_from_de",
                )

            elif token == "から":
                role = "cause" if not self._looks_like_location(prev_token, prev_entry) else "location"
                LOGGER.debug(
                    "slot_filler.apply_particle_patterns.hit particle=から prev_token=%s classified_role=%s",
                    prev_token,
                    role,
                )
                self._set_slot_if_better(
                    values=values,
                    slot_name=role,
                    value=prev_token,
                    confidence=0.80,
                    source_candidate=prev_token,
                    inferred=False,
                    note=f"{role}_from_kara",
                )
            else:
                LOGGER.debug(
                    "slot_filler.apply_particle_patterns.skip token=%s reason=not_target_particle",
                    token,
                )

    def _apply_predicate_state_rule(
        self,
        predicate_word: str,
        predicate_entry: Optional[LexiconEntry],
        values: Dict[str, SlotValue],
    ) -> None:
        LOGGER.debug(
            "slot_filler.apply_predicate_state_rule predicate=%s predicate_pos=%s",
            predicate_word,
            predicate_entry.grammar.pos if predicate_entry else None,
        )

        if not predicate_word or predicate_entry is None:
            LOGGER.debug(
                "slot_filler.apply_predicate_state_rule.skip reason=no_predicate"
            )
            return

        if predicate_entry.grammar.pos in self.STATE_LIKE_POS or self._is_nominal(predicate_entry):
            LOGGER.debug(
                "slot_filler.apply_predicate_state_rule.apply state=%s",
                predicate_word,
            )
            self._set_slot_if_better(
                values=values,
                slot_name="state",
                value=predicate_word,
                confidence=0.72,
                source_candidate=predicate_word,
                inferred=True,
                note="state_from_predicate",
            )
        else:
            LOGGER.debug(
                "slot_filler.apply_predicate_state_rule.skip reason=predicate_not_state_like"
            )

    def _apply_intent_specific_fallbacks(
        self,
        tokens: Sequence[str],
        recall_result: RecallResult,
        lexicon: LexiconContainer,
        values: Dict[str, SlotValue],
        intent_plan: IntentPlan,
    ) -> None:
        LOGGER.debug(
            "slot_filler.apply_intent_specific_fallbacks.begin intent=%s current_values=%s",
            intent_plan.intent,
            {k: v.value for k, v in values.items()},
        )

        if intent_plan.intent in {"question", "explain", "confirm", "respond"} and "topic" not in values:
            topic = self._find_best_nominal_topic(tokens=tokens, lexicon=lexicon, exclude_values=values)
            LOGGER.debug("slot_filler.topic_fallback.from_tokens result=%s", topic)

            if not topic and self.config.use_recall_topic_fallback:
                topic = self._find_topic_from_recall(
                    recall_result=recall_result,
                    lexicon=lexicon,
                    exclude_values=values,
                )
                LOGGER.debug("slot_filler.topic_fallback.from_recall result=%s", topic)

            if topic:
                self._set_slot_if_better(
                    values=values,
                    slot_name="topic",
                    value=topic,
                    confidence=self.config.inferred_confidence,
                    source_candidate=topic,
                    inferred=True,
                    note="topic_fallback",
                )

        if intent_plan.intent == "empathy" and "state" not in values:
            state_word = self._find_state_like_word(tokens=tokens, lexicon=lexicon)
            LOGGER.debug("slot_filler.state_fallback.from_tokens result=%s", state_word)
            if not state_word:
                state_word = self._find_state_from_recall(recall_result=recall_result, lexicon=lexicon)
                LOGGER.debug("slot_filler.state_fallback.from_recall result=%s", state_word)
            if state_word:
                self._set_slot_if_better(
                    values=values,
                    slot_name="state",
                    value=state_word,
                    confidence=self.config.inferred_confidence,
                    source_candidate=state_word,
                    inferred=True,
                    note="state_fallback_for_empathy",
                )

    def _apply_dialogue_fallbacks(
        self,
        values: Dict[str, SlotValue],
        dialogue_state: DialogueState,
    ) -> None:
        LOGGER.debug(
            "slot_filler.apply_dialogue_fallbacks begin current_values=%s dialogue_topic=%s",
            {k: v.value for k, v in values.items()},
            dialogue_state.current_topic,
        )

        if (
            self.config.use_dialogue_topic_fallback
            and "topic" not in values
            and dialogue_state.current_topic
        ):
            LOGGER.debug(
                "slot_filler.apply_dialogue_fallbacks.apply topic=%s",
                dialogue_state.current_topic,
            )
            self._set_slot_if_better(
                values=values,
                slot_name="topic",
                value=dialogue_state.current_topic,
                confidence=self.config.fallback_confidence,
                source_candidate=dialogue_state.current_topic,
                inferred=True,
                note="topic_from_dialogue_state",
            )
        else:
            LOGGER.debug("slot_filler.apply_dialogue_fallbacks.skip")

    def _compute_missing_slots(
        self,
        frame: SlotFrame,
        values: Dict[str, SlotValue],
    ) -> Tuple[List[str], List[str]]:
        missing_required: List[str] = []
        optional_unfilled: List[str] = []

        LOGGER.debug(
            "slot_filler.compute_missing_slots begin constraints=%s current_values=%s",
            [c.name for c in frame.constraints],
            list(values.keys()),
        )

        for constraint in frame.constraints:
            if not constraint.name:
                LOGGER.debug("slot_filler.compute_missing_slots.skip reason=empty_constraint_name")
                continue
            if constraint.name in values:
                LOGGER.debug(
                    "slot_filler.compute_missing_slots.hit name=%s",
                    constraint.name,
                )
                continue
            if constraint.required:
                missing_required.append(constraint.name)
                LOGGER.debug(
                    "slot_filler.compute_missing_slots.missing_required name=%s",
                    constraint.name,
                )
            else:
                optional_unfilled.append(constraint.name)
                LOGGER.debug(
                    "slot_filler.compute_missing_slots.optional_unfilled name=%s",
                    constraint.name,
                )

        LOGGER.debug(
            "slot_filler.compute_missing_slots.result missing_required=%s optional_unfilled=%s",
            missing_required,
            optional_unfilled,
        )
        return missing_required, optional_unfilled

    def _compute_consistency_score(
        self,
        frame: SlotFrame,
        values: Dict[str, SlotValue],
        missing_required: Sequence[str],
    ) -> float:
        constraint_count = len(frame.constraints)
        filled_constraint_count = sum(1 for c in frame.constraints if c.name in values)

        base = 0.55
        LOGGER.debug(
            "slot_filler.compute_consistency_score.start base=%.4f constraint_count=%s filled_constraint_count=%s missing_required=%s predicate=%s",
            base,
            constraint_count,
            filled_constraint_count,
            list(missing_required),
            frame.predicate,
        )

        if frame.predicate:
            base += 0.15
            LOGGER.debug(
                "slot_filler.compute_consistency_score.add reason=has_predicate add=0.15 running=%.4f",
                base,
            )
        if "predicate" in values:
            base += 0.05
            LOGGER.debug(
                "slot_filler.compute_consistency_score.add reason=predicate_slot_present add=0.05 running=%.4f",
                base,
            )

        if constraint_count > 0:
            coverage = filled_constraint_count / float(constraint_count)
            base += coverage * 0.20
            LOGGER.debug(
                "slot_filler.compute_consistency_score.add reason=constraint_coverage coverage=%.6f add=%.6f running=%.4f",
                coverage,
                coverage * 0.20,
                base,
            )
        else:
            if values:
                base += 0.10
                LOGGER.debug(
                    "slot_filler.compute_consistency_score.add reason=no_constraints_but_has_values add=0.10 running=%.4f",
                    base,
                )

        if missing_required:
            penalty = min(0.25, 0.08 * len(missing_required))
            base -= penalty
            LOGGER.debug(
                "slot_filler.compute_consistency_score.sub reason=missing_required penalty=%.6f running=%.4f",
                penalty,
                base,
            )

        score = max(0.0, min(1.0, base))
        LOGGER.debug(
            "slot_filler.compute_consistency_score.result=%.6f",
            score,
        )
        return score

    def _find_previous_content_token(
        self,
        tokens: Sequence[str],
        marker_index: int,
        lexicon: LexiconContainer,
    ) -> Tuple[str, Optional[LexiconEntry]]:
        LOGGER.debug(
            "slot_filler.find_previous_content_token begin marker_index=%s token=%s",
            marker_index,
            tokens[marker_index] if 0 <= marker_index < len(tokens) else None,
        )

        for j in range(marker_index - 1, -1, -1):
            word = tokens[j]
            entry = lexicon.entries.get(word)
            LOGGER.debug(
                "slot_filler.find_previous_content_token.scan index=%s word=%s pos=%s content_word=%s function_word=%s",
                j,
                word,
                entry.grammar.pos if entry else None,
                entry.grammar.content_word if entry else None,
                entry.grammar.function_word if entry else None,
            )
            if entry is None:
                continue
            if entry.grammar.function_word and not entry.grammar.content_word:
                LOGGER.debug(
                    "slot_filler.find_previous_content_token.skip word=%s reason=function_word_only",
                    word,
                )
                continue
            LOGGER.debug(
                "slot_filler.find_previous_content_token.hit word=%s pos=%s",
                word,
                entry.grammar.pos,
            )
            return word, entry

        LOGGER.debug("slot_filler.find_previous_content_token.result none")
        return "", None

    def _classify_ni_role(self, word: str, entry: LexiconEntry) -> str:
        LOGGER.debug(
            "slot_filler.classify_ni_role word=%s pos=%s hierarchy=%s category=%s",
            word,
            entry.grammar.pos,
            entry.hierarchy,
            entry.category,
        )

        if self._looks_like_time(word, entry):
            LOGGER.debug("slot_filler.classify_ni_role.result=time")
            return "time"
        if self._looks_like_person(word, entry):
            LOGGER.debug("slot_filler.classify_ni_role.result=recipient")
            return "recipient"
        if self._looks_like_location(word, entry):
            LOGGER.debug("slot_filler.classify_ni_role.result=location")
            return "location"

        LOGGER.debug("slot_filler.classify_ni_role.result=target")
        return "target"

    def _find_best_nominal_topic(
        self,
        tokens: Sequence[str],
        lexicon: LexiconContainer,
        exclude_values: Dict[str, SlotValue],
    ) -> str:
        used = {v.value for v in exclude_values.values()}
        LOGGER.debug(
            "slot_filler.find_best_nominal_topic begin tokens=%s used=%s",
            list(tokens),
            sorted(used),
        )

        for token in tokens:
            if token in used:
                LOGGER.debug(
                    "slot_filler.find_best_nominal_topic.skip token=%s reason=already_used",
                    token,
                )
                continue
            entry = lexicon.entries.get(token)
            LOGGER.debug(
                "slot_filler.find_best_nominal_topic.scan token=%s pos=%s",
                token,
                entry.grammar.pos if entry else None,
            )
            if entry is None:
                continue
            if self._is_nominal(entry):
                LOGGER.debug(
                    "slot_filler.find_best_nominal_topic.hit token=%s",
                    token,
                )
                return token

        LOGGER.debug("slot_filler.find_best_nominal_topic.result none")
        return ""

    def _find_topic_from_recall(
        self,
        recall_result: RecallResult,
        lexicon: LexiconContainer,
        exclude_values: Dict[str, SlotValue],
    ) -> str:
        used = {v.value for v in exclude_values.values()}
        LOGGER.debug(
            "slot_filler.find_topic_from_recall begin used=%s",
            sorted(used),
        )

        for candidate in recall_result.candidates:
            if candidate.word in used:
                LOGGER.debug(
                    "slot_filler.find_topic_from_recall.skip word=%s reason=already_used",
                    candidate.word,
                )
                continue
            entry = lexicon.entries.get(candidate.word)
            LOGGER.debug(
                "slot_filler.find_topic_from_recall.scan word=%s score=%.6f source=%s pos=%s",
                candidate.word,
                candidate.score,
                candidate.source,
                entry.grammar.pos if entry else None,
            )
            if entry is None:
                continue
            if self._is_nominal(entry):
                LOGGER.debug(
                    "slot_filler.find_topic_from_recall.hit word=%s",
                    candidate.word,
                )
                return candidate.word

        LOGGER.debug("slot_filler.find_topic_from_recall.result none")
        return ""

    def _find_state_like_word(
        self,
        tokens: Sequence[str],
        lexicon: LexiconContainer,
    ) -> str:
        LOGGER.debug("slot_filler.find_state_like_word begin tokens=%s", list(tokens))
        for token in reversed(tokens):
            entry = lexicon.entries.get(token)
            LOGGER.debug(
                "slot_filler.find_state_like_word.scan token=%s pos=%s",
                token,
                entry.grammar.pos if entry else None,
            )
            if entry is None:
                continue
            if entry.grammar.pos in {"adjective_i", "adjective_na"}:
                LOGGER.debug(
                    "slot_filler.find_state_like_word.hit token=%s",
                    token,
                )
                return token
        LOGGER.debug("slot_filler.find_state_like_word.result none")
        return ""

    def _find_state_from_recall(
        self,
        recall_result: RecallResult,
        lexicon: LexiconContainer,
    ) -> str:
        LOGGER.debug("slot_filler.find_state_from_recall begin")
        for candidate in recall_result.candidates:
            entry = lexicon.entries.get(candidate.word)
            LOGGER.debug(
                "slot_filler.find_state_from_recall.scan word=%s score=%.6f source=%s pos=%s",
                candidate.word,
                candidate.score,
                candidate.source,
                entry.grammar.pos if entry else None,
            )
            if entry is None:
                continue
            if entry.grammar.pos in {"adjective_i", "adjective_na", "interjection"}:
                LOGGER.debug(
                    "slot_filler.find_state_from_recall.hit word=%s",
                    candidate.word,
                )
                return candidate.word
        LOGGER.debug("slot_filler.find_state_from_recall.result none")
        return ""

    def _looks_like_time(self, word: str, entry: LexiconEntry) -> bool:
        joined_hierarchy = "/".join(entry.hierarchy)
        category = entry.category.lower()
        result = (
            word in self.TIME_WORDS
            or "time" in joined_hierarchy.lower()
            or "temporal" in joined_hierarchy.lower()
            or "time" in category
        )
        LOGGER.debug(
            "slot_filler.looks_like_time word=%s pos=%s hierarchy=%s category=%s result=%s",
            word,
            entry.grammar.pos,
            entry.hierarchy,
            entry.category,
            result,
        )
        return result

    def _looks_like_location(self, word: str, entry: LexiconEntry) -> bool:
        joined_hierarchy = "/".join(entry.hierarchy).lower()
        category = entry.category.lower()
        meta_text = " ".join(str(v) for v in entry.meta.values()).lower()
        result = (
            word in self.LOCATION_HINTS
            or "location" in joined_hierarchy
            or "place" in joined_hierarchy
            or "location" in category
            or "place" in category
            or "location" in meta_text
            or "place" in meta_text
        )
        LOGGER.debug(
            "slot_filler.looks_like_location word=%s pos=%s hierarchy=%s category=%s meta=%s result=%s",
            word,
            entry.grammar.pos,
            entry.hierarchy,
            entry.category,
            entry.meta,
            result,
        )
        return result

    def _looks_like_person(self, word: str, entry: LexiconEntry) -> bool:
        joined_hierarchy = "/".join(entry.hierarchy).lower()
        category = entry.category.lower()
        result = (
            word in self.PERSON_HINTS
            or entry.grammar.pos == "pronoun"
            or "person" in joined_hierarchy
            or "human" in joined_hierarchy
            or "person" in category
            or "human" in category
        )
        LOGGER.debug(
            "slot_filler.looks_like_person word=%s pos=%s hierarchy=%s category=%s result=%s",
            word,
            entry.grammar.pos,
            entry.hierarchy,
            entry.category,
            result,
        )
        return result

    def _is_nominal(self, entry: LexiconEntry) -> bool:
        result = entry.grammar.pos in self.NOMINAL_POS
        LOGGER.debug(
            "slot_filler.is_nominal word=%s pos=%s result=%s",
            entry.word,
            entry.grammar.pos,
            result,
        )
        return result

    def _set_slot_if_better(
        self,
        values: Dict[str, SlotValue],
        slot_name: str,
        value: str,
        confidence: float,
        source_candidate: str,
        inferred: bool,
        note: str,
    ) -> None:
        if not slot_name or not value:
            LOGGER.debug(
                "slot_filler.set_slot.skip slot_name=%s value=%s reason=empty_slot_or_value",
                slot_name,
                value,
            )
            return

        current = values.get(slot_name)
        if current is None:
            LOGGER.debug(
                "slot_filler.set_slot.insert slot=%s value=%s confidence=%.6f source_candidate=%s inferred=%s note=%s",
                slot_name,
                value,
                confidence,
                source_candidate,
                inferred,
                note,
            )
            values[slot_name] = SlotValue(
                slot_name=slot_name,
                value=value,
                confidence=confidence,
                source_candidate=source_candidate,
                inferred=inferred,
                note=note,
            )
            return

        if confidence > current.confidence:
            LOGGER.debug(
                "slot_filler.set_slot.replace slot=%s old_value=%s old_confidence=%.6f new_value=%s new_confidence=%.6f source_candidate=%s inferred=%s note=%s",
                slot_name,
                current.value,
                current.confidence,
                value,
                confidence,
                source_candidate,
                inferred,
                note,
            )
            values[slot_name] = SlotValue(
                slot_name=slot_name,
                value=value,
                confidence=confidence,
                source_candidate=source_candidate,
                inferred=inferred,
                note=note,
            )
            return

        LOGGER.debug(
            "slot_filler.set_slot.keep_existing slot=%s existing_value=%s existing_confidence=%.6f rejected_value=%s rejected_confidence=%.6f",
            slot_name,
            current.value,
            current.confidence,
            value,
            confidence,
        )


def fill_slots(
    input_state: InputState,
    recall_result: RecallResult,
    lexicon: LexiconContainer,
    intent_plan: Optional[IntentPlan] = None,
    dialogue_state: Optional[DialogueState] = None,
    config: Optional[SlotFillerConfig] = None,
) -> FilledSlots:
    filler = SlotFiller(config=config)
    return filler.fill(
        input_state=input_state,
        recall_result=recall_result,
        lexicon=lexicon,
        intent_plan=intent_plan,
        dialogue_state=dialogue_state,
    )