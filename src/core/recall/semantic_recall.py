from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.core.schema import (
    AxisVector,
    DialogueState,
    InputState,
    IntentPlan,
    LexiconContainer,
    LexiconEntry,
    RecallCandidate,
    RecallResult,
)

LOGGER = logging.getLogger(__name__)

DOMAIN_HINT_WORDS: Dict[str, Tuple[str, ...]] = {
    "weather": (
        "天気", "気温", "晴れ", "曇り", "雨", "雪", "風", "台風", "傘", "湿度", "暖かい", "涼しい", "暑い", "寒い", "空", "太陽",
    ),
    "time": (
        "今日", "明日", "今", "今夜", "今朝", "昨日", "時間", "予定", "朝", "昼", "夜",
    ),
    "news": (
        "ニュース", "情報", "最近", "今日", "話題", "速報",
    ),
    "availability": (
        "予定", "時間", "空き", "会議", "都合", "予約", "来週", "今週", "明日",
    ),
}


@dataclass(slots=True)
class SemanticRecallConfig:
    max_candidates: int = 24
    max_relation_hops: int = 1
    relation_weight_scale: float = 0.85
    axis_weight_scale: float = 0.75
    input_weight: float = 1.25
    function_seed_weight: float = 0.22
    content_seed_weight: float = 1.00
    prefer_content_words_in_axis: bool = True
    axis_probe_limit: int = 512
    question_axis_probe_limit: int = 96
    explain_axis_probe_limit: int = 128
    respond_axis_probe_limit: int = 80
    relation_limit_per_seed: int = 6
    enable_domain_hint_boost: bool = True
    domain_hint_bonus: float = 0.16
    domain_hint_probe_limit: int = 48


class SemanticRecallEngine:
    """
    かなり詳細な DEBUG ログ付きの Semantic Recall。
    どの seed が採用されたか、どの relation を辿ったか、
    axis 近傍で何が選ばれたかを全部追えるようにする。
    """

    def __init__(self, config: Optional[SemanticRecallConfig] = None) -> None:
        self.config = config or SemanticRecallConfig()

    def recall(
        self,
        input_state: InputState,
        lexicon: LexiconContainer,
        dialogue_state: Optional[DialogueState] = None,
        intent_plan: Optional[IntentPlan] = None,
    ) -> RecallResult:
        dialogue_state = dialogue_state or DialogueState()
        intent_plan = intent_plan or IntentPlan(intent="unknown")

        LOGGER.debug(
            "semantic_recall.start raw_text=%s normalized_tokens=%s raw_tokens=%s intent=%s dialogue_context=%s config=%s",
            input_state.raw_text,
            input_state.normalized_tokens,
            input_state.tokens,
            intent_plan.intent,
            {
                "current_topic": dialogue_state.current_topic,
                "last_subject": dialogue_state.last_subject,
                "last_object": dialogue_state.last_object,
                "context_vector": dialogue_state.context_vector,
            },
            self.config,
        )

        seeds = self._collect_seed_words(input_state=input_state, lexicon=lexicon)

        if not seeds:
            LOGGER.warning(
                "semantic_recall.no_seeds_found raw_text=%s tokens=%s",
                input_state.raw_text,
                input_state.normalized_tokens or input_state.tokens,
            )
            return RecallResult(seeds=[], candidates=[])

        candidates_by_word: Dict[str, RecallCandidate] = {}

        self._add_input_candidates(
            seeds=seeds,
            lexicon=lexicon,
            candidates_by_word=candidates_by_word,
        )

        domain_hints = self._detect_domain_hints(
            seeds=seeds,
            input_state=input_state,
            lexicon=lexicon,
            dialogue_state=dialogue_state,
        )
        LOGGER.debug("semantic_recall.domain_hints=%s", sorted(domain_hints))

        self._add_relation_candidates(
            seeds=seeds,
            lexicon=lexicon,
            candidates_by_word=candidates_by_word,
            domain_hints=domain_hints,
        )

        self._add_domain_hint_candidates(
            domain_hints=domain_hints,
            lexicon=lexicon,
            candidates_by_word=candidates_by_word,
        )

        self._add_axis_candidates(
            seeds=seeds,
            lexicon=lexicon,
            dialogue_state=dialogue_state,
            intent_plan=intent_plan,
            candidates_by_word=candidates_by_word,
            domain_hints=domain_hints,
        )

        ranked = sorted(
            candidates_by_word.values(),
            key=lambda item: (-item.score, item.axis_distance, item.word),
        )[: self.config.max_candidates]

        LOGGER.debug(
            "semantic_recall.ranked_candidates=%s",
            [
                {
                    "word": c.word,
                    "score": round(c.score, 6),
                    "source": c.source,
                    "axis_distance": round(c.axis_distance, 6),
                    "grammar_ok": c.grammar_ok,
                    "relation_path": c.relation_path,
                    "note": c.note,
                }
                for c in ranked
            ],
        )

        return RecallResult(
            seeds=seeds,
            candidates=ranked,
        )

    def _collect_seed_words(
        self,
        input_state: InputState,
        lexicon: LexiconContainer,
    ) -> List[str]:
        seen: Set[str] = set()
        ordered_content: List[str] = []
        ordered_function: List[str] = []
        raw_candidates = list(input_state.normalized_tokens) + list(input_state.tokens)

        LOGGER.debug(
            "semantic_recall.collect_seeds.begin raw_candidates=%s",
            raw_candidates,
        )

        for token in raw_candidates:
            token = str(token).strip()

            if not token:
                LOGGER.debug("semantic_recall.collect_seeds.skip reason=empty_token")
                continue

            if token in seen:
                LOGGER.debug(
                    "semantic_recall.collect_seeds.skip token=%s reason=already_seen",
                    token,
                )
                continue

            if token not in lexicon.entries:
                LOGGER.debug(
                    "semantic_recall.collect_seeds.skip token=%s reason=not_in_lexicon",
                    token,
                )
                continue

            entry = lexicon.entries[token]
            LOGGER.debug(
                "semantic_recall.collect_seeds.accept token=%s pos=%s content_word=%s function_word=%s hierarchy=%s",
                token,
                entry.grammar.pos,
                entry.grammar.content_word,
                entry.grammar.function_word,
                entry.hierarchy,
            )
            if entry.grammar.content_word:
                ordered_content.append(token)
            else:
                ordered_function.append(token)
            seen.add(token)

        ordered = ordered_content if ordered_content else ordered_function
        if ordered_content and len(ordered_content) < 3:
            ordered = ordered_content + ordered_function[: max(0, 3 - len(ordered_content))]

        LOGGER.debug("semantic_recall.collect_seeds.result=%s", ordered)
        return ordered

    def _add_input_candidates(
        self,
        seeds: Sequence[str],
        lexicon: LexiconContainer,
        candidates_by_word: Dict[str, RecallCandidate],
    ) -> None:
        LOGGER.debug("semantic_recall.add_input_candidates.begin seeds=%s", list(seeds))

        content_preferred = [seed for seed in seeds if (lexicon.entries.get(seed) and lexicon.entries.get(seed).grammar.content_word)]
        seed_sequence = content_preferred or list(seeds)

        for seed in seed_sequence:
            entry = lexicon.entries.get(seed)
            grammar_ok = self._is_grammar_ok(entry)

            seed_weight = self.config.content_seed_weight if (entry and entry.grammar.content_word) else self.config.function_seed_weight
            candidate = RecallCandidate(
                word=seed,
                score=self.config.input_weight * seed_weight,
                source="input",
                relation_path=[seed],
                axis_distance=0.0,
                grammar_ok=grammar_ok,
                note="seed_input_word",
            )

            LOGGER.debug(
                "semantic_recall.add_input_candidate word=%s score=%.4f grammar_ok=%s pos=%s",
                seed,
                candidate.score,
                grammar_ok,
                entry.grammar.pos if entry else None,
            )
            self._upsert_candidate(
                candidates_by_word=candidates_by_word,
                candidate=candidate,
            )

    def _add_relation_candidates(
        self,
        seeds: Sequence[str],
        lexicon: LexiconContainer,
        candidates_by_word: Dict[str, RecallCandidate],
        domain_hints: Set[str],
    ) -> None:
        if self.config.max_relation_hops <= 0:
            LOGGER.debug("semantic_recall.add_relation_candidates.skip reason=max_relation_hops<=0")
            return

        LOGGER.debug(
            "semantic_recall.add_relation_candidates.begin seeds=%s max_hops=%s",
            list(seeds),
            self.config.max_relation_hops,
        )

        for seed in seeds:
            seed_entry = lexicon.entries.get(seed)
            if seed_entry is None:
                LOGGER.debug(
                    "semantic_recall.add_relation_candidates.skip_seed seed=%s reason=no_entry",
                    seed,
                )
                continue

            LOGGER.debug(
                "semantic_recall.add_relation_candidates.seed seed=%s relation_count=%s relations=%s",
                seed,
                len(seed_entry.relations),
                [
                    {
                        "relation": edge.relation,
                        "target": edge.target,
                        "weight": edge.weight,
                        "bidirectional": edge.bidirectional,
                    }
                    for edge in seed_entry.relations
                ],
            )

            visited: Set[str] = {seed}
            frontier: List[Tuple[str, int, float, List[str]]] = [(seed, 0, 1.0, [seed])]

            while frontier:
                current_word, depth, cumulative_weight, path = frontier.pop(0)
                current_entry = lexicon.entries.get(current_word)

                LOGGER.debug(
                    "semantic_recall.relation_frontier_pop current_word=%s depth=%s cumulative_weight=%.6f path=%s frontier_remaining=%s",
                    current_word,
                    depth,
                    cumulative_weight,
                    path,
                    len(frontier),
                )

                if current_entry is None:
                    LOGGER.debug(
                        "semantic_recall.relation_skip current_word=%s reason=no_entry",
                        current_word,
                    )
                    continue

                if depth >= self.config.max_relation_hops:
                    LOGGER.debug(
                        "semantic_recall.relation_skip current_word=%s reason=depth_limit depth=%s",
                        current_word,
                        depth,
                    )
                    continue

                relation_edges = list(current_entry.relations)
                if self.config.relation_limit_per_seed > 0 and len(relation_edges) > self.config.relation_limit_per_seed:
                    relation_edges = relation_edges[: self.config.relation_limit_per_seed]
                    LOGGER.debug(
                        "semantic_recall.relation_truncate seed=%s current_word=%s kept=%s",
                        seed,
                        current_word,
                        len(relation_edges),
                    )

                for edge in relation_edges:
                    target = str(edge.target).strip()

                    LOGGER.debug(
                        "semantic_recall.relation_edge current_word=%s relation=%s target=%s weight=%s bidirectional=%s",
                        current_word,
                        edge.relation,
                        target,
                        edge.weight,
                        edge.bidirectional,
                    )

                    if not target:
                        LOGGER.debug(
                            "semantic_recall.relation_edge_skip reason=empty_target current_word=%s relation=%s",
                            current_word,
                            edge.relation,
                        )
                        continue

                    if target not in lexicon.entries:
                        LOGGER.debug(
                            "semantic_recall.relation_edge_skip reason=target_not_in_lexicon target=%s",
                            target,
                        )
                        continue

                    if target in visited:
                        LOGGER.debug(
                            "semantic_recall.relation_edge_skip reason=visited target=%s",
                            target,
                        )
                        continue

                    visited.add(target)

                    edge_weight = max(0.05, float(edge.weight))
                    domain_bonus = self._domain_bonus_for_word(target=target, domain_hints=domain_hints)
                    score = (
                        self.config.input_weight
                        * self.config.relation_weight_scale
                        * cumulative_weight
                        * edge_weight
                        / float(depth + 1)
                    ) + domain_bonus

                    target_entry = lexicon.entries[target]
                    grammar_ok = self._is_grammar_ok(target_entry)
                    relation_path = path + [f"{edge.relation}:{target}"]

                    LOGGER.debug(
                        "semantic_recall.relation_candidate target=%s source_seed=%s depth=%s edge_weight=%.6f cumulative_weight=%.6f score=%.6f domain_bonus=%.6f grammar_ok=%s relation_path=%s",
                        target,
                        seed,
                        depth + 1,
                        edge_weight,
                        cumulative_weight,
                        score,
                        domain_bonus,
                        grammar_ok,
                        relation_path,
                    )

                    self._upsert_candidate(
                        candidates_by_word=candidates_by_word,
                        candidate=RecallCandidate(
                            word=target,
                            score=score,
                            source="relation",
                            relation_path=relation_path,
                            axis_distance=0.0,
                            grammar_ok=grammar_ok,
                            note=f"relation_from:{seed}",
                        ),
                    )

                    frontier.append(
                        (
                            target,
                            depth + 1,
                            cumulative_weight * edge_weight,
                            relation_path,
                        )
                    )
                    LOGGER.debug(
                        "semantic_recall.relation_frontier_push target=%s next_depth=%s next_cumulative_weight=%.6f frontier_size=%s",
                        target,
                        depth + 1,
                        cumulative_weight * edge_weight,
                        len(frontier),
                    )

    def _add_axis_candidates(
        self,
        seeds: Sequence[str],
        lexicon: LexiconContainer,
        dialogue_state: DialogueState,
        intent_plan: IntentPlan,
        candidates_by_word: Dict[str, RecallCandidate],
        domain_hints: Set[str],
    ) -> None:
        target_vector = self._build_target_vector(
            seeds=seeds,
            lexicon=lexicon,
            dialogue_state=dialogue_state,
        )

        if target_vector is None:
            LOGGER.debug("semantic_recall.axis_skip reason=no_target_vector")
            return

        LOGGER.debug(
            "semantic_recall.axis_target_vector=%s",
            target_vector.to_dict(),
        )

        probe_words = self._build_axis_probe_words(
            lexicon=lexicon,
            existing_words=set(candidates_by_word.keys()),
            seeds=seeds,
            intent_plan=intent_plan,
            domain_hints=domain_hints,
        )

        LOGGER.debug(
            "semantic_recall.axis_probe begin probe_count=%s sample=%s",
            len(probe_words),
            probe_words[:50],
        )

        scored: List[Tuple[str, float]] = []
        for word in probe_words:
            entry = lexicon.entries.get(word)
            if entry is None:
                LOGGER.debug("semantic_recall.axis_probe_skip word=%s reason=no_entry", word)
                continue

            distance = self._axis_distance(entry.vector, target_vector)
            scored.append((word, distance))
            LOGGER.debug(
                "semantic_recall.axis_distance word=%s pos=%s distance=%.6f vector=%s",
                word,
                entry.grammar.pos,
                distance,
                entry.vector.to_dict(),
            )

        scored.sort(key=lambda item: item[1])

        LOGGER.debug(
            "semantic_recall.axis_sorted_top=%s",
            [
                {
                    "word": word,
                    "distance": round(distance, 6),
                }
                for word, distance in scored[:20]
            ],
        )

        axis_take = max(4, min(self.config.max_candidates, 12))
        LOGGER.debug("semantic_recall.axis_take=%s", axis_take)

        for word, distance in scored[:axis_take]:
            entry = lexicon.entries[word]
            grammar_ok = self._is_grammar_ok(entry)
            bonus = self._intent_axis_bonus(intent=intent_plan.intent, entry=entry) + self._domain_bonus_for_word(target=word, domain_hints=domain_hints)
            score = max(
                0.05,
                self.config.axis_weight_scale * (1.0 / (1.0 + distance)) + bonus,
            )

            LOGGER.debug(
                "semantic_recall.axis_candidate word=%s distance=%.6f base=%.6f bonus=%.6f final_score=%.6f grammar_ok=%s pos=%s",
                word,
                distance,
                self.config.axis_weight_scale * (1.0 / (1.0 + distance)),
                bonus,
                score,
                grammar_ok,
                entry.grammar.pos,
            )

            self._upsert_candidate(
                candidates_by_word=candidates_by_word,
                candidate=RecallCandidate(
                    word=word,
                    score=score,
                    source="axis",
                    relation_path=[],
                    axis_distance=distance,
                    grammar_ok=grammar_ok,
                    note="axis_neighbor",
                ),
            )

    def _build_target_vector(
        self,
        seeds: Sequence[str],
        lexicon: LexiconContainer,
        dialogue_state: DialogueState,
    ) -> Optional[AxisVector]:
        seed_vectors: List[AxisVector] = []

        LOGGER.debug(
            "semantic_recall.build_target_vector.begin seeds=%s dialogue_context_vector=%s",
            list(seeds),
            dialogue_state.context_vector.to_dict(),
        )

        content_preferred = [seed for seed in seeds if (lexicon.entries.get(seed) and lexicon.entries.get(seed).grammar.content_word)]
        seed_sequence = content_preferred or list(seeds)

        for seed in seed_sequence:
            entry = lexicon.entries.get(seed)
            if entry is None:
                LOGGER.debug(
                    "semantic_recall.build_target_vector.skip seed=%s reason=no_entry",
                    seed,
                )
                continue
            seed_vectors.append(entry.vector)
            LOGGER.debug(
                "semantic_recall.build_target_vector.seed_vector seed=%s vector=%s",
                seed,
                entry.vector.to_dict(),
            )

        if not seed_vectors and self._is_zero_axis(dialogue_state.context_vector):
            LOGGER.debug(
                "semantic_recall.build_target_vector.result=None reason=no_seed_vectors_and_zero_dialogue_context"
            )
            return None

        merged = self._average_axis_vectors(seed_vectors)
        LOGGER.debug(
            "semantic_recall.build_target_vector.average=%s",
            merged.to_dict(),
        )

        if not self._is_zero_axis(dialogue_state.context_vector):
            merged = self._blend_axis_vectors(
                first=merged,
                second=dialogue_state.context_vector,
                first_weight=0.8,
                second_weight=0.2,
            )
            LOGGER.debug(
                "semantic_recall.build_target_vector.blended=%s",
                merged.to_dict(),
            )

        return merged

    def _build_axis_probe_words(
        self,
        lexicon: LexiconContainer,
        existing_words: Set[str],
        seeds: Sequence[str],
        intent_plan: IntentPlan,
        domain_hints: Set[str],
    ) -> List[str]:
        probe_words: List[str] = []
        seed_entries = [lexicon.entries[word] for word in seeds if word in lexicon.entries]
        allowed_categories = {entry.category for entry in seed_entries if str(entry.category).strip()}
        allowed_pos = {entry.grammar.pos for entry in seed_entries if str(entry.grammar.pos).strip()}
        allowed_roots = {entry.hierarchy[0] for entry in seed_entries if entry.hierarchy}
        base_iterable: Iterable[str]

        LOGGER.debug(
            "semantic_recall.build_axis_probe_words begin prefer_content_words=%s existing_words=%s allowed_categories=%s allowed_pos=%s allowed_roots=%s domain_hints=%s intent=%s",
            self.config.prefer_content_words_in_axis,
            sorted(existing_words),
            sorted(allowed_categories),
            sorted(allowed_pos),
            sorted(allowed_roots),
            sorted(domain_hints),
            intent_plan.intent,
        )

        if self.config.prefer_content_words_in_axis and lexicon.indexes.content_words:
            base_iterable = list(lexicon.indexes.content_words)
        else:
            base_iterable = list(lexicon.entries.keys())

        for word in base_iterable:
            if word in existing_words:
                continue
            entry = lexicon.entries.get(word)
            if entry is None:
                continue
            if domain_hints and word in self._domain_probe_words(domain_hints, lexicon):
                probe_words.append(word)
                continue
            same_category = bool(allowed_categories and entry.category in allowed_categories)
            same_pos = bool(allowed_pos and entry.grammar.pos in allowed_pos)
            same_root = bool(entry.hierarchy and allowed_roots and entry.hierarchy[0] in allowed_roots)
            if same_category or same_pos or same_root:
                probe_words.append(word)

        if not probe_words:
            for word in base_iterable:
                if word not in existing_words and word in lexicon.entries:
                    probe_words.append(word)

        intent_limit = self._axis_probe_limit_for_intent(intent_plan.intent)
        limit = min(max(1, intent_limit), max(1, int(self.config.axis_probe_limit)))
        if len(probe_words) > limit:
            LOGGER.debug(
                "semantic_recall.build_axis_probe_words.truncate original_count=%s limit=%s",
                len(probe_words),
                limit,
            )
            probe_words = probe_words[:limit]

        LOGGER.debug(
            "semantic_recall.build_axis_probe_words.result count=%s sample=%s",
            len(probe_words),
            probe_words[:50],
        )
        return probe_words

    def _upsert_candidate(
        self,
        candidates_by_word: Dict[str, RecallCandidate],
        candidate: RecallCandidate,
    ) -> None:
        current = candidates_by_word.get(candidate.word)

        if current is None:
            candidates_by_word[candidate.word] = candidate
            LOGGER.debug(
                "semantic_recall.upsert action=insert word=%s source=%s score=%.6f axis_distance=%.6f note=%s",
                candidate.word,
                candidate.source,
                candidate.score,
                candidate.axis_distance,
                candidate.note,
            )
            return

        better = False
        if candidate.score > current.score:
            better = True
        elif math.isclose(candidate.score, current.score) and candidate.axis_distance < current.axis_distance:
            better = True

        if better:
            LOGGER.debug(
                "semantic_recall.upsert action=replace word=%s old_source=%s old_score=%.6f old_axis_distance=%.6f new_source=%s new_score=%.6f new_axis_distance=%.6f",
                candidate.word,
                current.source,
                current.score,
                current.axis_distance,
                candidate.source,
                candidate.score,
                candidate.axis_distance,
            )
            candidates_by_word[candidate.word] = candidate
            return

        merged_note = self._merge_note(current.note, candidate.note)
        if merged_note != current.note:
            LOGGER.debug(
                "semantic_recall.upsert action=merge_note word=%s old_note=%s new_note=%s",
                current.word,
                current.note,
                merged_note,
            )
        current.note = merged_note

        if candidate.source != current.source and f"also_from:{candidate.source}" not in current.note:
            current.note = self._merge_note(current.note, f"also_from:{candidate.source}")
            LOGGER.debug(
                "semantic_recall.upsert action=append_source word=%s appended_source=%s merged_note=%s",
                current.word,
                candidate.source,
                current.note,
            )

        if candidate.relation_path and len(candidate.relation_path) > len(current.relation_path):
            LOGGER.debug(
                "semantic_recall.upsert action=replace_relation_path word=%s old_path=%s new_path=%s",
                current.word,
                current.relation_path,
                candidate.relation_path,
            )
            current.relation_path = list(candidate.relation_path)

        old_grammar_ok = current.grammar_ok
        current.grammar_ok = current.grammar_ok and candidate.grammar_ok
        if current.grammar_ok != old_grammar_ok:
            LOGGER.debug(
                "semantic_recall.upsert action=merge_grammar_ok word=%s old=%s new=%s",
                current.word,
                old_grammar_ok,
                current.grammar_ok,
            )

    def _detect_domain_hints(
        self,
        seeds: Sequence[str],
        input_state: InputState,
        lexicon: LexiconContainer,
        dialogue_state: DialogueState,
    ) -> Set[str]:
        joined = " ".join(list(seeds) + list(input_state.normalized_tokens) + [str(dialogue_state.current_topic or '')]).strip()
        hints: Set[str] = set()
        for name, keywords in DOMAIN_HINT_WORDS.items():
            if any(keyword and keyword in joined for keyword in keywords):
                hints.add(name)
        return hints

    def _domain_probe_words(self, domain_hints: Set[str], lexicon: LexiconContainer) -> List[str]:
        words: List[str] = []
        seen: Set[str] = set()
        for domain in sorted(domain_hints):
            for word in DOMAIN_HINT_WORDS.get(domain, ()):
                if word in lexicon.entries and word not in seen:
                    seen.add(word)
                    words.append(word)
        return words[: max(1, int(self.config.domain_hint_probe_limit))]

    def _axis_probe_limit_for_intent(self, intent: str) -> int:
        if intent == "question":
            return int(self.config.question_axis_probe_limit)
        if intent == "explain":
            return int(self.config.explain_axis_probe_limit)
        return int(self.config.respond_axis_probe_limit)

    def _domain_bonus_for_word(self, target: str, domain_hints: Set[str]) -> float:
        if not self.config.enable_domain_hint_boost or not domain_hints:
            return 0.0
        for domain in domain_hints:
            if target in DOMAIN_HINT_WORDS.get(domain, ()):
                return float(self.config.domain_hint_bonus)
        return 0.0

    def _add_domain_hint_candidates(
        self,
        domain_hints: Set[str],
        lexicon: LexiconContainer,
        candidates_by_word: Dict[str, RecallCandidate],
    ) -> None:
        if not domain_hints:
            return
        for word in self._domain_probe_words(domain_hints, lexicon):
            entry = lexicon.entries.get(word)
            if entry is None:
                continue
            grammar_ok = self._is_grammar_ok(entry)
            score = max(0.05, self.config.axis_weight_scale * 0.75 + self._domain_bonus_for_word(word, domain_hints))
            self._upsert_candidate(
                candidates_by_word=candidates_by_word,
                candidate=RecallCandidate(
                    word=word,
                    score=score,
                    source="axis",
                    relation_path=[],
                    axis_distance=0.0,
                    grammar_ok=grammar_ok,
                    note="domain_hint",
                ),
            )

    def _intent_axis_bonus(self, intent: str, entry: LexiconEntry) -> float:
        pos = entry.grammar.pos
        bonus = 0.0

        if intent == "question":
            if pos in {"noun", "pronoun", "adverb"}:
                bonus = 0.05
        elif intent == "explain":
            if pos in {"noun", "verb", "adjective_i", "adjective_na"}:
                bonus = 0.06
        elif intent == "empathy":
            if pos in {"adjective_i", "adjective_na", "interjection"}:
                bonus = 0.07
        elif intent == "confirm":
            if pos in {"copula", "auxiliary", "noun"}:
                bonus = 0.04

        LOGGER.debug(
            "semantic_recall.intent_axis_bonus intent=%s word=%s pos=%s bonus=%.6f",
            intent,
            entry.word,
            pos,
            bonus,
        )
        return bonus

    def _is_grammar_ok(self, entry: Optional[LexiconEntry]) -> bool:
        if entry is None:
            LOGGER.debug("semantic_recall.is_grammar_ok entry=None result=False")
            return False
        grammar = entry.grammar
        result = grammar.pos != "unknown"
        LOGGER.debug(
            "semantic_recall.is_grammar_ok word=%s pos=%s result=%s",
            entry.word,
            grammar.pos,
            result,
        )
        return result

    def _axis_distance(self, left: AxisVector, right: AxisVector) -> float:
        lv = left.to_dict()
        rv = right.to_dict()
        total = 0.0
        parts: Dict[str, float] = {}

        for key in lv.keys():
            diff = float(lv[key]) - float(rv[key])
            squared = diff * diff
            total += squared
            parts[key] = squared

        distance = math.sqrt(total)
        LOGGER.debug(
            "semantic_recall.axis_distance_calc left=%s right=%s squared_parts=%s total=%.6f distance=%.6f",
            lv,
            rv,
            parts,
            total,
            distance,
        )
        return distance

    def _average_axis_vectors(self, vectors: Sequence[AxisVector]) -> AxisVector:
        if not vectors:
            LOGGER.debug("semantic_recall.average_axis_vectors empty -> zero_vector")
            return AxisVector()

        count = float(len(vectors))
        sums = {
            "valence": 0.0,
            "arousal": 0.0,
            "abstractness": 0.0,
            "sociality": 0.0,
            "temporality": 0.0,
            "agency": 0.0,
            "causality": 0.0,
            "certainty": 0.0,
            "deixis": 0.0,
            "discourse_force": 0.0,
        }

        for index, vector in enumerate(vectors, start=1):
            data = vector.to_dict()
            LOGGER.debug(
                "semantic_recall.average_axis_vectors.input index=%s vector=%s",
                index,
                data,
            )
            for key, value in data.items():
                sums[key] += float(value)

        averaged = AxisVector(
            valence=sums["valence"] / count,
            arousal=sums["arousal"] / count,
            abstractness=sums["abstractness"] / count,
            sociality=sums["sociality"] / count,
            temporality=sums["temporality"] / count,
            agency=sums["agency"] / count,
            causality=sums["causality"] / count,
            certainty=sums["certainty"] / count,
            deixis=sums["deixis"] / count,
            discourse_force=sums["discourse_force"] / count,
        )
        LOGGER.debug(
            "semantic_recall.average_axis_vectors.result count=%s sums=%s averaged=%s",
            count,
            sums,
            averaged.to_dict(),
        )
        return averaged

    def _blend_axis_vectors(
        self,
        first: AxisVector,
        second: AxisVector,
        first_weight: float,
        second_weight: float,
    ) -> AxisVector:
        total = first_weight + second_weight
        if total <= 0.0:
            LOGGER.debug(
                "semantic_recall.blend_axis_vectors invalid_weights first_weight=%.6f second_weight=%.6f -> zero_vector",
                first_weight,
                second_weight,
            )
            return AxisVector()

        a = first.to_dict()
        b = second.to_dict()

        blended = AxisVector(
            valence=((a["valence"] * first_weight) + (b["valence"] * second_weight)) / total,
            arousal=((a["arousal"] * first_weight) + (b["arousal"] * second_weight)) / total,
            abstractness=((a["abstractness"] * first_weight) + (b["abstractness"] * second_weight)) / total,
            sociality=((a["sociality"] * first_weight) + (b["sociality"] * second_weight)) / total,
            temporality=((a["temporality"] * first_weight) + (b["temporality"] * second_weight)) / total,
            agency=((a["agency"] * first_weight) + (b["agency"] * second_weight)) / total,
            causality=((a["causality"] * first_weight) + (b["causality"] * second_weight)) / total,
            certainty=((a["certainty"] * first_weight) + (b["certainty"] * second_weight)) / total,
            deixis=((a["deixis"] * first_weight) + (b["deixis"] * second_weight)) / total,
            discourse_force=((a["discourse_force"] * first_weight) + (b["discourse_force"] * second_weight)) / total,
        )
        LOGGER.debug(
            "semantic_recall.blend_axis_vectors first=%s second=%s first_weight=%.6f second_weight=%.6f result=%s",
            a,
            b,
            first_weight,
            second_weight,
            blended.to_dict(),
        )
        return blended

    def _is_zero_axis(self, vector: AxisVector) -> bool:
        values = vector.to_dict()
        result = all(abs(v) < 1e-12 for v in values.values())
        LOGGER.debug(
            "semantic_recall.is_zero_axis values=%s result=%s",
            values,
            result,
        )
        return result

    def _merge_note(self, left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        if right in left:
            return left
        return f"{left} | {right}"


def recall_semantics(
    input_state: InputState,
    lexicon: LexiconContainer,
    dialogue_state: Optional[DialogueState] = None,
    intent_plan: Optional[IntentPlan] = None,
    config: Optional[SemanticRecallConfig] = None,
) -> RecallResult:
    engine = SemanticRecallEngine(config=config)
    return engine.recall(
        input_state=input_state,
        lexicon=lexicon,
        dialogue_state=dialogue_state,
        intent_plan=intent_plan,
    )