from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence

from src.core.schema import FilledSlots, RealizationCandidate

_PUNCT_RE = re.compile(r'[\s、。？！!?,，．・「」『』（）()\[\]{}]+')


@dataclass(slots=True)
class TeacherGuidanceConfig:
    target_weight: float = 0.35
    min_override_delta: float = 0.015
    min_alignment_gain: float = 0.05
    max_blended_regression: float = 0.01
    min_override_alignment: float = 0.50
    max_slot_score_regression: float = 0.20


@dataclass(slots=True)
class TeacherCandidateRank:
    index: int
    text: str
    base_score: float
    target_alignment: float
    blended_score: float
    slot_score: float = 0.0


@dataclass(slots=True)
class TeacherGuidanceResult:
    selected_index: int = 0
    selected_text: str = ''
    overridden: bool = False
    guard_blocked: bool = False
    guard_reason: str = ''
    rankings: List[TeacherCandidateRank] = field(default_factory=list)


class TeacherGuidedReranker:
    def __init__(self, config: TeacherGuidanceConfig | None = None) -> None:
        self.config = config or TeacherGuidanceConfig()

    def rerank(
        self,
        *,
        candidates: Sequence[RealizationCandidate],
        target_text: str,
        filled_slots: FilledSlots | None = None,
    ) -> TeacherGuidanceResult:
        if not candidates:
            return TeacherGuidanceResult()

        cleaned_target = self._normalize_text(target_text)
        if not cleaned_target:
            return TeacherGuidanceResult(
                selected_index=0,
                selected_text=candidates[0].text,
                overridden=False,
                rankings=[
                    TeacherCandidateRank(
                        index=index,
                        text=item.text,
                        base_score=float(item.final_score),
                        target_alignment=0.0,
                        blended_score=float(item.final_score),
                        slot_score=self._slot_alignment_score(item.text, filled_slots),
                    )
                    for index, item in enumerate(candidates)
                ],
            )

        target_weight = max(0.0, min(1.0, float(self.config.target_weight)))
        rankings: List[TeacherCandidateRank] = []
        for index, item in enumerate(candidates):
            alignment = self._alignment_score(item.text, cleaned_target)
            blended = ((1.0 - target_weight) * float(item.final_score)) + (target_weight * alignment)
            rankings.append(
                TeacherCandidateRank(
                    index=index,
                    text=item.text,
                    base_score=float(item.final_score),
                    target_alignment=alignment,
                    blended_score=blended,
                    slot_score=self._slot_alignment_score(item.text, filled_slots),
                )
            )

        rankings.sort(
            key=lambda item: (item.blended_score, item.target_alignment, item.base_score),
            reverse=True,
        )
        selected = rankings[0]
        base_index = max(range(len(candidates)), key=lambda idx: float(candidates[idx].final_score))
        base_ranking = next((item for item in rankings if item.index == base_index), None)
        if base_ranking is None:
            base_ranking = TeacherCandidateRank(
                index=base_index,
                text=candidates[base_index].text,
                base_score=float(candidates[base_index].final_score),
                target_alignment=self._alignment_score(candidates[base_index].text, cleaned_target),
                blended_score=float(candidates[base_index].final_score),
                slot_score=self._slot_alignment_score(candidates[base_index].text, filled_slots),
            )

        blended_gain = selected.blended_score - base_ranking.blended_score
        alignment_gain = selected.target_alignment - base_ranking.target_alignment
        has_enough_blended_gain = blended_gain >= float(self.config.min_override_delta)
        has_alignment_win = (
            alignment_gain >= float(self.config.min_alignment_gain)
            and selected.blended_score >= (base_ranking.blended_score - float(self.config.max_blended_regression))
        )
        overridden = selected.index != base_index and (has_enough_blended_gain or has_alignment_win)
        guard_blocked = False
        guard_reason = ''

        if overridden:
            has_alignment_floor = selected.target_alignment >= float(self.config.min_override_alignment)
            keeps_slot_structure = (
                selected.slot_score + float(self.config.max_slot_score_regression)
                >= base_ranking.slot_score
            )
            if not has_alignment_floor:
                overridden = False
                guard_blocked = True
                guard_reason = 'override_alignment_below_floor'
            elif not keeps_slot_structure:
                overridden = False
                guard_blocked = True
                guard_reason = 'override_slot_regression'

        final_index = selected.index if overridden else base_index
        final_text = candidates[final_index].text

        return TeacherGuidanceResult(
            selected_index=final_index,
            selected_text=final_text,
            overridden=overridden,
            guard_blocked=guard_blocked,
            guard_reason=guard_reason,
            rankings=rankings,
        )

    def _alignment_score(self, text: str, target_text: str) -> float:
        normalized_text = self._normalize_text(text)
        normalized_target = self._normalize_text(target_text)
        if not normalized_text or not normalized_target:
            return 0.0
        if normalized_text == normalized_target:
            return 1.0

        token_score = self._token_f1(normalized_text, normalized_target)
        bigram_score = self._bigram_f1(normalized_text, normalized_target)
        length_score = self._length_score(normalized_text, normalized_target)
        score = (token_score * 0.45) + (bigram_score * 0.45) + (length_score * 0.10)
        return max(0.0, min(1.0, score))

    def _slot_alignment_score(self, text: str, filled_slots: FilledSlots | None) -> float:
        if filled_slots is None:
            return 0.0
        normalized_text = self._normalize_text(text)
        if not normalized_text:
            return 0.0

        values = [
            self._normalize_text(slot.value)
            for slot in filled_slots.values.values()
            if self._normalize_text(slot.value)
        ]
        if not values:
            return 0.0

        hits = 0
        for value in values:
            if value in normalized_text:
                hits += 1
        return hits / float(len(values))

    def _normalize_text(self, text: str) -> str:
        return _PUNCT_RE.sub('', str(text or '')).strip()

    def _token_f1(self, text: str, target: str) -> float:
        text_tokens = self._tokenize(text)
        target_tokens = self._tokenize(target)
        if not text_tokens or not target_tokens:
            return 0.0
        text_set = set(text_tokens)
        target_set = set(target_tokens)
        overlap = len(text_set & target_set)
        if overlap <= 0:
            return 0.0
        precision = overlap / float(len(text_set))
        recall = overlap / float(len(target_set))
        if precision + recall <= 0.0:
            return 0.0
        return (2.0 * precision * recall) / (precision + recall)

    def _bigram_f1(self, text: str, target: str) -> float:
        text_bigrams = self._char_ngrams(text, 2)
        target_bigrams = self._char_ngrams(target, 2)
        if not text_bigrams or not target_bigrams:
            return 0.0
        overlap = len(text_bigrams & target_bigrams)
        if overlap <= 0:
            return 0.0
        precision = overlap / float(len(text_bigrams))
        recall = overlap / float(len(target_bigrams))
        if precision + recall <= 0.0:
            return 0.0
        return (2.0 * precision * recall) / (precision + recall)

    def _length_score(self, text: str, target: str) -> float:
        if not text or not target:
            return 0.0
        longer = max(len(text), len(target))
        shorter = min(len(text), len(target))
        return shorter / float(longer) if longer > 0 else 0.0

    def _tokenize(self, text: str) -> List[str]:
        chunks = [chunk for chunk in re.split(r'(?<=[ぁ-んァ-ヶー一-龠a-zA-Z0-9])(?=[ぁ-んァ-ヶー一-龠a-zA-Z0-9])', text) if chunk]
        if chunks:
            return chunks
        return [text]

    def _char_ngrams(self, text: str, n: int) -> set[str]:
        if len(text) < n:
            return {text} if text else set()
        return {text[index:index + n] for index in range(len(text) - n + 1)}
