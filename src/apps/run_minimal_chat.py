from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from src.core.io.lsd_lexicon import load_lexicon_container
from src.core.logging.trace_logger import JsonlTraceLogger
from src.core.planner.intent_planner import IntentPlannerConfig, plan_intent
from src.core.recall.semantic_recall import SemanticRecallConfig, recall_semantics
from src.core.scoring.basic_scorer import BasicScorerConfig, choose_best_response
from src.core.schema import (
    ActionCandidateSnapshot,
    DialogueState,
    EpisodeAction,
    ExternalRewardBreakdown,
    ExternalRewardComponent,
    InternalRewardBreakdown,
    LexiconContainer,
    RewardBreakdown,
    SlotTrace,
    SlotTraceItem,
    TokenizationToken,
    TraceLog,
    UnknownSpan,
    build_input_state,
    build_runtime_state_snapshot,
    dataclass_to_dict,
    new_episode_id,
    new_session_id,
    new_turn_id,
)
from src.core.slots.slot_filler import SlotFillerConfig, fill_slots
from src.core.surface.surface_realizer import SurfaceRealizerConfig, realize_surface
from src.training.policy_memory import PolicyMemoryConfig, PolicyMemoryStore
from src.training.unknown_word_learner import (
    UnknownWordLearner,
    build_unknown_word_learner_config,
)
from src.utils.settings import apply_arg_defaults, build_dataclass_config, get_setting, load_settings

LOGGER = logging.getLogger(__name__)
JST = ZoneInfo("Asia/Tokyo")


@dataclass(slots=True)
class TokenizationResult:
    tokens: List[str] = field(default_factory=list)
    normalized_tokens: List[str] = field(default_factory=list)
    tokenization: List[TokenizationToken] = field(default_factory=list)
    unknown_spans: List[UnknownSpan] = field(default_factory=list)


class SurfaceNormalizer:
    def __init__(self, lexicon: LexiconContainer) -> None:
        self.lexicon = lexicon
        settings = load_settings()
        unknown_word_data = get_setting(settings, "pipeline", "unknown_word", default={})
        self.unknown_word_config = build_unknown_word_learner_config(unknown_word_data)
        self.unknown_word_learner: Optional[UnknownWordLearner] = None
        if self.unknown_word_config.enabled:
            self.unknown_word_learner = UnknownWordLearner(
                lexicon=self.lexicon,
                config=self.unknown_word_config,
            )
            self.unknown_word_learner.apply_existing_overlay()
        self._refresh_indexes()

    def _refresh_indexes(self) -> None:
        self.surface_map = self._build_surface_map(self.lexicon)
        self.length_index = self._build_length_index(self.surface_map.keys())
        self.pos_by_surface = self._build_pos_by_surface(self.lexicon)
        self.proper_noun_by_surface = self._build_proper_noun_by_surface(self.lexicon)
        self.named_entity_type_by_surface = self._build_named_entity_type_by_surface(self.lexicon)

    def refresh_runtime_lexicon(self) -> None:
        self._refresh_indexes()

    def normalize_token(self, token: str) -> List[str]:
        result = self.tokenize_text(str(token or ""))
        return list(result.normalized_tokens)

    def normalize_text(self, raw_text: str) -> List[str]:
        result = self.tokenize_text(raw_text)
        return list(result.normalized_tokens)

    def tokenize_words(self, explicit_words: Sequence[str]) -> TokenizationResult:
        tokenization: List[TokenizationToken] = []
        unknown_spans: List[UnknownSpan] = []
        for index, word in enumerate(explicit_words):
            piece = str(word or "").strip()
            if not piece:
                continue
            part = self.tokenize_text(piece)
            if not part.tokenization:
                continue
            tokenization.extend(part.tokenization)
            unknown_spans.extend(part.unknown_spans)
        normalized = [token for item in tokenization for token in item.normalized_tokens if token]
        raw_tokens = [item.surface for item in tokenization if item.surface]
        return TokenizationResult(
            tokens=raw_tokens,
            normalized_tokens=normalized,
            tokenization=tokenization,
            unknown_spans=unknown_spans,
        )

    def tokenize_text(self, raw_text: str) -> TokenizationResult:
        text = str(raw_text or "").strip()
        if not text:
            return TokenizationResult()

        tokenization: List[TokenizationToken] = []
        unknown_spans: List[UnknownSpan] = []
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch.isspace():
                i += 1
                continue

            matched = self._longest_match(text, i)
            if matched and not self._prefer_unknown_over_match(text, i, matched):
                tokenization.append(
                    TokenizationToken(
                        surface=matched,
                        normalized_tokens=list(self.surface_map[matched]),
                        known=True,
                        pos_hint=self.pos_by_surface.get(matched, ""),
                        proper_noun_candidate=bool(self.proper_noun_by_surface.get(matched, False)),
                        named_entity_type_hint=self.named_entity_type_by_surface.get(matched, ""),
                        start=i,
                        end=i + len(matched),
                        reason="lexicon_longest_match",
                    )
                )
                i += len(matched)
                continue

            surface, end, pos_hint, proper_noun_candidate, named_entity_type_hint = self._consume_unknown_span(text, i, ignore_single_char_matches=True)
            tokenization.append(
                TokenizationToken(
                    surface=surface,
                    normalized_tokens=[surface],
                    known=False,
                    pos_hint=pos_hint,
                    proper_noun_candidate=proper_noun_candidate,
                    named_entity_type_hint=named_entity_type_hint,
                    start=i,
                    end=end,
                    reason="unknown_span",
                )
            )
            unknown_spans.append(
                UnknownSpan(
                    surface=surface,
                    start=i,
                    end=end,
                    reason="unmatched_surface_span",
                    pos_hint=pos_hint,
                    proper_noun_candidate=proper_noun_candidate,
                    named_entity_type_hint=named_entity_type_hint,
                    status="detected",
                    suggested_words=[surface],
                )
            )
            i = end

        normalized = [token for item in tokenization for token in item.normalized_tokens if token]
        raw_tokens = [item.surface for item in tokenization if item.surface]
        return TokenizationResult(
            tokens=raw_tokens,
            normalized_tokens=normalized,
            tokenization=tokenization,
            unknown_spans=unknown_spans,
        )

    def maybe_learn_unknown_words(self, *, raw_text: str, tokenization_result: TokenizationResult) -> Dict[str, object]:
        if not self.unknown_word_learner or not tokenization_result.unknown_spans:
            return {
                "enabled": bool(self.unknown_word_learner),
                "examined_spans": [],
                "applied_words": [],
                "skipped_spans": [],
                "pending_records": [],
                "retokenized": False,
            }
        learning_result = self.unknown_word_learner.learn_unknown_spans(
            raw_text=raw_text,
            unknown_spans=tokenization_result.unknown_spans,
        )
        if learning_result.applied_words:
            self.refresh_runtime_lexicon()
        return {
            "enabled": True,
            "examined_spans": list(learning_result.examined_spans),
            "applied_words": list(learning_result.applied_words),
            "skipped_spans": list(learning_result.skipped_spans),
            "pending_records": list(learning_result.pending_records),
            "overlay_path": learning_result.overlay_path,
            "pending_path": learning_result.pending_path,
            "retokenized": bool(learning_result.applied_words),
        }

    def _build_surface_map(self, lexicon: LexiconContainer) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for entry in lexicon.entries.values():
            if entry.word:
                mapping.setdefault(entry.word, [entry.word])
            for alias in entry.aliases:
                alias_text = str(alias).strip()
                if alias_text:
                    mapping.setdefault(alias_text, [entry.word])
            for form in entry.surface_forms:
                surface = str(form.surface).strip()
                if not surface:
                    continue
                mapping[surface] = list(form.tokens or [entry.word])
        return mapping

    def _build_pos_by_surface(self, lexicon: LexiconContainer) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for entry in lexicon.entries.values():
            pos = str(entry.grammar.pos or "")
            if entry.word and pos:
                mapping.setdefault(entry.word, pos)
            for alias in entry.aliases:
                alias_text = str(alias).strip()
                if alias_text and pos:
                    mapping.setdefault(alias_text, pos)
            for form in entry.surface_forms:
                surface = str(form.surface).strip()
                if surface and pos:
                    mapping.setdefault(surface, pos)
        return mapping

    def _build_proper_noun_by_surface(self, lexicon: LexiconContainer) -> Dict[str, bool]:
        mapping: Dict[str, bool] = {}
        for entry in lexicon.entries.values():
            proper = bool(entry.category == "proper_noun" or entry.grammar.proper_noun or entry.grammar.sub_pos == "proper_noun")
            if not proper:
                continue
            if entry.word:
                mapping[entry.word] = True
            for alias in entry.aliases:
                alias_text = str(alias).strip()
                if alias_text:
                    mapping[alias_text] = True
            for form in entry.surface_forms:
                surface = str(form.surface).strip()
                if surface:
                    mapping[surface] = True
        return mapping

    def _build_named_entity_type_by_surface(self, lexicon: LexiconContainer) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for entry in lexicon.entries.values():
            entity_type = str(entry.grammar.named_entity_type or entry.meta.get("named_entity_type", "")).strip()
            if not entity_type:
                continue
            if entry.word:
                mapping.setdefault(entry.word, entity_type)
            for alias in entry.aliases:
                alias_text = str(alias).strip()
                if alias_text:
                    mapping.setdefault(alias_text, entity_type)
            for form in entry.surface_forms:
                surface = str(form.surface).strip()
                if surface:
                    mapping.setdefault(surface, entity_type)
        return mapping

    def _build_length_index(self, words: Iterable[str]) -> Dict[int, set[str]]:
        index: Dict[int, set[str]] = {}
        for word in words:
            w = str(word).strip()
            if not w:
                continue
            index.setdefault(len(w), set()).add(w)
        return index

    def _longest_match(self, text: str, start: int) -> str:
        remaining = len(text) - start
        lengths = sorted((length for length in self.length_index if length <= remaining), reverse=True)
        for length in lengths:
            piece = text[start : start + length]
            if piece in self.length_index[length]:
                return piece
        return ""

    def _consume_unknown_span(self, text: str, start: int, ignore_single_char_matches: bool = False) -> Tuple[str, int, str, bool, str]:
        n = len(text)
        if start >= n:
            return "", start, "unknown", False, ""
        ch = text[start]
        if self._is_hard_boundary(ch):
            return ch, start + 1, "symbol", False, ""

        end = start + 1
        while end < n:
            current = text[end]
            if current.isspace():
                break
            if self._is_hard_boundary(current):
                break
            if end > start:
                next_match = self._longest_match(text, end)
                if next_match and not (ignore_single_char_matches and len(next_match) == 1):
                    break
            if end > start and self._should_break_unknown_span(text[end - 1], current):
                break
            end += 1
        if end <= start:
            end = start + 1
        surface = text[start:end]
        pos_hint, proper_noun_candidate, named_entity_type_hint = self._guess_unknown_hints(
            surface=surface,
            full_text=text,
            start=start,
            end=end,
        )
        return surface, end, pos_hint, proper_noun_candidate, named_entity_type_hint

    def _prefer_unknown_over_match(self, text: str, start: int, matched: str) -> bool:
        if len(str(matched)) != 1:
            return False
        if start + 1 >= len(text):
            return False
        if text[start + 1].isspace() or self._is_hard_boundary(text[start + 1]):
            return False
        preview_surface, _, _, _, _ = self._consume_unknown_span(text, start, ignore_single_char_matches=True)
        if len(preview_surface) <= 1:
            return False
        pos = self.pos_by_surface.get(matched, "")
        if pos in {"particle", "auxiliary"}:
            return True
        matched_class = self._char_class(matched)
        next_class = self._char_class(text[start + 1])
        return matched_class not in {"symbol", "space"} and next_class not in {"symbol", "space"}

    def _guess_unknown_hints(self, *, surface: str, full_text: str, start: int, end: int) -> Tuple[str, bool, str]:
        text = str(surface or "").strip()
        if not text:
            return "unknown", False, ""

        pos_hint = "unknown"
        proper_noun_candidate = False
        named_entity_type_hint = ""

        if all('ぁ' <= ch <= 'ん' or ch == 'ー' for ch in text):
            if text.endswith('に'):
                pos_hint = 'adverb'
            elif text.endswith('い'):
                pos_hint = 'adjective'
            else:
                pos_hint = 'adverb'
        elif all('ァ' <= ch <= 'ヶ' or ch == 'ー' for ch in text):
            pos_hint = 'noun'
            if len(text) >= 3:
                proper_noun_candidate = True
                named_entity_type_hint = 'product'
        elif all(ch.isascii() and (ch.isalpha() or ch.isdigit() or ch in {'_', '-', '.'}) for ch in text):
            pos_hint = 'noun'
            proper_noun_candidate = True
            named_entity_type_hint = 'service'
        elif any('一' <= ch <= '龯' for ch in text):
            pos_hint = 'noun'
        else:
            pos_hint = 'unknown'

        if any(ch.isascii() and ch.isupper() for ch in text):
            proper_noun_candidate = True
            named_entity_type_hint = named_entity_type_hint or 'organization'

        if '・' in text or '·' in text:
            proper_noun_candidate = True
            named_entity_type_hint = named_entity_type_hint or 'person'

        next_window = full_text[end:end + 3]
        if next_window.startswith(('さん', 'ちゃん', 'くん', '氏', '様', '先生')):
            pos_hint = 'noun'
            proper_noun_candidate = True
            named_entity_type_hint = 'person'

        if text.startswith(('株式会社', '有限会社', '合同会社', '学校法人')):
            pos_hint = 'noun'
            proper_noun_candidate = True
            named_entity_type_hint = 'organization'

        if text.endswith(('駅', '市', '町', '村', '県', '府', '都')) and any('一' <= ch <= '龯' for ch in text):
            proper_noun_candidate = True
            named_entity_type_hint = named_entity_type_hint or 'place'

        if proper_noun_candidate and pos_hint == 'unknown':
            pos_hint = 'noun'

        return pos_hint, proper_noun_candidate, named_entity_type_hint

    def _should_break_unknown_span(self, left: str, right: str) -> bool:
        left_class = self._char_class(left)
        right_class = self._char_class(right)
        if left_class == 'latin' and right_class == 'latin':
            return False
        if left_class == 'digit' and right_class == 'digit':
            return False
        if left_class == 'kanji' and right_class == 'hiragana':
            return False
        if left_class == 'hiragana' and right_class == 'kanji':
            return True
        return left_class != right_class and 'symbol' not in {left_class, right_class}

    def _char_class(self, ch: str) -> str:
        if ch.isspace():
            return 'space'
        if 'ぁ' <= ch <= 'ん' or ch == 'ー':
            return 'hiragana'
        if 'ァ' <= ch <= 'ヶ':
            return 'katakana'
        if '一' <= ch <= '龯':
            return 'kanji'
        if ch.isdigit():
            return 'digit'
        if ch.isascii() and ch.isalpha():
            return 'latin'
        return 'symbol'

    def _is_hard_boundary(self, ch: str) -> bool:
        return ch in "、。,.!?！？()（）[]{}「」『』:;：；/\\"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSLM v3 minimal chat runner")
    parser.add_argument("--lexicon", default=None, help="辞書ファイルパス (.json / .lsd / .lsdx)")
    parser.add_argument("--text", default="", help="入力テキスト")
    parser.add_argument("--words", nargs="*", default=None, help="すでに分かち書き済みの入力トークン列")
    parser.add_argument("--trace-dir", default=None, help="trace JSONL の保存先ディレクトリ")
    parser.add_argument("--policy-memory", default=None, help="学習済み方針メモリ JSON")
    parser.add_argument("--no-policy-memory", action="store_true", help="学習済み方針メモリを使わない")
    parser.add_argument("--no-trace", action="store_true", help="trace JSONL を保存しない")
    parser.add_argument("--console-debug", action="store_true", help="main 側互換用フラグ")
    return parser.parse_args(argv)




def resolve_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    settings = load_settings()
    return apply_arg_defaults(
        args,
        settings,
        [
            ("lexicon", ("paths", "lexicon"), "libs/dict.lsdx"),
            ("trace_dir", ("paths", "trace_dir"), "runtime/traces"),
            ("policy_memory", ("paths", "policy_memory"), "runtime/policy_memory.json"),
        ],
    )

def load_lexicon(path: Path) -> LexiconContainer:
    LOGGER.info("lexicon.load.start path=%s", path)
    container_raw = load_lexicon_container(path)
    lexicon = LexiconContainer.from_dict(container_raw)
    LOGGER.info("lexicon.load.done entries=%s version=%s", len(lexicon.entries), lexicon.meta.version)
    return lexicon


def build_raw_text(args: argparse.Namespace) -> str:
    if args.words:
        return " ".join(str(word) for word in args.words if str(word).strip())
    return str(args.text or "").strip()


def build_tokenization_result(
    raw_text: str,
    explicit_words: Optional[Sequence[str]],
    normalizer: SurfaceNormalizer,
) -> TokenizationResult:
    if explicit_words:
        return normalizer.tokenize_words(explicit_words)

    if not raw_text:
        return TokenizationResult()

    if " " in raw_text:
        return normalizer.tokenize_words(raw_text.split())

    return normalizer.tokenize_text(raw_text)


def build_tokens(
    raw_text: str,
    explicit_words: Optional[Sequence[str]],
    normalizer: SurfaceNormalizer,
) -> List[str]:
    return list(build_tokenization_result(raw_text=raw_text, explicit_words=explicit_words, normalizer=normalizer).normalized_tokens)


def build_slot_trace(filled_slots) -> SlotTrace:
    constraint_map = {constraint.name: constraint for constraint in filled_slots.frame.constraints}

    all_slots: List[str] = []
    for slot_name in list(constraint_map.keys()) + list(filled_slots.values.keys()) + list(filled_slots.missing_required) + list(filled_slots.optional_unfilled):
        if slot_name not in all_slots:
            all_slots.append(slot_name)

    items: List[SlotTraceItem] = []
    for slot_name in all_slots:
        value = filled_slots.values.get(slot_name)
        constraint = constraint_map.get(slot_name)
        items.append(
            SlotTraceItem(
                slot_name=slot_name,
                expected=constraint is not None,
                required=bool(constraint.required) if constraint else False,
                filled=value is not None,
                value=value.value if value else "",
                confidence=float(value.confidence) if value else 0.0,
                source_candidate=value.source_candidate if value else "",
                inferred=bool(value.inferred) if value else False,
                note=value.note if value else (constraint.note if constraint else ""),
            )
        )

    return SlotTrace(
        predicate=filled_slots.frame.predicate,
        predicate_type=filled_slots.frame.predicate_type,
        frame_constraints=list(constraint_map.keys()),
        all_slots=all_slots,
        filled_slots=items,
        missing_required=list(filled_slots.missing_required),
        optional_unfilled=list(filled_slots.optional_unfilled),
        consistency_score=float(filled_slots.consistency_score),
    )


def build_reward_from_response(response, evaluation: Optional[List[object]] = None) -> RewardBreakdown:
    evaluation = evaluation or []

    internal = InternalRewardBreakdown(
        semantic=float(response.score.semantic_consistency),
        slot=float(response.score.slot_fitness),
        grammar=float(response.score.grammar_fitness),
        retention=float(response.score.input_retention),
        policy=float(response.score.policy_fitness),
        total=float(response.score.total),
        reasons=list(response.score.reasons),
    )

    external_components: List[ExternalRewardComponent] = []
    for item in evaluation:
        weighted_score = float(item.score)
        external_components.append(
            ExternalRewardComponent(
                evaluator_name=str(item.evaluator_name),
                score=float(item.score),
                weight=1.0,
                weighted_score=weighted_score,
                label=str(item.label),
                feedback=str(item.feedback),
                metadata=dict(getattr(item, "metadata", {}) or {}),
            )
        )

    external_total = sum(component.weighted_score for component in external_components)
    external = ExternalRewardBreakdown(
        components=external_components,
        total=float(external_total),
    )

    total = float(internal.total + external.total)
    reasons = list(internal.reasons)
    for component in external.components:
        if component.label:
            reasons.append(f"external:{component.evaluator_name}:{component.label}")

    return RewardBreakdown(
        internal=internal,
        external=external,
        total=total,
        reasons=reasons,
        metadata={
            "response_text": response.text,
            "intent": response.intent,
            "policy": response.policy,
            "formula": "total = internal.total + external.total",
        },
    )


def build_episode_actions(
    intent_plan,
    recall_result,
    filled_slots,
    surface_plan,
    scored_candidates,
    response,
) -> List[EpisodeAction]:
    actions: List[EpisodeAction] = []

    intent_candidates = [
        ActionCandidateSnapshot(
            key=intent_name,
            label=intent_name,
            score=float(intent_plan.confidence) if intent_name == intent_plan.intent else 0.0,
            rank=1 if intent_name == intent_plan.intent else 0,
            source="rule",
            kept=intent_name == intent_plan.intent,
            dropped=intent_name != intent_plan.intent,
            drop_reason="not_selected" if intent_name != intent_plan.intent else "",
        )
        for intent_name in ["respond", "empathy", "question", "confirm", "explain", "unknown"]
    ]
    actions.append(
        EpisodeAction(
            stage="intent",
            action_type="choose_intent",
            selected={
                "intent": intent_plan.intent,
                "response_policy_hint": intent_plan.response_policy_hint,
            },
            candidates=intent_candidates,
            confidence=float(intent_plan.confidence),
            note=intent_plan.note,
            candidate_count=len(intent_candidates),
            selected_count=1,
            dropped_count=max(0, len(intent_candidates) - 1),
            metadata={
                "required_slots": list(intent_plan.required_slots),
                "optional_slots": list(intent_plan.optional_slots),
            },
        )
    )

    ranked_recall = sorted(
        recall_result.candidates,
        key=lambda item: float(item.score),
        reverse=True,
    )
    recall_keep_count = min(8, len(ranked_recall))

    recall_ranked_candidates: List[ActionCandidateSnapshot] = []
    for idx, item in enumerate(ranked_recall[: min(24, len(ranked_recall))], start=1):
        kept = idx <= recall_keep_count
        recall_ranked_candidates.append(
            ActionCandidateSnapshot(
                key=item.word,
                label=item.word,
                score=float(item.score),
                rank=idx,
                source=str(item.source),
                kept=kept,
                dropped=not kept,
                drop_reason="below_keep_rank" if not kept else "",
                metadata={
                    "axis_distance": float(item.axis_distance),
                    "grammar_ok": bool(item.grammar_ok),
                    "relation_path": list(item.relation_path),
                    "note": item.note,
                },
            )
        )

    kept_recall_words = [item.word for item in ranked_recall[:recall_keep_count]]
    dropped_recall_words = [item.word for item in ranked_recall[recall_keep_count : min(24, len(ranked_recall))]]

    actions.append(
        EpisodeAction(
            stage="recall",
            action_type="rank_recall_candidates",
            selected={
                "top_word": kept_recall_words[0] if kept_recall_words else "",
                "keep_rank_limit": recall_keep_count,
            },
            candidates=recall_ranked_candidates,
            confidence=float(ranked_recall[0].score) if ranked_recall else 0.0,
            note="recall candidates ranked before pruning",
            candidate_count=len(recall_ranked_candidates),
            selected_count=1 if kept_recall_words else 0,
            dropped_count=max(0, len(recall_ranked_candidates) - 1),
            metadata={
                "seed_count": len(recall_result.seeds),
                "seed_words": list(recall_result.seeds),
            },
        )
    )
    actions.append(
        EpisodeAction(
            stage="recall",
            action_type="prune_recall_candidates",
            selected={
                "kept_words": kept_recall_words,
                "dropped_words": dropped_recall_words,
            },
            candidates=recall_ranked_candidates,
            confidence=float(ranked_recall[0].score) if ranked_recall else 0.0,
            note="top ranked recall candidates kept for downstream slot filling",
            candidate_count=len(recall_ranked_candidates),
            selected_count=len(kept_recall_words),
            dropped_count=len(dropped_recall_words),
            metadata={
                "kept_count": len(kept_recall_words),
                "dropped_count": len(dropped_recall_words),
            },
        )
    )

    constraint_map = {constraint.name: constraint for constraint in filled_slots.frame.constraints}
    slot_names: List[str] = []
    for slot_name in (
        list(constraint_map.keys())
        + list(filled_slots.values.keys())
        + list(filled_slots.missing_required)
        + list(filled_slots.optional_unfilled)
    ):
        if slot_name not in slot_names:
            slot_names.append(slot_name)

    frame_slot_names = list(constraint_map.keys())
    derived_slot_names = [slot_name for slot_name in slot_names if slot_name not in constraint_map]

    slot_frame_candidates: List[ActionCandidateSnapshot] = []
    for idx, slot_name in enumerate(slot_names, start=1):
        constraint = constraint_map.get(slot_name)
        is_expected = constraint is not None

        slot_frame_candidates.append(
            ActionCandidateSnapshot(
                key=slot_name,
                label=slot_name,
                score=1.0 if is_expected else 0.0,
                rank=idx,
                source="slot_frame",
                kept=is_expected,
                dropped=not is_expected,
                drop_reason="" if is_expected else "not_in_frame",
                metadata={
                    "required": bool(constraint.required) if constraint else False,
                    "expected": is_expected,
                    "note": constraint.note if constraint else "derived_slot_not_in_frame",
                },
            )
        )

    slot_value_candidates: List[ActionCandidateSnapshot] = []
    filled_slot_values: Dict[str, Dict[str, object]] = {}
    for idx, slot_name in enumerate(slot_names, start=1):
        slot_value = filled_slots.values.get(slot_name)
        constraint = constraint_map.get(slot_name)
        kept = slot_value is not None

        slot_value_candidates.append(
            ActionCandidateSnapshot(
                key=slot_name,
                label=slot_value.value if slot_value else slot_name,
                score=float(slot_value.confidence) if slot_value else 0.0,
                rank=idx,
                source=slot_value.source_candidate if slot_value else "slot_frame",
                kept=kept,
                dropped=not kept,
                drop_reason=(
                    "missing_required"
                    if slot_name in filled_slots.missing_required
                    else ("optional_unfilled" if slot_name in filled_slots.optional_unfilled else "unfilled")
                ) if not kept else "",
                metadata={
                    "required": bool(constraint.required) if constraint else False,
                    "expected": constraint is not None,
                    "inferred": bool(slot_value.inferred) if slot_value else False,
                    "note": slot_value.note if slot_value else (constraint.note if constraint else ""),
                },
            )
        )

        if slot_value is not None:
            filled_slot_values[slot_name] = {
                "value": slot_value.value,
                "confidence": float(slot_value.confidence),
                "source_candidate": slot_value.source_candidate,
                "inferred": bool(slot_value.inferred),
            }

    actions.append(
        EpisodeAction(
            stage="slot",
            action_type="resolve_slot_frame",
            selected={
                "predicate": filled_slots.frame.predicate,
                "predicate_type": filled_slots.frame.predicate_type,
                "frame_constraints": frame_slot_names,
                "derived_slots": derived_slot_names,
            },
            candidates=slot_frame_candidates,
            confidence=float(filled_slots.consistency_score),
            note=filled_slots.frame.predicate,
            candidate_count=len(slot_frame_candidates),
            selected_count=len(frame_slot_names),
            dropped_count=len(derived_slot_names),
            metadata={
                "expected_slot_count": len(frame_slot_names),
                "derived_slot_count": len(derived_slot_names),
                "missing_required": list(filled_slots.missing_required),
                "optional_unfilled": list(filled_slots.optional_unfilled),
            },
        )
    )
    actions.append(
        EpisodeAction(
            stage="slot",
            action_type="select_slot_values",
            selected={
                "filled_slots": filled_slot_values,
                "missing_required": list(filled_slots.missing_required),
                "optional_unfilled": list(filled_slots.optional_unfilled),
            },
            candidates=slot_value_candidates,
            confidence=float(filled_slots.consistency_score),
            note="slot values resolved from recall candidates",
            candidate_count=len(slot_value_candidates),
            selected_count=len(filled_slot_values),
            dropped_count=len(slot_value_candidates) - len(filled_slot_values),
            metadata={
                "predicate": filled_slots.frame.predicate,
                "predicate_type": filled_slots.frame.predicate_type,
                "filled_count": len(filled_slot_values),
                "unfilled_count": len(slot_value_candidates) - len(filled_slot_values),
            },
        )
    )

    chosen_text = response.text
    chosen_template = surface_plan.template_id
    if response.chosen_candidate is not None:
        chosen_template = response.chosen_candidate.template_id or chosen_template

    surface_candidates: List[ActionCandidateSnapshot] = []
    for idx, item in enumerate(scored_candidates, start=1):
        kept = item.text == chosen_text
        surface_candidates.append(
            ActionCandidateSnapshot(
                key=item.template_id or f"candidate_{idx}",
                label=item.text,
                score=float(item.final_score),
                rank=idx,
                source="surface_realizer",
                kept=kept,
                dropped=not kept,
                drop_reason="not_selected" if not kept else "",
                metadata={
                    "template_id": item.template_id,
                    "slot_coverage": float(item.slot_coverage),
                    "semantic_score": float(item.semantic_score),
                    "grammar_violations": list(item.grammar_violations),
                },
            )
        )

    actions.append(
        EpisodeAction(
            stage="surface",
            action_type="choose_surface_candidate",
            selected={
                "text": chosen_text,
                "template_id": chosen_template,
                "policy": response.policy,
            },
            candidates=surface_candidates,
            confidence=float(response.score.total),
            note=surface_plan.note,
            candidate_count=len(surface_candidates),
            selected_count=1 if surface_candidates else 0,
            dropped_count=max(0, len(surface_candidates) - 1),
            metadata={
                "style": surface_plan.style,
                "politeness": surface_plan.politeness,
                "sentence_count": int(surface_plan.sentence_count),
                "order": list(surface_plan.order),
                "auxiliaries": list(surface_plan.auxiliaries),
            },
        )
    )

    return actions

def build_dialogue_state_after(dialogue_state: DialogueState, intent_plan, filled_slots, response) -> DialogueState:
    state_after = deepcopy(dialogue_state)
    state_after.inferred_intent_history.append(intent_plan.intent)
    state_after.inferred_intent_history = state_after.inferred_intent_history[-20:]

    actor = response.used_slots.get("actor") or (
        filled_slots.values["actor"].value if "actor" in filled_slots.values else ""
    )
    target = response.used_slots.get("target") or (
        filled_slots.values["target"].value if "target" in filled_slots.values else ""
    )
    topic = response.used_slots.get("topic") or (
        filled_slots.values["topic"].value if "topic" in filled_slots.values else ""
    )

    if actor:
        state_after.last_subject = actor
    if target:
        state_after.last_object = target
    if topic:
        state_after.current_topic = topic
    elif filled_slots.frame.predicate:
        state_after.current_topic = filled_slots.frame.predicate

    state_after.variables["last_response_text"] = response.text
    state_after.variables["last_policy"] = response.policy
    state_after.variables["last_intent"] = response.intent
    return state_after


def run_pipeline(
    args: argparse.Namespace,
    lexicon: LexiconContainer,
    normalizer: SurfaceNormalizer,
) -> Tuple[str, Optional[Path]]:
    args = resolve_runtime_args(args)
    settings = load_settings()
    lexicon_path = Path(args.lexicon)
    if not lexicon_path.exists():
        LOGGER.error("lexicon_not_found path=%s", lexicon_path)
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")

    LOGGER.info("minimal_chat.start lexicon=%s", lexicon_path)

    raw_text = build_raw_text(args)
    if not raw_text.strip():
        LOGGER.error("empty_input")
        raise ValueError("Input text is empty. Use --text or --words.")

    tokenization_result = build_tokenization_result(raw_text=raw_text, explicit_words=args.words, normalizer=normalizer)
    unknown_word_learning = normalizer.maybe_learn_unknown_words(raw_text=raw_text, tokenization_result=tokenization_result)
    if unknown_word_learning.get("retokenized"):
        tokenization_result = build_tokenization_result(raw_text=raw_text, explicit_words=args.words, normalizer=normalizer)
    tokens = list(tokenization_result.normalized_tokens)
    LOGGER.info("input raw_text=%s", raw_text)
    LOGGER.info("input tokens=%s", tokens)
    if tokenization_result.unknown_spans:
        LOGGER.info("input unknown_spans=%s", [item.surface for item in tokenization_result.unknown_spans])

    session_id = new_session_id()
    turn_id = new_turn_id()
    episode_id = new_episode_id()

    input_state = build_input_state(
        raw_text=raw_text,
        tokens=list(tokenization_result.tokens or tokens),
        normalized_tokens=tokens,
        tokenization=list(tokenization_result.tokenization),
        unknown_spans=list(tokenization_result.unknown_spans),
        session_id=session_id,
        turn_id=turn_id,
        timestamp=datetime.now(JST).isoformat(timespec="seconds"),
    )
    dialogue_state = DialogueState()
    dialogue_state_before = deepcopy(dialogue_state)

    intent_planner_config = build_dataclass_config(IntentPlannerConfig, get_setting(settings, "pipeline", "intent_planner", default={}))
    recall_config = build_dataclass_config(SemanticRecallConfig, get_setting(settings, "pipeline", "semantic_recall", default={}))
    slot_filler_config = build_dataclass_config(SlotFillerConfig, get_setting(settings, "pipeline", "slot_filler", default={}))
    surface_config = build_dataclass_config(SurfaceRealizerConfig, get_setting(settings, "pipeline", "surface_realizer", default={}))
    scorer_config = build_dataclass_config(BasicScorerConfig, get_setting(settings, "pipeline", "basic_scorer", default={}))
    policy_memory_config = build_dataclass_config(PolicyMemoryConfig, get_setting(settings, "policy_memory", default={}))
    policy_memory_limit = max(1, int(get_setting(settings, "learning", "runtime", "policy_memory_limit", default=4)))

    intent_plan = plan_intent(input_state=input_state, dialogue_state=dialogue_state, config=intent_planner_config)
    recall_result = recall_semantics(
        input_state=input_state,
        lexicon=lexicon,
        dialogue_state=dialogue_state,
        intent_plan=intent_plan,
        config=recall_config,
    )
    filled_slots = fill_slots(
        input_state=input_state,
        recall_result=recall_result,
        lexicon=lexicon,
        intent_plan=intent_plan,
        dialogue_state=dialogue_state,
        config=slot_filler_config,
    )
    surface_plan, candidates = realize_surface(
        filled_slots=filled_slots,
        intent_plan=intent_plan,
        lexicon=lexicon,
        config=surface_config,
    )

    policy_memory_matches: List[dict[str, object]] = []
    if not args.no_policy_memory:
        policy_memory = PolicyMemoryStore(args.policy_memory, config=policy_memory_config, autoload=True)
        memory_candidates, policy_memory_matches = policy_memory.suggest(
            intent_plan=intent_plan,
            filled_slots=filled_slots,
            existing_texts=[item.text for item in candidates],
            limit=policy_memory_limit,
        )
        if memory_candidates:
            LOGGER.info('policy_memory.augmented count=%s', len(memory_candidates))
            candidates = list(candidates) + list(memory_candidates)

    response, scored_candidates = choose_best_response(
        input_state=input_state,
        intent_plan=intent_plan,
        filled_slots=filled_slots,
        candidates=candidates,
        config=scorer_config,
    )

    evaluation: List[object] = []
    reward = build_reward_from_response(response=response, evaluation=evaluation)
    slot_trace = build_slot_trace(filled_slots)
    actions = build_episode_actions(
        intent_plan=intent_plan,
        recall_result=recall_result,
        filled_slots=filled_slots,
        surface_plan=surface_plan,
        scored_candidates=scored_candidates,
        response=response,
    )
    dialogue_state_after = build_dialogue_state_after(
        dialogue_state=dialogue_state_before,
        intent_plan=intent_plan,
        filled_slots=filled_slots,
        response=response,
    )

    trace = TraceLog(
        session_id=session_id,
        turn_id=turn_id,
        episode_id=episode_id,
        timestamp=datetime.now(JST).isoformat(timespec="seconds"),
        input_state=input_state,
        dialogue_state=dialogue_state_before,
        dialogue_state_after=dialogue_state_after,
        state_before=build_runtime_state_snapshot(input_state, dialogue_state_before),
        state_after=build_runtime_state_snapshot(input_state, dialogue_state_after),
        intent_plan=intent_plan,
        recall_result=recall_result,
        filled_slots=filled_slots,
        surface_plan=surface_plan,
        candidates=scored_candidates,
        actions=actions,
        slot_trace=slot_trace,
        response=response,
        reward=reward,
        evaluation=evaluation,
        debug={
            "lexicon_path": str(lexicon_path),
            "entry_count": len(lexicon.entries),
            "raw_text": raw_text,
            "tokens": tokens,
            "tokenization": [dataclass_to_dict(item) for item in tokenization_result.tokenization],
            "unknown_spans": [dataclass_to_dict(item) for item in tokenization_result.unknown_spans],
            "unknown_word_learning": unknown_word_learning,
            "response_score": dataclass_to_dict(response.score),
            "reward": dataclass_to_dict(reward),
            "slot_trace": dataclass_to_dict(slot_trace),
            "policy_memory_matches": policy_memory_matches,
        },
    )

    trace_path: Optional[Path] = None
    if not args.no_trace:
        trace_logger = JsonlTraceLogger(
            log_dir=args.trace_dir,
            latest_name="latest_trace.jsonl",
            rotate_on_start=False,
        )
        trace_path = trace_logger.append_episode_trace(trace)
        LOGGER.info("trace_saved path=%s episode_id=%s", trace_path, episode_id)

    LOGGER.info("response chosen=%s total=%.4f internal=%.4f external=%.4f episode_id=%s", response.text, reward.total, reward.internal.total, reward.external.total, episode_id)
    return response.text, trace_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = resolve_runtime_args(parse_args(argv))

    from src.utils.logging import setup_logging
    import logging as root_logging

    setup_logging(
        app_name="lslm_minimal_chat",
        console_level=root_logging.DEBUG if args.console_debug else root_logging.INFO,
    )

    args = resolve_runtime_args(args)
    settings = load_settings()
    lexicon_path = Path(args.lexicon)
    if not lexicon_path.exists():
        raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")

    lexicon = load_lexicon(lexicon_path)
    normalizer = SurfaceNormalizer(lexicon)

    response_text, _ = run_pipeline(
        args,
        lexicon,
        normalizer,
    )

    sys.stdout.write(response_text + "\n")


if __name__ == "__main__":
    main()