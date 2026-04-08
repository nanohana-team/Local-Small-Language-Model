from __future__ import annotations

import argparse
import json
import os
import logging
import re
import sys
from time import perf_counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from src.core.io.lsd_lexicon import load_lexicon_container
from src.core.logging.audit_helpers import build_dict_update_event, build_stage_metric, build_turn_audit_summary
from src.core.logging.trace_logger import JsonlTraceLogger
from src.core.planner.intent_planner import IntentPlannerConfig, plan_intent
from src.core.recall.semantic_recall import SemanticRecallConfig, recall_semantics
from src.core.scoring.basic_scorer import BasicScorerConfig, choose_best_response
from src.core.schema import (
    ActionCandidateSnapshot,
    AxisVector,
    DecisionCandidateTrace,
    DecisionTrace,
    DialogueState,
    EpisodeAction,
    ExternalRewardBreakdown,
    ExternalRewardComponent,
    InternalRewardBreakdown,
    GrammarConstraints,
    LexiconContainer,
    LexiconEntry,
    RealizationCandidate,
    RewardBreakdown,
    ScoreBreakdown,
    ResponseResult,
    SlotValue,
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


@dataclass(slots=True)
class InputSegment:
    text: str
    start: int = 0
    end: int = 0
    question_like: bool = False
    alternative_branch: bool = False
    score: float = 0.0


@dataclass(slots=True)
class InputFocusConfig:
    enabled: bool = True
    max_segment_chars: int = 140
    max_segments: int = 8
    long_text_threshold: int = 80
    alternative_markers: Tuple[str, ...] = ("それとも", "または", "あるいは", "or", "OR")


@dataclass(slots=True)
class InputFocusDecision:
    original_text: str
    focused_text: str
    segments: List[InputSegment] = field(default_factory=list)
    used_segmentation: bool = False
    has_alternative_question: bool = False
    question_like_segment_count: int = 0
    reason: str = ''


@dataclass(slots=True)
class ResponseAccumulationConfig:
    enabled: bool = True
    max_sentences: int = 3
    max_chars: int = 180
    min_chars_to_expand: int = 24
    include_slot_detail: bool = True
    include_followup: bool = True
    include_context_bridge: bool = True
    similarity_threshold: float = 0.82


@dataclass(slots=True)
class ChatHistoryConfig:
    enabled: bool = True
    max_turns: int = 12
    recent_response_window: int = 4


@dataclass(slots=True)
class ChatTurnRecord:
    role: str
    text: str
    intent: str = ''
    topic: str = ''
    policy: str = ''
    turn_id: str = ''
    timestamp: str = ''


@dataclass(slots=True)
class ChatRuntimeContext:
    session_id: str = ''
    dialogue_state: DialogueState = field(default_factory=DialogueState)
    history: List[ChatTurnRecord] = field(default_factory=list)
    history_path: str = ''
    history_enabled: bool = True
    max_turns: int = 12
    recent_response_window: int = 4

    def trim_history(self) -> None:
        limit = max(1, int(self.max_turns)) * 2
        if len(self.history) > limit:
            self.history = self.history[-limit:]

    def recent_response_texts(self) -> List[str]:
        texts = [item.text for item in self.history if item.role == 'assistant' and str(item.text).strip()]
        window = max(1, int(self.recent_response_window))
        return texts[-window:]

    def recent_user_texts(self) -> List[str]:
        texts = [item.text for item in self.history if item.role == 'user' and str(item.text).strip()]
        window = max(1, int(self.recent_response_window))
        return texts[-window:]


def _normalize_sentence(text: str) -> str:
    value = str(text or '').strip()
    if not value:
        return ''
    if not value.endswith(('。', '？', '!', '！')):
        value += '。'
    return value


def _text_similarity(left: str, right: str) -> float:
    left = str(left or '').strip()
    right = str(right or '').strip()
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if len(left) < 2 or len(right) < 2:
        return 1.0 if left == right else 0.0
    left_bigrams = {left[index:index + 2] for index in range(len(left) - 1)}
    right_bigrams = {right[index:index + 2] for index in range(len(right) - 1)}
    if not left_bigrams or not right_bigrams:
        return 0.0
    return len(left_bigrams & right_bigrams) / float(len(left_bigrams | right_bigrams))


def _dedupe_sentences_keep_order(sentences: Sequence[str], similarity_threshold: float = 0.82) -> List[str]:
    kept: List[str] = []
    normalized_kept: List[str] = []
    for sentence in sentences:
        normalized = _normalize_sentence(sentence)
        if not normalized:
            continue
        if any(_text_similarity(normalized.rstrip('。？！!?'), prev.rstrip('。？！!?')) >= similarity_threshold for prev in normalized_kept):
            continue
        kept.append(normalized)
        normalized_kept.append(normalized)
    return kept


def _restore_dialogue_state(data: Optional[Dict[str, Any]]) -> DialogueState:
    if not isinstance(data, dict):
        return DialogueState()
    context_vector = data.get('context_vector', {})
    return DialogueState(
        current_topic=str(data.get('current_topic', '') or ''),
        last_subject=str(data.get('last_subject', '') or ''),
        last_object=str(data.get('last_object', '') or ''),
        referents={str(k): str(v) for k, v in dict(data.get('referents', {}) or {}).items()},
        context_vector=AxisVector.from_dict(context_vector),
        inferred_intent_history=[str(item) for item in list(data.get('inferred_intent_history', []) or []) if str(item).strip()],
        variables=dict(data.get('variables', {}) or {}),
    )


def load_chat_runtime_context(
    *,
    session_id: str = '',
    history_path: str = '',
    history_enabled: bool = True,
    max_turns: int = 12,
    recent_response_window: int = 4,
    reset: bool = False,
) -> ChatRuntimeContext:
    context = ChatRuntimeContext(
        session_id=str(session_id or '').strip() or new_session_id(),
        history_path=str(history_path or '').strip(),
        history_enabled=bool(history_enabled),
        max_turns=max(1, int(max_turns)),
        recent_response_window=max(1, int(recent_response_window)),
    )
    if not context.history_enabled or not context.history_path or reset:
        return context

    path = Path(context.history_path)
    if not path.exists():
        return context

    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        LOGGER.exception('chat_history.load_failed path=%s', path)
        return context

    if isinstance(payload, dict):
        stored_session_id = str(payload.get('session_id', '') or '').strip()
        if stored_session_id:
            context.session_id = stored_session_id
        context.dialogue_state = _restore_dialogue_state(payload.get('dialogue_state'))
        history_items = []
        for item in list(payload.get('history', []) or []):
            if not isinstance(item, dict):
                continue
            text = str(item.get('text', '') or '').strip()
            role = str(item.get('role', '') or '').strip()
            if not text or role not in {'user', 'assistant'}:
                continue
            history_items.append(ChatTurnRecord(
                role=role,
                text=text,
                intent=str(item.get('intent', '') or ''),
                topic=str(item.get('topic', '') or ''),
                policy=str(item.get('policy', '') or ''),
                turn_id=str(item.get('turn_id', '') or ''),
                timestamp=str(item.get('timestamp', '') or ''),
            ))
        context.history = history_items
        context.trim_history()
        LOGGER.info('chat_history.loaded path=%s turns=%s session_id=%s', path, len(context.history), context.session_id)
    return context


def save_chat_runtime_context(context: ChatRuntimeContext) -> Optional[Path]:
    if not context.history_enabled or not context.history_path:
        return None
    path = Path(context.history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    context.trim_history()
    payload = {
        'session_id': context.session_id,
        'saved_at': datetime.now(JST).isoformat(timespec='seconds'),
        'dialogue_state': dataclass_to_dict(context.dialogue_state),
        'history': [dataclass_to_dict(item) for item in context.history],
    }
    path.write_text(json.dumps(dataclass_to_dict(payload), ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def _append_history_item(
    context: Optional[ChatRuntimeContext],
    *,
    role: str,
    text: str,
    turn_id: str,
    intent: str = '',
    topic: str = '',
    policy: str = '',
) -> None:
    if context is None or not context.history_enabled:
        return
    normalized = _normalize_sentence(text) if role == 'assistant' else str(text or '').strip()
    if not normalized:
        return
    context.history.append(
        ChatTurnRecord(
            role=role,
            text=normalized,
            intent=intent,
            topic=topic,
            policy=policy,
            turn_id=turn_id,
            timestamp=datetime.now(JST).isoformat(timespec='seconds'),
        )
    )
    context.trim_history()


def _history_summary(context: Optional[ChatRuntimeContext], max_items: int = 6) -> List[Dict[str, str]]:
    if context is None or not context.history:
        return []
    items = context.history[-max(1, int(max_items)): ]
    return [
        {
            'role': item.role,
            'text': item.text,
            'intent': item.intent,
            'topic': item.topic,
            'policy': item.policy,
            'turn_id': item.turn_id,
            'timestamp': item.timestamp,
        }
        for item in items
    ]


def _maybe_apply_topic_history_fallback(
    *,
    raw_text: str,
    dialogue_state: DialogueState,
    filled_slots,
) -> bool:
    current_topic = str(dialogue_state.current_topic or '').strip()
    if not current_topic:
        return False
    if len(str(raw_text or '').strip()) > 24:
        return False

    generic_topics = {'これ', 'それ', 'あれ', 'こと', 'もの', '何', 'なに', '何か', 'どう', 'どれ', '良い', 'いい'}
    topic_slot = filled_slots.values.get('topic')
    topic_value = str(topic_slot.value or '').strip() if topic_slot is not None else ''
    if topic_value and topic_value not in generic_topics:
        return False

    filled_slots.values['topic'] = SlotValue(
        slot_name='topic',
        value=current_topic,
        confidence=max(0.62, float(topic_slot.confidence) if topic_slot is not None else 0.0),
        source_candidate=current_topic,
        inferred=True,
        note='topic_from_history_short_followup',
    )
    if 'topic' in filled_slots.optional_unfilled:
        filled_slots.optional_unfilled = [name for name in filled_slots.optional_unfilled if name != 'topic']
    filled_slots.consistency_score = max(float(filled_slots.consistency_score), 0.62)
    return True


def _build_accumulated_response(
    *,
    base_text: str,
    input_state,
    intent_plan,
    filled_slots,
    dialogue_state: DialogueState,
    runtime_context: Optional[ChatRuntimeContext],
    config: ResponseAccumulationConfig,
    lexicon: Optional[LexiconContainer],
) -> Tuple[str, Dict[str, Any]]:
    normalized_base = _normalize_sentence(base_text)
    if not config.enabled or not normalized_base:
        return normalized_base or str(base_text or '').strip(), {
            'enabled': bool(config.enabled),
            'applied': False,
            'reason': 'disabled_or_empty',
            'sentences': [normalized_base] if normalized_base else [],
        }
    if len(normalized_base) >= int(config.max_chars):
        return normalized_base, {
            'enabled': True,
            'applied': False,
            'reason': 'base_too_long',
            'sentences': [normalized_base],
        }

    topic = ''
    state = ''
    cause = ''
    time_value = ''
    location = ''
    target = ''
    predicate = ''
    if filled_slots.values.get('topic') is not None:
        topic = str(filled_slots.values['topic'].value or '').strip()
    if filled_slots.values.get('state') is not None:
        state = str(filled_slots.values['state'].value or '').strip()
    if filled_slots.values.get('cause') is not None:
        cause = str(filled_slots.values['cause'].value or '').strip()
    if filled_slots.values.get('time') is not None:
        time_value = str(filled_slots.values['time'].value or '').strip()
    if filled_slots.values.get('location') is not None:
        location = str(filled_slots.values['location'].value or '').strip()
    if filled_slots.values.get('target') is not None:
        target = str(filled_slots.values['target'].value or '').strip()
    if filled_slots.values.get('predicate') is not None:
        predicate = str(filled_slots.values['predicate'].value or '').strip()

    topic = topic or str(dialogue_state.current_topic or '').strip()
    focus = topic or target

    extras: List[str] = []
    if config.include_context_bridge and len(str(input_state.raw_text or '').strip()) <= 20 and topic:
        extras.append(f'今の流れだと、{topic}の話として続けて考えるのが自然です。')

    if config.include_slot_detail:
        if intent_plan.intent == 'explain':
            if focus and cause:
                extras.append(f'{focus}を見るときは、{cause}も一緒に切り分けると整理しやすいです。')
            elif focus and state:
                extras.append(f'{focus}は、いまは{state}として見ると捉えやすいです。')
        elif intent_plan.intent == 'ask_recommendation':
            if focus and state:
                extras.append(f'{focus}は、{state}寄りかどうかを軸にすると候補を絞りやすいです。')
            elif focus:
                extras.append(f'{focus}は、条件を一つ決めるだけでも選びやすさがかなり変わります。')
        elif intent_plan.intent == 'ask_progress':
            if focus and time_value:
                extras.append(f'{time_value}時点で見るのか最新で見るのかで、{focus}の見え方は少し変わります。')
            elif focus:
                extras.append(f'{focus}は、今どの段階かを一つずつ分けて見ると把握しやすいです。')
        elif intent_plan.intent == 'ask_availability':
            if time_value or location:
                joined = '・'.join([value for value in [time_value, location] if value])
                extras.append(f'{joined}の条件が固まっているなら、そこから先に見るとかなり早いです。')
        elif intent_plan.intent == 'ask_fact':
            if focus:
                extras.append(f'{focus}は、前提条件を先に固定すると解釈のズレを減らしやすいです。')
        elif intent_plan.intent == 'empathy':
            if state and cause:
                extras.append(f'{cause}が重なっているなら、{state}になるのも無理はないです。')
        elif intent_plan.intent == 'confirm':
            if focus:
                extras.append(f'{focus}については、その前提で進めて問題なさそうです。')
        else:
            if focus and predicate:
                extras.append(f'{focus}の中では、特に{predicate}の部分を掘ると話が進めやすいです。')
            elif focus and state:
                extras.append(f'{focus}は、いまは{state}の方向として受け取るのが自然です。')

    if config.include_followup:
        if intent_plan.intent in {'ask_recommendation', 'ask_progress', 'ask_availability', 'ask_fact'}:
            extras.append('条件や範囲がもう一つあると、ここからかなり具体化できます。')
        elif intent_plan.intent == 'explain':
            extras.append('必要なら、このまま要点を一つずつ分けて整理できます。')
        elif intent_plan.intent in {'respond', 'smalltalk_expand', 'share_experience'}:
            extras.append('気になる点が一つあれば、そこだけ続けて掘れます。')

    sentences = _dedupe_sentences_keep_order([normalized_base] + extras, similarity_threshold=float(config.similarity_threshold))
    limited: List[str] = []
    current_length = 0
    for sentence in sentences:
        projected = current_length + len(sentence)
        if limited and (len(limited) >= int(config.max_sentences) or projected > int(config.max_chars)):
            break
        limited.append(sentence)
        current_length = projected

    if len(normalized_base) < int(config.min_chars_to_expand) and len(limited) == 1 and extras:
        for sentence in sentences[1:]:
            candidate_length = current_length + len(sentence)
            if len(limited) >= int(config.max_sentences) or candidate_length > int(config.max_chars):
                break
            limited.append(sentence)
            current_length = candidate_length
            break

    final_text = ''.join(limited) if limited else normalized_base
    applied = len(limited) > 1 and final_text != normalized_base
    return final_text, {
        'enabled': True,
        'applied': applied,
        'reason': 'expanded' if applied else 'base_only',
        'sentences': limited,
        'base_text': normalized_base,
        'history_size': len(runtime_context.history) if runtime_context is not None else 0,
    }


BUILTIN_DISCOURSE_ENTRIES: tuple[dict[str, object], ...] = (
    {"word": "かな", "aliases": ["かなぁ"], "pos": "sentence_ending_particle", "category": "discourse_particle", "can_end": True, "certainty": 0.22, "force": 0.58},
    {"word": "かも", "aliases": ["かもね"], "pos": "sentence_ending_particle", "category": "discourse_particle", "can_end": True, "certainty": 0.18, "force": 0.52},
    {"word": "よね", "aliases": ["だよね", "ですよね"], "pos": "sentence_ending_particle", "category": "discourse_particle", "can_end": True, "certainty": 0.42, "force": 0.54},
    {"word": "ね", "aliases": ["ねぇ"], "pos": "sentence_ending_particle", "category": "discourse_particle", "can_end": True, "certainty": 0.36, "force": 0.48},
    {"word": "よ", "aliases": [], "pos": "sentence_ending_particle", "category": "discourse_particle", "can_end": True, "certainty": 0.58, "force": 0.62},
)

EXTERNAL_KNOWLEDGE_TOPICS: dict[str, tuple[str, ...]] = {
    "weather": ("天気", "気温", "雨", "晴れ", "曇り", "雪", "台風", "傘", "湿度"),
    "time": ("今何時", "時間", "時刻", "何時", "何日", "日付"),
    "news": ("ニュース", "速報", "最近", "話題", "最新"),
    "inventory": ("在庫", "売ってる", "ある", "残ってる"),
}


class SurfaceNormalizer:
    def __init__(self, lexicon: LexiconContainer) -> None:
        self.lexicon = lexicon
        settings = load_settings()
        unknown_word_data = get_setting(settings, "pipeline", "unknown_word", default={})
        self.unknown_word_config = build_unknown_word_learner_config(unknown_word_data)
        pending_override = str(os.environ.get("LSLM_UNKNOWN_WORD_PENDING_PATH", "") or "").strip()
        overlay_override = str(os.environ.get("LSLM_UNKNOWN_WORD_OVERLAY_PATH", "") or "").strip()
        if pending_override:
            self.unknown_word_config.pending_path = pending_override
        if overlay_override:
            self.unknown_word_config.overlay_path = overlay_override
        self.unknown_word_learner: Optional[UnknownWordLearner] = None
        if self.unknown_word_config.enabled:
            self.unknown_word_learner = UnknownWordLearner(
                lexicon=self.lexicon,
                config=self.unknown_word_config,
            )
            self.unknown_word_learner.apply_existing_overlay()
        self._inject_builtin_discourse_entries()
        self._refresh_indexes()

    def _inject_builtin_discourse_entries(self) -> None:
        for spec in BUILTIN_DISCOURSE_ENTRIES:
            word = str(spec.get("word", "") or "").strip()
            if not word or word in self.lexicon.entries:
                continue
            self.lexicon.entries[word] = LexiconEntry(
                word=word,
                category=str(spec.get("category", "discourse_particle") or "discourse_particle"),
                hierarchy=["function_words", "sentence_ending_particles"],
                vector=AxisVector(
                    certainty=float(spec.get("certainty", 0.3) or 0.3),
                    discourse_force=float(spec.get("force", 0.5) or 0.5),
                    sociality=0.35,
                    valence=0.50,
                ),
                grammar=GrammarConstraints(
                    pos=str(spec.get("pos", "sentence_ending_particle") or "sentence_ending_particle"),
                    sub_pos="modality",
                    can_end=bool(spec.get("can_end", True)),
                    independent=False,
                    content_word=False,
                    function_word=True,
                ),
                aliases=[str(alias).strip() for alias in list(spec.get("aliases", []) or []) if str(alias).strip()],
                style_tags=["discourse", "modality"],
                meta={"builtin": True, "source": "builtin_discourse_entries"},
            )

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
        text = self._apply_pre_token_replacements(str(raw_text or "")).strip()
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
            "relearned_words": list(getattr(learning_result, "relearned_words", []) or []),
            "quarantined_records": list(getattr(learning_result, "quarantined_records", []) or []),
            "skipped_spans": list(learning_result.skipped_spans),
            "pending_records": list(learning_result.pending_records),
            "overlay_path": learning_result.overlay_path,
            "pending_path": learning_result.pending_path,
            "quarantine_path": str(getattr(getattr(self.unknown_word_learner, "config", None), "quarantine_path", "") or ""),
            "retokenized": bool(learning_result.applied_words),
        }

    def _apply_pre_token_replacements(self, text: str) -> str:
        if not text:
            return text
        replacements = {
            "それとも": " それとも ",
            "または": " または ",
            "あるいは": " あるいは ",
            "けれども": " けれども ",
            "ですが": " ですが ",
        }
        normalized = str(text)
        for src, dst in replacements.items():
            normalized = normalized.replace(src, dst)
        return normalized

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


def _looks_like_question_text(text: str) -> bool:
    text = str(text or '').strip()
    if not text:
        return False
    question_words = ("どう", "どこ", "いつ", "なに", "何", "どんな", "どれ", "どうして", "なぜ")
    if any(marker in text for marker in ('?', '？')):
        return True
    if any(word in text for word in question_words):
        return True
    return bool(re.search(r'(ですか|ますか|でしょうか|かな|か)$', text))


def _split_input_segments(raw_text: str, config: InputFocusConfig) -> List[InputSegment]:
    text = str(raw_text or '').strip()
    if not text:
        return []

    markers = tuple(config.alternative_markers)
    segments: List[InputSegment] = []
    start = 0
    i = 0
    alternative_branch = False

    def push(end_index: int, *, alt_branch: bool = False) -> None:
        nonlocal start
        piece = text[start:end_index].strip(' 、。！？?!')
        if not piece:
            start = end_index
            return
        question_like = _looks_like_question_text(piece)
        score = (3.0 if question_like else 0.0) + min(2.0, len(piece) / 40.0)
        segments.append(
            InputSegment(
                text=piece,
                start=start,
                end=end_index,
                question_like=question_like,
                alternative_branch=alt_branch,
                score=score,
            )
        )
        start = end_index

    while i < len(text):
        matched_marker = None
        for marker in markers:
            if text.startswith(marker, i):
                matched_marker = marker
                break
        if matched_marker is not None:
            push(i, alt_branch=alternative_branch)
            i += len(matched_marker)
            start = i
            alternative_branch = True
            continue
        ch = text[i]
        if ch in '。！？?!':
            push(i + 1, alt_branch=alternative_branch)
            alternative_branch = False
            i += 1
            continue
        if i - start >= config.max_segment_chars and ch in '、,，;； ':
            push(i + 1, alt_branch=alternative_branch)
            alternative_branch = False
        i += 1

    if start < len(text):
        push(len(text), alt_branch=alternative_branch)

    if len(segments) > config.max_segments:
        segments = segments[: config.max_segments]
    return segments


def choose_input_focus(raw_text: str, normalizer: SurfaceNormalizer, config: InputFocusConfig) -> InputFocusDecision:
    text = str(raw_text or '').strip()
    if not text or not config.enabled:
        return InputFocusDecision(original_text=text, focused_text=text, reason='focus_disabled')

    segments = _split_input_segments(text, config)
    if len(segments) <= 1 and len(text) <= config.long_text_threshold:
        return InputFocusDecision(
            original_text=text,
            focused_text=text,
            segments=segments,
            used_segmentation=False,
            has_alternative_question=False,
            question_like_segment_count=sum(1 for seg in segments if seg.question_like),
            reason='single_segment',
        )

    best_segment = None
    best_score = float('-inf')
    for seg in segments:
        normalized_tokens = normalizer.normalize_text(seg.text)
        content_hits = 0
        question_hits = 0
        for token in normalized_tokens:
            entry = normalizer.lexicon.entries.get(token)
            if entry is None:
                continue
            if entry.grammar.content_word:
                content_hits += 1
            if entry.grammar.pos == 'question_word' or token in {'どう', 'どこ', 'いつ', 'なに', '何', 'どんな', 'どれ', 'かな', 'か'}:
                question_hits += 1
        seg.score += (content_hits * 0.20) + (question_hits * 1.4)
        if seg.question_like:
            seg.score += 2.4
            if seg.text.endswith(('?', '？', 'かな', 'ですか')):
                seg.score += 1.2
            if seg.end >= len(text):
                seg.score += 1.0
        if seg.alternative_branch:
            seg.score += 0.4
        if seg.score > best_score:
            best_score = seg.score
            best_segment = seg

    if best_segment is None:
        best_segment = InputSegment(text=text, question_like=_looks_like_question_text(text), score=0.0)

    question_like_count = sum(1 for seg in segments if seg.question_like)
    has_alternative_question = question_like_count >= 2 and any(seg.alternative_branch for seg in segments)
    return InputFocusDecision(
        original_text=text,
        focused_text=best_segment.text or text,
        segments=segments,
        used_segmentation=True,
        has_alternative_question=has_alternative_question,
        question_like_segment_count=question_like_count,
        reason='segment_focus' if len(segments) > 1 else 'long_text_focus',
    )


def _segment_summary(text: str, max_len: int = 16) -> str:
    summary = re.sub(r'[。！？?!]+$', '', str(text or '').strip())
    summary = re.sub(r'^(こんにちは|こんばんは|おはよう)[、。!！]*', '', summary).strip()
    if len(summary) > max_len:
        return summary[:max_len].rstrip() + '…'
    return summary or 'その話'




def detect_external_knowledge_topic(raw_text: str, tokens: Sequence[str]) -> str:
    joined = f"{str(raw_text or '').strip()} {' '.join(str(token) for token in tokens if str(token).strip())}"
    for topic, keywords in EXTERNAL_KNOWLEDGE_TOPICS.items():
        if any(keyword and keyword in joined for keyword in keywords):
            return topic
    return ''


def build_external_lookup_guidance_candidate(raw_text: str, topic: str) -> RealizationCandidate | None:
    text = str(raw_text or '').strip()
    if not text or not topic:
        return None
    topic_map = {
        'weather': '今の天気は内部辞書だけでは確定できないので、外部の天気情報を確認して答えるべき内容です。',
        'time': '現在時刻や日付は内部辞書だけでは確定できないので、外部の時刻情報を確認して答えるべき内容です。',
        'news': '最新ニュースは内部辞書だけでは確定できないので、外部情報を確認して答えるべき内容です。',
        'inventory': '在庫や販売状況は内部辞書だけでは確定できないので、外部情報を確認して答えるべき内容です。',
    }
    guidance = topic_map.get(topic, 'この質問は外部情報の確認が必要です。')
    return RealizationCandidate(
        text=guidance,
        token_sequence=guidance.replace('。', ' 。').split(),
        template_id=f'external_lookup_required_{topic}',
        grammar_violations=[],
        slot_coverage=0.88,
        semantic_score=0.92,
        final_score=0.0,
        selection_metadata={'external_lookup_required': True, 'external_topic': topic},
    )


def build_long_context_candidate_text(raw_text: str, focus: InputFocusDecision) -> str:
    text = str(raw_text or '')
    if any(marker in text for marker in ('何から', 'どこから', 'まず何', '優先')):
        return 'やることが多いので、まず時間や締切が決まっているものから一つずつ片付けるのがよさそうです。'
    if any(marker in text for marker in ('どうしたら', 'どうすれば', 'どう考える')):
        return '話題が多いので、一番気になっている点を一つに絞ると整理しやすいです。'
    focused = _segment_summary(focus.focused_text, max_len=18)
    return f'話題がいくつかあるので、まず{focused}から順番に整理すると考えやすいです。'


def build_clarify_candidate_text(focus: InputFocusDecision) -> str:
    question_segments = [seg for seg in focus.segments if seg.question_like]
    if len(question_segments) >= 2:
        left = _segment_summary(question_segments[0].text)
        right = _segment_summary(question_segments[1].text)
        return f"{left}のことと、{right}のことが混ざっているので、どちらから答えればいいですか？"
    return '話題がいくつかあるので、先にどの部分から答えればいいですか？'


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LSLM v3 minimal chat runner")
    parser.add_argument("--lexicon", default=None, help="辞書ファイルパス (.json / .lsd / .lsdx)")
    parser.add_argument("--text", default="", help="入力テキスト")
    parser.add_argument("--words", nargs="*", default=None, help="すでに分かち書き済みの入力トークン列")
    parser.add_argument("--trace-dir", default=None, help="trace JSONL の保存先ディレクトリ")
    parser.add_argument("--policy-memory", default=None, help="学習済み方針メモリ JSON")
    parser.add_argument("--history-path", default=None, help="chat 履歴 JSON の保存先")
    parser.add_argument("--session-id", default="", help="chat セッションID。未指定時は自動採番")
    parser.add_argument("--no-policy-memory", action="store_true", help="学習済み方針メモリを使わない")
    parser.add_argument("--no-history", action="store_true", help="chat 履歴保持を使わない")
    parser.add_argument("--reset-history", action="store_true", help="起動時に chat 履歴をリセットする")
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
            ("history_path", ("paths", "chat_history"), "runtime/chat_history/latest_session.json"),
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


def build_decision_trace(
    *,
    scored_candidates: Sequence[RealizationCandidate],
    response,
    filled_slots,
    recall_result,
    reward: RewardBreakdown,
    evaluation: Optional[Sequence[object]] = None,
    selection_source: str = "scorer",
    selected_index: int = -1,
    selection_metadata: Optional[Dict[str, Any]] = None,
) -> DecisionTrace:
    evaluation = list(evaluation or [])
    selection_metadata = dict(selection_metadata or {})

    candidate_list = list(scored_candidates or [])
    if not candidate_list and getattr(response, "chosen_candidate", None) is not None:
        candidate_list = [response.chosen_candidate]
        if selected_index < 0:
            selected_index = 0

    if not candidate_list:
        return DecisionTrace(
            selection_source=selection_source,
            selected_index=-1,
            selected_text=str(getattr(response, "text", "") or ""),
            selected_template_id=str(getattr(getattr(response, "chosen_candidate", None), "template_id", "") or ""),
            selected_score=float(getattr(getattr(response, "score", None), "total", 0.0) or 0.0),
            selected_reward_total=float(getattr(reward, "total", 0.0) or 0.0),
            selected_reward_internal=float(getattr(getattr(reward, "internal", None), "total", 0.0) or 0.0),
            selected_reward_external=float(getattr(getattr(reward, "external", None), "total", 0.0) or 0.0),
            selection_reason_codes=["no_candidates"],
            metadata={
                **selection_metadata,
                "evaluation_count": len(evaluation),
            },
        )

    ranking = sorted(
        enumerate(candidate_list),
        key=lambda item: float(getattr(item[1], "final_score", 0.0) or 0.0),
        reverse=True,
    )
    rank_map = {original_index: rank for rank, (original_index, _) in enumerate(ranking, start=1)}
    best_score = float(getattr(ranking[0][1], "final_score", 0.0) or 0.0)

    matched_index = -1
    if 0 <= int(selected_index) < len(candidate_list):
        matched_index = int(selected_index)
    else:
        chosen_candidate = getattr(response, "chosen_candidate", None)
        chosen_text = str(getattr(chosen_candidate, "text", "") or getattr(response, "text", "") or "")
        chosen_template_id = str(getattr(chosen_candidate, "template_id", "") or "")
        for idx, candidate in enumerate(candidate_list):
            if chosen_template_id and str(getattr(candidate, "template_id", "") or "") == chosen_template_id:
                matched_index = idx
                break
            if chosen_text and str(getattr(candidate, "text", "") or "") == chosen_text:
                matched_index = idx
                break
        if matched_index < 0:
            matched_index = 0

    selected_candidate = candidate_list[matched_index]
    second_best_score = float(getattr(ranking[1][1], "final_score", 0.0) or 0.0) if len(ranking) >= 2 else best_score

    compared_candidates: List[DecisionCandidateTrace] = []
    for idx, candidate in enumerate(candidate_list):
        candidate_score = float(getattr(candidate, "final_score", 0.0) or 0.0)
        score_breakdown = getattr(candidate, "score_breakdown", None) or ScoreBreakdown(total=candidate_score)
        rejected_reason_codes = []
        if idx != matched_index:
            rejected_reason_codes.append("not_selected")
        grammar_violations = list(getattr(candidate, "grammar_violations", []) or [])
        rejected_reason_codes.extend(f"grammar:{item}" for item in grammar_violations)
        selection_meta = dict(getattr(candidate, "selection_metadata", {}) or {})
        compared_candidates.append(
            DecisionCandidateTrace(
                index=idx,
                rank=int(rank_map.get(idx, len(candidate_list))),
                text=str(getattr(candidate, "text", "") or ""),
                template_id=str(getattr(candidate, "template_id", "") or ""),
                final_score=candidate_score,
                score_breakdown=score_breakdown,
                internal_reward_estimate=float(getattr(candidate, "internal_reward_estimate", candidate_score) or 0.0),
                selected=idx == matched_index,
                rejected_reason_codes=rejected_reason_codes,
                score_gap_from_best=max(0.0, best_score - candidate_score),
                metadata=selection_meta,
            )
        )

    selected_slot_evidence: Dict[str, Dict[str, Any]] = {}
    for slot_name, slot_value in getattr(filled_slots, "values", {}).items():
        value_text = str(getattr(slot_value, "value", "") or "")
        in_selected_text = bool(value_text) and value_text in str(getattr(selected_candidate, "text", "") or "")
        selected_slot_evidence[str(slot_name)] = {
            "value": value_text,
            "confidence": float(getattr(slot_value, "confidence", 0.0) or 0.0),
            "source_candidate": str(getattr(slot_value, "source_candidate", "") or ""),
            "inferred": bool(getattr(slot_value, "inferred", False)),
            "note": str(getattr(slot_value, "note", "") or ""),
            "used_in_selected_text": in_selected_text,
        }

    relation_map: Dict[str, List[List[str]]] = {}
    for item in list(getattr(recall_result, "candidates", []) or []):
        word = str(getattr(item, "word", "") or "")
        path = list(getattr(item, "relation_path", []) or [])
        if not word or not path:
            continue
        relation_map.setdefault(word, []).append(path)

    selected_relation_paths: List[List[str]] = []
    seen_paths: set[tuple[str, ...]] = set()
    selected_text = str(getattr(selected_candidate, "text", "") or "")
    response_slot_values = [str(value) for value in dict(getattr(response, "used_slots", {}) or {}).values() if str(value).strip()]
    probe_words = set(response_slot_values)
    for slot_name, slot_value in getattr(filled_slots, "values", {}).items():
        probe_words.add(str(getattr(slot_value, "value", "") or ""))
    for word in probe_words:
        if not word:
            continue
        if word not in selected_text and word not in response_slot_values:
            continue
        for path in relation_map.get(word, []):
            path_key = tuple(path)
            if path_key in seen_paths:
                continue
            seen_paths.add(path_key)
            selected_relation_paths.append(path)

    reason_codes = list(getattr(getattr(response, "score", None), "reasons", []) or [])
    if selection_source:
        reason_codes.append(f"selection_source:{selection_source}")
    if matched_index == ranking[0][0]:
        reason_codes.append("selected_top_scoring_candidate")
    if matched_index != ranking[0][0]:
        reason_codes.append("selected_non_top_candidate")
    if getattr(response, "used_slots", None):
        reason_codes.append("selected_candidate_uses_slots")
    if selected_relation_paths:
        reason_codes.append("selected_candidate_has_relation_paths")
    deduped_reason_codes = list(dict.fromkeys(str(code) for code in reason_codes if str(code).strip()))

    return DecisionTrace(
        selection_source=selection_source,
        selected_index=matched_index,
        selected_text=str(getattr(selected_candidate, "text", "") or getattr(response, "text", "") or ""),
        selected_template_id=str(getattr(selected_candidate, "template_id", "") or ""),
        selected_score=float(getattr(selected_candidate, "final_score", 0.0) or 0.0),
        selected_reward_total=float(getattr(reward, "total", 0.0) or 0.0),
        selected_reward_internal=float(getattr(getattr(reward, "internal", None), "total", 0.0) or 0.0),
        selected_reward_external=float(getattr(getattr(reward, "external", None), "total", 0.0) or 0.0),
        margin_vs_second=max(0.0, best_score - second_best_score),
        compared_candidates=compared_candidates,
        selected_slot_evidence=selected_slot_evidence,
        selected_relation_paths=selected_relation_paths,
        selection_reason_codes=deduped_reason_codes,
        metadata={
            **selection_metadata,
            "candidate_count": len(candidate_list),
            "best_candidate_index": int(ranking[0][0]),
            "best_candidate_score": best_score,
            "evaluation_count": len(evaluation),
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

def _slot_value_for_state_update(filled_slots, slot_name: str):
    return filled_slots.values.get(slot_name)


def _state_update_guard(response, filled_slots) -> dict[str, object]:
    retention = float(getattr(getattr(response, 'score', None), 'input_retention', 0.0) or 0.0)
    total = float(getattr(getattr(response, 'score', None), 'total', 0.0) or 0.0)
    topic_value = _slot_value_for_state_update(filled_slots, 'topic')
    topic_note = str(topic_value.note or '') if topic_value is not None else ''
    topic_confidence = float(topic_value.confidence) if topic_value is not None else 0.0
    weak_topic = topic_note in {'topic_fallback_from_recall', 'topic_from_dialogue_state'}
    return {
        'retention': retention,
        'total': total,
        'allow_subject_object_update': retention >= 0.34 and total >= 0.45,
        'allow_topic_update': retention >= 0.34 and total >= 0.45 and not weak_topic and topic_confidence >= 0.52,
        'allow_predicate_topic_fallback': retention >= 0.52 and total >= 0.60,
        'topic_note': topic_note,
        'topic_confidence': topic_confidence,
        'weak_topic': weak_topic,
    }


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

    guard = _state_update_guard(response, filled_slots)

    if actor and bool(guard['allow_subject_object_update']):
        state_after.last_subject = actor
    if target and bool(guard['allow_subject_object_update']):
        state_after.last_object = target
    if topic and bool(guard['allow_topic_update']):
        state_after.current_topic = topic
    elif filled_slots.frame.predicate and bool(guard['allow_predicate_topic_fallback']):
        state_after.current_topic = filled_slots.frame.predicate

    state_after.variables["last_response_text"] = response.text
    state_after.variables["last_policy"] = response.policy
    state_after.variables["last_intent"] = response.intent
    state_after.variables["state_update_guard"] = guard
    return state_after


def _find_latest_entry_payload(records: Sequence[dict[str, object]], word: str) -> dict[str, object]:
    latest: dict[str, object] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        entry = record.get('entry')
        if not isinstance(entry, dict):
            continue
        entry_word = str(entry.get('word', record.get('surface', '')) or record.get('surface', '')).strip()
        if entry_word == str(word or '').strip():
            latest = dict(entry)
    return latest


def _build_dict_update_events_from_unknown_learning(
    *,
    source_turn_id: str,
    response_score_total: float,
    unknown_word_learning: dict[str, object],
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    pending_records = [item for item in list(unknown_word_learning.get('pending_records', []) or []) if isinstance(item, dict)]
    applied_words = {str(item).strip() for item in list(unknown_word_learning.get('applied_words', []) or []) if str(item).strip()}
    for record in [item for item in list(unknown_word_learning.get('quarantined_records', []) or []) if isinstance(item, dict)]:
        entry = dict(record.get('entry', {}) or {})
        risk = dict(record.get('risk_assessment', {}) or {})
        entry_id = str(entry.get('word', record.get('surface', '')) or record.get('surface', '')).strip()
        if not entry_id:
            continue
        events.append(
            build_dict_update_event(
                update_type='lexicon_candidate',
                entry_id=entry_id,
                source_turn_id=source_turn_id,
                reason='unknown_word_quarantined',
                pollution_risk=float(risk.get('risk_score', 1.0) or 1.0),
                status='quarantined',
                before={},
                after=entry,
                evaluation_score=response_score_total,
                metadata={
                    'surface': str(record.get('surface', '') or ''),
                    'risk_assessment': risk,
                    'next_action': str(record.get('next_action', '') or ''),
                },
            )
        )
    for word in sorted(applied_words):
        entry_payload = _find_latest_entry_payload(pending_records, word)
        risk = dict(entry_payload.get('meta', {}).get('risk_assessment', {}) or {}) if isinstance(entry_payload.get('meta', {}), dict) else {}
        if not entry_payload:
            entry_payload = {'word': word}
        events.append(
            build_dict_update_event(
                update_type='lexicon_overlay',
                entry_id=word,
                source_turn_id=source_turn_id,
                reason='unknown_word_promoted',
                pollution_risk=float(risk.get('risk_score', 0.0) or 0.0),
                status='applied',
                before={},
                after=entry_payload,
                evaluation_score=response_score_total,
                metadata={
                    'overlay_path': str(unknown_word_learning.get('overlay_path', '') or ''),
                },
            )
        )
    return events


def _attach_stage_metrics_to_actions(actions: Sequence[EpisodeAction], stage_metrics: Sequence[dict[str, object]]) -> None:
    metrics_by_stage = {str(item.get('stage', '')).strip(): dict(item) for item in stage_metrics if isinstance(item, dict)}
    for action in actions:
        metric = metrics_by_stage.get(str(action.stage))
        if not metric:
            continue
        action.metadata.setdefault('stage_metric', metric)


def run_pipeline(
    args: argparse.Namespace,
    lexicon: LexiconContainer,
    normalizer: SurfaceNormalizer,
    runtime_context: Optional[ChatRuntimeContext] = None,
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

    stage_metrics: list[dict[str, object]] = []

    input_focus_config = build_dataclass_config(InputFocusConfig, get_setting(settings, "pipeline", "input_focus", default={}))
    _stage_started = perf_counter()
    focus = choose_input_focus(raw_text=raw_text, normalizer=normalizer, config=input_focus_config)
    focused_raw_text = focus.focused_text or raw_text
    stage_metrics.append(
        build_stage_metric(
            stage="input_focus",
            elapsed_ms=(perf_counter() - _stage_started) * 1000.0,
            candidate_count=len(focus.segments),
            kept_count=1 if focused_raw_text else 0,
            dropped_count=max(0, len(focus.segments) - 1) if focus.used_segmentation else 0,
            converge_reason_codes=[focus.reason] if str(focus.reason or '').strip() else [],
            rule_ids=["input_focus_segmentation"] if focus.used_segmentation else ["input_focus_passthrough"],
            metadata={
                "used_segmentation": bool(focus.used_segmentation),
                "has_alternative_question": bool(focus.has_alternative_question),
                "question_like_segment_count": int(focus.question_like_segment_count),
            },
        )
    )
    if focus.used_segmentation:
        LOGGER.info("input.focus used_segmentation=%s reason=%s focused=%s segment_count=%s", focus.used_segmentation, focus.reason, focused_raw_text, len(focus.segments))

    _stage_started = perf_counter()
    tokenization_result = build_tokenization_result(raw_text=focused_raw_text, explicit_words=args.words, normalizer=normalizer)
    if focus.used_segmentation and len(focus.segments) >= 2:
        unknown_word_learning = {
            "enabled": bool(normalizer.unknown_word_learner),
            "examined_spans": [],
            "applied_words": [],
            "relearned_words": [],
            "quarantined_records": [],
            "skipped_spans": [],
            "pending_records": [],
            "retokenized": False,
            "skipped_due_to_long_text": True,
        }
    else:
        unknown_word_learning = normalizer.maybe_learn_unknown_words(raw_text=focused_raw_text, tokenization_result=tokenization_result)
        if unknown_word_learning.get("retokenized"):
            tokenization_result = build_tokenization_result(raw_text=focused_raw_text, explicit_words=args.words, normalizer=normalizer)
    stage_metrics.append(
        build_stage_metric(
            stage="tokenize",
            elapsed_ms=(perf_counter() - _stage_started) * 1000.0,
            candidate_count=len(tokenization_result.tokenization),
            kept_count=len(tokenization_result.normalized_tokens),
            dropped_count=len(tokenization_result.unknown_spans),
            expand_reason_codes=["retokenized_after_unknown_learning"] if unknown_word_learning.get("retokenized") else [],
            converge_reason_codes=["unknown_spans_detected"] if tokenization_result.unknown_spans else [],
            rule_ids=["surface_normalization"],
            dict_feature_ids=["surface_forms", "aliases"],
            metadata={
                "unknown_span_count": len(tokenization_result.unknown_spans),
                "applied_word_count": len(list(unknown_word_learning.get("applied_words", []) or [])),
                "quarantined_count": len(list(unknown_word_learning.get("quarantined_records", []) or [])),
                "relearned_count": len(list(unknown_word_learning.get("relearned_words", []) or [])),
            },
        )
    )
    tokens = list(tokenization_result.normalized_tokens)
    LOGGER.info("input raw_text=%s", raw_text)
    if focus.used_segmentation and focused_raw_text != raw_text:
        LOGGER.info("input focused_raw_text=%s", focused_raw_text)
    LOGGER.info("input tokens=%s", tokens)
    if tokenization_result.unknown_spans:
        LOGGER.info("input unknown_spans=%s", [item.surface for item in tokenization_result.unknown_spans])

    history_config = build_dataclass_config(ChatHistoryConfig, get_setting(settings, "pipeline", "chat_history", default={}))
    if runtime_context is None and history_config.enabled and not bool(getattr(args, 'no_history', False)):
        runtime_context = load_chat_runtime_context(
            session_id=str(getattr(args, 'session_id', '') or ''),
            history_path=str(getattr(args, 'history_path', '') or ''),
            history_enabled=True,
            max_turns=history_config.max_turns,
            recent_response_window=history_config.recent_response_window,
            reset=bool(getattr(args, 'reset_history', False)),
        )
    elif runtime_context is not None:
        runtime_context.history_enabled = runtime_context.history_enabled and history_config.enabled and not bool(getattr(args, 'no_history', False))
        runtime_context.max_turns = max(1, int(history_config.max_turns))
        runtime_context.recent_response_window = max(1, int(history_config.recent_response_window))
        if not runtime_context.session_id:
            runtime_context.session_id = str(getattr(args, 'session_id', '') or '').strip() or new_session_id()
        if not runtime_context.history_path:
            runtime_context.history_path = str(getattr(args, 'history_path', '') or '').strip()

    session_id = runtime_context.session_id if runtime_context is not None else (str(getattr(args, 'session_id', '') or '').strip() or new_session_id())
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
    dialogue_state = deepcopy(runtime_context.dialogue_state) if runtime_context is not None else DialogueState()
    dialogue_state.variables["chat_history"] = _history_summary(runtime_context)
    dialogue_state.variables["recent_user_texts"] = list(runtime_context.recent_user_texts()) if runtime_context is not None else []
    dialogue_state.variables["recent_response_texts"] = list(runtime_context.recent_response_texts()) if runtime_context is not None else []
    dialogue_state.variables["session_id"] = session_id
    dialogue_state.variables["input_focus"] = {
        "original_text": raw_text,
        "focused_text": focused_raw_text,
        "used_segmentation": focus.used_segmentation,
        "has_alternative_question": focus.has_alternative_question,
        "question_like_segment_count": focus.question_like_segment_count,
        "segments": [
            {
                "text": seg.text,
                "start": seg.start,
                "end": seg.end,
                "question_like": seg.question_like,
                "alternative_branch": seg.alternative_branch,
                "score": seg.score,
            }
            for seg in focus.segments
        ],
    }
    dialogue_state_before = deepcopy(dialogue_state)

    intent_planner_config = build_dataclass_config(IntentPlannerConfig, get_setting(settings, "pipeline", "intent_planner", default={}))
    recall_config = build_dataclass_config(SemanticRecallConfig, get_setting(settings, "pipeline", "semantic_recall", default={}))
    slot_filler_config = build_dataclass_config(SlotFillerConfig, get_setting(settings, "pipeline", "slot_filler", default={}))
    surface_config = build_dataclass_config(SurfaceRealizerConfig, get_setting(settings, "pipeline", "surface_realizer", default={}))
    scorer_config = build_dataclass_config(BasicScorerConfig, get_setting(settings, "pipeline", "basic_scorer", default={}))
    policy_memory_config = build_dataclass_config(PolicyMemoryConfig, get_setting(settings, "policy_memory", default={}))
    policy_memory_limit = max(1, int(get_setting(settings, "learning", "runtime", "policy_memory_limit", default=4)))

    _stage_started = perf_counter()
    intent_plan = plan_intent(input_state=input_state, dialogue_state=dialogue_state, config=intent_planner_config)
    stage_metrics.append(
        build_stage_metric(
            stage="intent",
            elapsed_ms=(perf_counter() - _stage_started) * 1000.0,
            candidate_count=6,
            kept_count=1,
            dropped_count=5,
            converge_reason_codes=[str(intent_plan.intent)],
            rule_ids=["intent_planner_rule_v1"],
            metadata={
                "confidence": float(intent_plan.confidence),
                "policy_hint": str(intent_plan.response_policy_hint),
                "required_slots": list(intent_plan.required_slots),
            },
        )
    )
    _stage_started = perf_counter()
    recall_result = recall_semantics(
        input_state=input_state,
        lexicon=lexicon,
        dialogue_state=dialogue_state,
        intent_plan=intent_plan,
        config=recall_config,
    )
    _recall_keep_count = min(8, len(recall_result.candidates))
    stage_metrics.append(
        build_stage_metric(
            stage="recall",
            elapsed_ms=(perf_counter() - _stage_started) * 1000.0,
            candidate_count=len(recall_result.candidates),
            kept_count=_recall_keep_count,
            dropped_count=max(0, len(recall_result.candidates) - _recall_keep_count),
            expand_reason_codes=["input_seed", "relation_expand", "axis_probe"],
            converge_reason_codes=["rank_top_k"],
            rule_ids=["semantic_recall_v1"],
            dict_feature_ids=["relations", "axis", "grammar"],
            metadata={
                "seed_count": len(recall_result.seeds),
                "seed_words": list(recall_result.seeds),
            },
        )
    )
    _stage_started = perf_counter()
    filled_slots = fill_slots(
        input_state=input_state,
        recall_result=recall_result,
        lexicon=lexicon,
        intent_plan=intent_plan,
        dialogue_state=dialogue_state,
        config=slot_filler_config,
    )
    topic_history_applied = _maybe_apply_topic_history_fallback(
        raw_text=raw_text,
        dialogue_state=dialogue_state,
        filled_slots=filled_slots,
    )
    _slot_count = len(filled_slots.values) + len(filled_slots.missing_required) + len(filled_slots.optional_unfilled)
    stage_metrics.append(
        build_stage_metric(
            stage="slot",
            elapsed_ms=(perf_counter() - _stage_started) * 1000.0,
            candidate_count=_slot_count,
            kept_count=len(filled_slots.values),
            dropped_count=len(filled_slots.missing_required) + len(filled_slots.optional_unfilled),
            converge_reason_codes=["slot_frame_resolution"],
            rule_ids=["slot_filler_v1"],
            dict_feature_ids=["slots", "grammar", "topic_history"],
            metadata={
                "predicate": str(filled_slots.frame.predicate or ''),
                "predicate_type": str(filled_slots.frame.predicate_type or ''),
                "topic_history_applied": bool(topic_history_applied),
            },
        )
    )
    _stage_started = perf_counter()
    surface_plan, candidates = realize_surface(
        filled_slots=filled_slots,
        intent_plan=intent_plan,
        lexicon=lexicon,
        config=surface_config,
    )
    stage_metrics.append(
        build_stage_metric(
            stage="surface",
            elapsed_ms=(perf_counter() - _stage_started) * 1000.0,
            candidate_count=len(candidates),
            kept_count=min(1, len(candidates)),
            dropped_count=max(0, len(candidates) - 1),
            expand_reason_codes=["template_variant_generation"],
            converge_reason_codes=["candidate_scoring_pending"],
            rule_ids=["surface_realizer_v1"],
            dict_feature_ids=["surface_forms", "style_tags"],
            metadata={
                "template_id": str(surface_plan.template_id or ''),
                "sentence_count": int(surface_plan.sentence_count),
            },
        )
    )

    if focus.has_alternative_question:
        clarify_text = build_clarify_candidate_text(focus)
        candidates = [
            RealizationCandidate(
                text=clarify_text,
                token_sequence=clarify_text.replace("。", "").split(),
                template_id="clarify_alternative_input_v1",
                grammar_violations=[],
                slot_coverage=1.0 if filled_slots.values else 0.6,
                semantic_score=0.86,
                final_score=0.0,
            )
        ] + list(candidates)
    elif focus.used_segmentation and len(focus.segments) >= 2:
        long_text = build_long_context_candidate_text(raw_text, focus)
        candidates = [
            RealizationCandidate(
                text=long_text,
                token_sequence=long_text.replace("。", "").split(),
                template_id="long_context_guidance_v1",
                grammar_violations=[],
                slot_coverage=0.92 if filled_slots.values else 0.68,
                semantic_score=0.82,
                final_score=0.0,
            )
        ] + list(candidates)

    external_topic = detect_external_knowledge_topic(raw_text=focused_raw_text, tokens=tokens)
    external_lookup_candidate = build_external_lookup_guidance_candidate(raw_text=focused_raw_text, topic=external_topic)
    if external_lookup_candidate is not None:
        LOGGER.info('external_lookup.required topic=%s raw_text=%s', external_topic, focused_raw_text)
        candidates = [external_lookup_candidate] + list(candidates)

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

    _stage_started = perf_counter()
    response, scored_candidates = choose_best_response(
        input_state=input_state,
        intent_plan=intent_plan,
        filled_slots=filled_slots,
        candidates=candidates,
        config=scorer_config,
        recent_texts=runtime_context.recent_response_texts() if runtime_context is not None else None,
    )
    stage_metrics.append(
        build_stage_metric(
            stage="scoring",
            elapsed_ms=(perf_counter() - _stage_started) * 1000.0,
            candidate_count=len(scored_candidates),
            kept_count=1 if scored_candidates else 0,
            dropped_count=max(0, len(scored_candidates) - 1),
            converge_reason_codes=["best_total_score"],
            rule_ids=["basic_scorer_v1"],
            dict_feature_ids=["score_breakdown", "policy_memory"],
            metadata={
                "response_total": float(response.score.total),
                "response_policy": str(response.policy),
            },
        )
    )

    if focus.used_segmentation and len(focus.segments) >= 2 and not focus.has_alternative_question:
        if response.text.startswith(("確認できる範囲",)) or any(marker in response.text for marker in ("と考えられます", "について", "自然です")):
            override_text = build_long_context_candidate_text(raw_text, focus)
            override_candidate = RealizationCandidate(
                text=override_text,
                token_sequence=override_text.replace("。", "").split(),
                template_id="long_context_guidance_override_v1",
                grammar_violations=[],
                slot_coverage=0.9 if filled_slots.values else 0.7,
                semantic_score=0.86,
                final_score=max(0.78, response.score.total),
            )
            response.text = override_text
            response.chosen_candidate = override_candidate
            response.policy = "answer"
            response.score.semantic_consistency = max(response.score.semantic_consistency, 0.78)
            response.score.slot_fitness = max(response.score.slot_fitness, 0.60)
            response.score.grammar_fitness = max(response.score.grammar_fitness, 0.92)
            response.score.input_retention = max(response.score.input_retention, 0.52)
            response.score.policy_fitness = max(response.score.policy_fitness, 0.88)
            response.score.total = max(response.score.total, 0.78)
            if "long_context_override" not in response.score.reasons:
                response.score.reasons.append("long_context_override")
            scored_candidates = [override_candidate] + list(scored_candidates)

    response_accumulation_config = build_dataclass_config(ResponseAccumulationConfig, get_setting(settings, "pipeline", "response_accumulation", default={}))
    accumulated_text, accumulation_debug = _build_accumulated_response(
        base_text=response.text,
        input_state=input_state,
        intent_plan=intent_plan,
        filled_slots=filled_slots,
        dialogue_state=dialogue_state_before,
        runtime_context=runtime_context,
        config=response_accumulation_config,
        lexicon=lexicon,
    )
    if accumulation_debug.get('applied'):
        chosen_candidate = response.chosen_candidate or RealizationCandidate(
            text=response.text,
            token_sequence=response.text.replace('。', ' 。').split(),
            template_id='accumulation_base',
            grammar_violations=[],
            slot_coverage=1.0 if filled_slots.values else 0.0,
            semantic_score=max(0.40, float(response.score.semantic_consistency)),
            final_score=float(response.score.total),
        )
        accumulated_candidate = RealizationCandidate(
            text=accumulated_text,
            token_sequence=accumulated_text.replace('。', ' 。').replace('？', ' ？').split(),
            template_id=f"{chosen_candidate.template_id}_accumulated",
            grammar_violations=list(chosen_candidate.grammar_violations),
            slot_coverage=chosen_candidate.slot_coverage,
            semantic_score=min(1.0, chosen_candidate.semantic_score + 0.04),
            final_score=chosen_candidate.final_score,
        )
        response.text = accumulated_text
        response.chosen_candidate = accumulated_candidate
        if 'response_accumulated' not in response.score.reasons:
            response.score.reasons.append('response_accumulated')
        scored_candidates = [accumulated_candidate] + list(scored_candidates)

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
    _attach_stage_metrics_to_actions(actions, stage_metrics)
    audit_summary = build_turn_audit_summary(
        input_text=raw_text,
        response_text=response.text,
        stage_metrics=stage_metrics,
        actions=actions,
        scored_candidates=scored_candidates,
        missing_required=filled_slots.missing_required,
        unknown_word_learning=unknown_word_learning,
        reward_total=reward.total,
        reward_internal=reward.internal.total,
        reward_external=reward.external.total,
    )
    dict_update_events = _build_dict_update_events_from_unknown_learning(
        source_turn_id=turn_id,
        response_score_total=float(response.score.total),
        unknown_word_learning=unknown_word_learning,
    )
    dialogue_state_after = build_dialogue_state_after(
        dialogue_state=dialogue_state_before,
        intent_plan=intent_plan,
        filled_slots=filled_slots,
        response=response,
    )
    dialogue_state_after.variables['chat_history'] = _history_summary(runtime_context)
    dialogue_state_after.variables['recent_user_texts'] = list(runtime_context.recent_user_texts()) if runtime_context is not None else []
    dialogue_state_after.variables['recent_response_texts'] = list(runtime_context.recent_response_texts()) if runtime_context is not None else []
    dialogue_state_after.variables['session_id'] = session_id

    history_topic = response.used_slots.get('topic') or (filled_slots.values['topic'].value if 'topic' in filled_slots.values else '') or dialogue_state_after.current_topic
    history_saved_path: Optional[Path] = None
    if runtime_context is not None and runtime_context.history_enabled:
        _append_history_item(
            runtime_context,
            role='user',
            text=raw_text,
            turn_id=turn_id,
            intent=intent_plan.intent,
            topic=history_topic,
            policy=intent_plan.response_policy_hint,
        )
        _append_history_item(
            runtime_context,
            role='assistant',
            text=response.text,
            turn_id=turn_id,
            intent=response.intent,
            topic=str(history_topic or ''),
            policy=response.policy,
        )
        dialogue_state_after.variables['chat_history'] = _history_summary(runtime_context)
        dialogue_state_after.variables['recent_user_texts'] = list(runtime_context.recent_user_texts())
        dialogue_state_after.variables['recent_response_texts'] = list(runtime_context.recent_response_texts())
        runtime_context.dialogue_state = deepcopy(dialogue_state_after)
        history_saved_path = save_chat_runtime_context(runtime_context)

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
            "raw_text": focused_raw_text,
            "original_raw_text": raw_text,
            "tokens": tokens,
            "tokenization": [dataclass_to_dict(item) for item in tokenization_result.tokenization],
            "unknown_spans": [dataclass_to_dict(item) for item in tokenization_result.unknown_spans],
            "unknown_word_learning": unknown_word_learning,
            "stage_metrics": stage_metrics,
            "audit_summary": audit_summary,
            "dict_update_events": dict_update_events,
            "input_focus": {
                "reason": focus.reason,
                "used_segmentation": focus.used_segmentation,
                "focused_text": focused_raw_text,
                "has_alternative_question": focus.has_alternative_question,
                "question_like_segment_count": focus.question_like_segment_count,
                "segments": [
                    {
                        "text": seg.text,
                        "question_like": seg.question_like,
                        "alternative_branch": seg.alternative_branch,
                        "score": seg.score,
                    }
                    for seg in focus.segments
                ],
            },
            "response_score": dataclass_to_dict(response.score),
            "reward": dataclass_to_dict(reward),
            "slot_trace": dataclass_to_dict(slot_trace),
            "policy_memory_matches": policy_memory_matches,
            "topic_history_applied": topic_history_applied,
            "response_accumulation": accumulation_debug,
            "history": {
                "enabled": bool(runtime_context.history_enabled) if runtime_context is not None else False,
                "turn_count": len(runtime_context.history) if runtime_context is not None else 0,
                "recent_user_texts": list(runtime_context.recent_user_texts()) if runtime_context is not None else [],
                "recent_response_texts": list(runtime_context.recent_response_texts()) if runtime_context is not None else [],
                "history_path": str(history_saved_path) if history_saved_path else str(getattr(runtime_context, 'history_path', '') or ''),
            },
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
        trace_logger.append_turn_audit_summary(
            session_id=session_id,
            turn_id=turn_id,
            episode_id=episode_id,
            summary=audit_summary,
        )
        for event in dict_update_events:
            trace_logger.append_dict_update(
                session_id=session_id,
                turn_id=turn_id,
                episode_id=episode_id,
                dict_update=event,
            )
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