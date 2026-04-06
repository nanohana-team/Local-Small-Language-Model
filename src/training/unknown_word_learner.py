from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from src.core.schema import GrammarConstraints, LexiconContainer, LexiconEntry, SurfaceForm, UnknownSpan
from src.training.llm_gateway import LLMGateway

LOGGER = logging.getLogger(__name__)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


@dataclass(slots=True)
class UnknownWordLearnerConfig:
    enabled: bool = True
    min_span_length: int = 2
    max_spans_per_turn: int = 2
    promote_threshold: int = 1
    pending_path: str = 'runtime/unknown_word_candidates.jsonl'
    overlay_path: str = 'runtime/lexicon_overlay.json'
    temperature: float = 0.1
    max_output_tokens: int = 220
    preferred_models: List[str] = field(default_factory=list)
    request_timeout_note: str = ''


@dataclass(slots=True)
class UnknownWordLearningResult:
    examined_spans: List[str] = field(default_factory=list)
    applied_words: List[str] = field(default_factory=list)
    skipped_spans: List[str] = field(default_factory=list)
    pending_records: List[Dict[str, Any]] = field(default_factory=list)
    overlay_path: str = ''
    pending_path: str = ''


class UnknownWordLearner:
    def __init__(
        self,
        *,
        lexicon: LexiconContainer,
        config: UnknownWordLearnerConfig,
        llm_gateway: Optional[LLMGateway] = None,
    ) -> None:
        self.lexicon = lexicon
        self.config = config
        self.llm_gateway = llm_gateway or LLMGateway()
        self._overlay_cache: Optional[Dict[str, Any]] = None

    def apply_existing_overlay(self) -> List[str]:
        overlay = self._load_overlay()
        applied: List[str] = []
        for word, entry in dict(overlay.get('entries', {})).items():
            if self._apply_entry_to_lexicon(word, entry):
                applied.append(word)
        if applied:
            LOGGER.info('unknown_word.overlay_applied count=%s path=%s', len(applied), self.config.overlay_path)
        return applied

    def learn_unknown_spans(
        self,
        *,
        raw_text: str,
        unknown_spans: Sequence[UnknownSpan],
    ) -> UnknownWordLearningResult:
        result = UnknownWordLearningResult(
            overlay_path=self.config.overlay_path,
            pending_path=self.config.pending_path,
        )
        if not self.config.enabled:
            result.skipped_spans.extend(span.surface for span in unknown_spans)
            return result

        candidates = [
            span for span in unknown_spans
            if len(str(span.surface).strip()) >= max(1, int(self.config.min_span_length))
        ]
        candidates = candidates[: max(1, int(self.config.max_spans_per_turn))]
        if not candidates:
            return result

        for span in candidates:
            surface = str(span.surface).strip()
            if not surface:
                continue
            result.examined_spans.append(surface)
            if surface in self.lexicon.entries:
                result.skipped_spans.append(surface)
                continue

            occurrence_count = self._count_pending_occurrences(surface) + 1
            entry_payload = self._request_entry_payload(surface=surface, raw_text=raw_text, span=span)
            pending_record = {
                'surface': surface,
                'raw_text': raw_text,
                'start': int(span.start),
                'end': int(span.end),
                'reason': span.reason,
                'pos_hint': span.pos_hint,
                'proper_noun_candidate': bool(span.proper_noun_candidate),
                'named_entity_type_hint': span.named_entity_type_hint,
                'occurrence_count': occurrence_count,
                'entry': entry_payload,
            }
            self._append_pending_record(pending_record)
            result.pending_records.append(pending_record)

            if occurrence_count >= max(1, int(self.config.promote_threshold)):
                word = str(entry_payload.get('word', surface)).strip() or surface
                self._save_overlay_entry(word, entry_payload)
                if self._apply_entry_to_lexicon(word, entry_payload):
                    result.applied_words.append(word)
                    LOGGER.info(
                        'unknown_word.promoted surface=%s word=%s occurrences=%s overlay=%s',
                        surface,
                        word,
                        occurrence_count,
                        self.config.overlay_path,
                    )
        return result

    def _request_entry_payload(self, *, surface: str, raw_text: str, span: UnknownSpan) -> Dict[str, Any]:
        system_prompt = (
            'You are generating a minimal Japanese lexicon entry for an unknown token in LSLM v3. '
            'Return JSON only. Keep the schema minimal and conservative. '
            'Do not hallucinate rich semantics. When the token is a proper noun, prefer category="proper_noun". '
            'A proper noun must set grammar.proper_noun=true and grammar.named_entity_type to a stable value such as person/place/organization/product/service/work/event/other.'
        )
        user_prompt = (
            'Create one lexicon entry candidate for the unknown Japanese token below.\n'
            'Requirements:\n'
            '- JSON object only\n'
            '- required keys: word, category, aliases, surface_forms, grammar, style_tags, meta\n'
            '- allowed category examples: proper_noun, surface, noun, adverb, adjective, verb\n'
            '- grammar must contain at least pos, can_start, can_end, independent, content_word, function_word, proper_noun, named_entity_type\n'
            '- use category="proper_noun" when this token is likely a person/place/organization/product/service/work/event name\n'
            '- if category is proper_noun, set grammar.pos="noun", grammar.sub_pos="proper_noun", grammar.proper_noun=true\n'
            '- if not a proper noun, set grammar.proper_noun=false and grammar.named_entity_type=""\n'
            '- surface_forms must include the original surface\n'
            '- keep aliases small\n'
            '- if unsure, use category="surface" and grammar.pos="unknown"\n\n'
            f'surface: {surface}\n'
            f'context: {raw_text}\n'
            f'pos_hint: {span.pos_hint}\n'
            f'proper_noun_candidate: {span.proper_noun_candidate}\n'
            f'named_entity_type_hint: {span.named_entity_type_hint or ""}\n'
            'Return example:\n'
            '{"word":"VRChat","category":"proper_noun","aliases":[],"surface_forms":[{"form":"plain","surface":"VRChat","tokens":["VRChat"]}],'
            '"grammar":{"pos":"noun","sub_pos":"proper_noun","can_start":true,"can_end":false,"independent":true,"content_word":true,"function_word":false,"proper_noun":true,"named_entity_type":"service"},'
            '"style_tags":["daily"],"meta":{"generated_by":"llm_unknown_word","confidence":0.78}}'
        )
        try:
            response = self.llm_gateway.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                purpose='unknown_word_entry',
                preferred_models=self.config.preferred_models or None,
                temperature=float(self.config.temperature),
                max_output_tokens=max(1, int(self.config.max_output_tokens)),
            )
            parsed = self._parse_json_object(response.text)
            normalized = self._normalize_entry_payload(surface=surface, payload=parsed, span=span)
            normalized.setdefault('meta', {})
            normalized['meta'].update({
                'generated_by': 'llm_unknown_word',
                'llm_provider': response.provider,
                'llm_model': response.model,
            })
            return normalized
        except Exception as exc:
            LOGGER.warning('unknown_word.llm_failed surface=%s error=%s', surface, exc)
            return self._fallback_entry_payload(
                surface=surface,
                pos_hint=span.pos_hint,
                proper_noun_candidate=bool(span.proper_noun_candidate),
                named_entity_type_hint=span.named_entity_type_hint,
                reason='llm_failed',
            )

    def _fallback_entry_payload(
        self,
        *,
        surface: str,
        pos_hint: str,
        proper_noun_candidate: bool,
        named_entity_type_hint: str,
        reason: str,
    ) -> Dict[str, Any]:
        normalized_pos = str(pos_hint or 'unknown').strip() or 'unknown'
        proper_noun = bool(proper_noun_candidate or named_entity_type_hint)
        category = 'proper_noun' if proper_noun else 'surface'
        if not proper_noun and normalized_pos in {'adverb', 'noun', 'adjective', 'verb'}:
            category = normalized_pos
        grammar_pos = 'noun' if proper_noun else normalized_pos
        grammar_sub_pos = 'proper_noun' if proper_noun else ''
        named_entity_type = str(named_entity_type_hint or ('other' if proper_noun else '')).strip()
        return {
            'word': surface,
            'category': category,
            'aliases': [],
            'surface_forms': [
                {
                    'form': 'plain',
                    'surface': surface,
                    'tokens': [surface],
                }
            ],
            'grammar': {
                'pos': grammar_pos,
                'sub_pos': grammar_sub_pos,
                'can_start': True,
                'can_end': False,
                'independent': True,
                'content_word': True,
                'function_word': False,
                'proper_noun': proper_noun,
                'named_entity_type': named_entity_type,
            },
            'style_tags': [],
            'meta': {
                'generated_by': 'fallback_unknown_word',
                'reason': reason,
                'proper_noun_candidate': proper_noun_candidate,
                'named_entity_type_hint': named_entity_type_hint,
            },
        }

    def _normalize_entry_payload(self, *, surface: str, payload: Mapping[str, Any], span: UnknownSpan) -> Dict[str, Any]:
        word = str(payload.get('word', surface)).strip() or surface
        category = str(payload.get('category', 'surface')).strip() or 'surface'
        grammar = dict(payload.get('grammar', {}) or {})
        grammar.setdefault('pos', 'unknown')
        grammar.setdefault('can_start', True)
        grammar.setdefault('can_end', False)
        grammar.setdefault('independent', True)
        grammar.setdefault('content_word', True)
        grammar.setdefault('function_word', False)

        payload_entity_type = str(
            payload.get('named_entity_type', '')
            or grammar.get('named_entity_type', '')
            or span.named_entity_type_hint
            or ''
        ).strip()
        proper_noun = bool(
            payload.get('proper_noun', False)
            or grammar.get('proper_noun', False)
            or category == 'proper_noun'
            or str(grammar.get('sub_pos', '')).strip() == 'proper_noun'
            or payload_entity_type
            or span.proper_noun_candidate
        )

        if proper_noun:
            category = 'proper_noun'
            grammar['pos'] = 'noun'
            grammar['sub_pos'] = 'proper_noun'
            grammar['proper_noun'] = True
            grammar['named_entity_type'] = payload_entity_type or 'other'
            grammar['content_word'] = True
            grammar['function_word'] = False
        else:
            grammar['proper_noun'] = False
            grammar['named_entity_type'] = ''

        surface_forms = list(payload.get('surface_forms', []) or [])
        if not surface_forms:
            surface_forms = [{'form': 'plain', 'surface': surface, 'tokens': [word]}]
        normalized_surface_forms: List[Dict[str, Any]] = []
        for item in surface_forms:
            if not isinstance(item, Mapping):
                continue
            sf_surface = str(item.get('surface', surface)).strip() or surface
            tokens = [str(x).strip() for x in item.get('tokens', [word]) if str(x).strip()]
            if not tokens:
                tokens = [word]
            normalized_surface_forms.append(
                {
                    'form': str(item.get('form', 'plain')),
                    'surface': sf_surface,
                    'tokens': tokens,
                }
            )
        if not normalized_surface_forms:
            normalized_surface_forms.append({'form': 'plain', 'surface': surface, 'tokens': [word]})

        aliases = [str(x).strip() for x in payload.get('aliases', []) if str(x).strip()]
        style_tags = [str(x).strip() for x in payload.get('style_tags', []) if str(x).strip()]
        meta = dict(payload.get('meta', {}) or {})
        meta.setdefault('source_surface', surface)
        meta.setdefault('generated_by', 'llm_unknown_word')
        meta.setdefault('proper_noun_candidate', bool(span.proper_noun_candidate))
        if span.named_entity_type_hint:
            meta.setdefault('named_entity_type_hint', span.named_entity_type_hint)

        return {
            'word': word,
            'category': category,
            'aliases': aliases,
            'surface_forms': normalized_surface_forms,
            'grammar': grammar,
            'style_tags': style_tags,
            'meta': meta,
        }

    def _apply_entry_to_lexicon(self, word: str, entry_payload: Mapping[str, Any]) -> bool:
        normalized_word = str(word).strip()
        if not normalized_word:
            return False
        entry = LexiconEntry.from_dict(dict(entry_payload))
        self.lexicon.entries[normalized_word] = entry
        self._update_indexes(entry)
        return True

    def _update_indexes(self, entry: LexiconEntry) -> None:
        pos = str(entry.grammar.pos or 'unknown')
        bucket = self.lexicon.indexes.by_pos.setdefault(pos, [])
        if entry.word not in bucket:
            bucket.append(entry.word)
        if entry.grammar.can_start and entry.word not in self.lexicon.indexes.can_start:
            self.lexicon.indexes.can_start.append(entry.word)
        if entry.grammar.can_end and entry.word not in self.lexicon.indexes.can_end:
            self.lexicon.indexes.can_end.append(entry.word)
        if entry.grammar.content_word and entry.word not in self.lexicon.indexes.content_words:
            self.lexicon.indexes.content_words.append(entry.word)
        if entry.grammar.function_word and entry.word not in self.lexicon.indexes.function_words:
            self.lexicon.indexes.function_words.append(entry.word)
        if entry.hierarchy and entry.word not in self.lexicon.indexes.entry_path:
            self.lexicon.indexes.entry_path[entry.word] = list(entry.hierarchy)
        self.lexicon.meta.entry_count = max(int(self.lexicon.meta.entry_count or 0), len(self.lexicon.entries))

    def _append_pending_record(self, record: Mapping[str, Any]) -> None:
        path = Path(self.config.pending_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False, separators=(',', ':')))
            handle.write('\n')

    def _count_pending_occurrences(self, surface: str) -> int:
        path = Path(self.config.pending_path)
        if not path.exists():
            return 0
        count = 0
        try:
            with path.open('r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    if str(payload.get('surface', '')).strip() == surface:
                        count += 1
        except Exception as exc:
            LOGGER.warning('unknown_word.pending_count_failed path=%s error=%s', path, exc)
        return count

    def _load_overlay(self) -> Dict[str, Any]:
        if self._overlay_cache is not None:
            return self._overlay_cache
        path = Path(self.config.overlay_path)
        if not path.exists():
            self._overlay_cache = {'meta': {'version': 'unknown-overlay-v1'}, 'entries': {}}
            return self._overlay_cache
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            LOGGER.warning('unknown_word.overlay_load_failed path=%s error=%s', path, exc)
            payload = {'meta': {'version': 'unknown-overlay-v1'}, 'entries': {}}
        if not isinstance(payload, dict):
            payload = {'meta': {'version': 'unknown-overlay-v1'}, 'entries': {}}
        payload.setdefault('meta', {'version': 'unknown-overlay-v1'})
        payload.setdefault('entries', {})
        self._overlay_cache = payload
        return payload

    def _save_overlay_entry(self, word: str, entry_payload: Mapping[str, Any]) -> None:
        overlay = self._load_overlay()
        overlay_entries = dict(overlay.get('entries', {}))
        overlay_entries[str(word)] = dict(entry_payload)
        overlay['entries'] = overlay_entries
        path = Path(self.config.overlay_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(overlay, ensure_ascii=False, indent=2), encoding='utf-8')
        self._overlay_cache = overlay

    def _parse_json_object(self, text: str) -> Dict[str, Any]:
        text = str(text or '').strip()
        if not text:
            raise ValueError('empty LLM response')
        candidate = text
        match = _JSON_BLOCK_RE.search(text)
        if match:
            candidate = match.group(1)
        if not candidate.startswith('{'):
            start = candidate.find('{')
            end = candidate.rfind('}')
            if start >= 0 and end > start:
                candidate = candidate[start:end + 1]
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError('LLM response JSON must be an object')
        return parsed


def build_unknown_word_learner_config(data: Mapping[str, Any] | None = None) -> UnknownWordLearnerConfig:
    payload = dict(data or {})
    preferred = payload.get('preferred_models', [])
    if preferred is None:
        payload['preferred_models'] = []
    elif isinstance(preferred, (list, tuple)):
        payload['preferred_models'] = [str(item).strip() for item in preferred if str(item).strip()]
    else:
        payload['preferred_models'] = [str(preferred).strip()] if str(preferred).strip() else []
    return UnknownWordLearnerConfig(**{k: v for k, v in payload.items() if k in UnknownWordLearnerConfig.__dataclass_fields__})
