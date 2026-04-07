from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

from src.core.schema import FilledSlots, IntentPlan, RealizationCandidate
from src.training.contamination_guard import assess_policy_memory_record

LOGGER = logging.getLogger(__name__)
JST = ZoneInfo('Asia/Tokyo')

_SLOT_PRIORITY: Tuple[str, ...] = (
    'topic',
    'state',
    'predicate',
    'target',
    'actor',
    'location',
    'time',
    'cause',
    'recipient',
    'manner',
)

_PUNCT_RE = re.compile(r'[\s、。？！!?,，．・「」『』（）()\[\]{}]+')
_GENERIC_LOW_VALUE_TEXTS: Tuple[str, ...] = (
    '確認できる範囲ではそのように見えます。',
    '何かについては面白いことが考えられます。',
    '予定については多いことが考えられます。',
    'それについては考えられます。',
)
_ANCHOR_SLOT_KEYS: Tuple[str, ...] = ('topic', 'predicate', 'target', 'state')
_LOW_VALUE_SLOT_WORDS: Tuple[str, ...] = (
    'この', 'その', 'あの', 'それ', 'これ', 'あれ', 'ここ', 'そこ', 'あそこ',
    '何か', '何も', 'もの', 'こと', '一番', '色々', '〇〇', 'どれ', 'どんな',
)
_LOW_VALUE_SLOT_ENDINGS: Tuple[str, ...] = ('は', 'が', 'を', 'に', 'で', 'の', 'も', 'って', 'んだ', 'かな', 'よね', 'っけ')


@dataclass(slots=True)
class PolicyMemoryRecord:
    intent: str
    slots: Dict[str, str]
    text: str
    source: str = 'teacher_target'
    template_id: str = ''
    count: int = 1
    weight: float = 0.0
    last_reward_total: float = 0.0
    last_internal: float = 0.0
    last_external: float = 0.0
    created_at: str = ''
    updated_at: str = ''


@dataclass(slots=True)
class PolicyMemoryMatch:
    record: PolicyMemoryRecord
    match_score: float
    slot_hits: int
    slot_total: int
    exact_intent: bool
    text_slot_hits: int


@dataclass(slots=True)
class PolicyMemoryConfig:
    min_match_score: float = 0.55
    quarantine_path: str = 'runtime/policy_memory_quarantine.jsonl'
    contamination_watch_threshold: float = 0.45
    contamination_danger_threshold: float = 0.74
    max_suggestions: int = 3
    max_records_per_key: int = 12
    teacher_reward_scale: float = 1.0
    response_reward_scale: float = 0.6
    teacher_source_bonus: float = 0.06
    response_source_bonus: float = 0.02
    selected_response_hard_reject_external: float = 0.24
    selected_response_min_reward_total: float = 0.42
    selected_response_low_external_threshold: float = 0.50
    selected_response_low_external_scale: float = 0.10
    teacher_hard_reject_external: float = 0.05
    teacher_min_reward_total: float = 0.46
    min_suggest_weight: float = 0.12
    min_suggest_external: float = 0.28
    min_exact_slot_hits: int = 1
    min_anchor_slot_hits: int = 1
    require_anchor_slot_match: bool = True
    min_anchor_mentions_in_text: int = 1
    anchor_free_external_threshold: float = 0.82


class PolicyMemoryStore:
    def __init__(
        self,
        path: str | Path,
        *,
        config: PolicyMemoryConfig | None = None,
        autoload: bool = True,
    ) -> None:
        self.path = Path(path)
        self.config = config or PolicyMemoryConfig()
        self.records: List[PolicyMemoryRecord] = []
        self.quarantine_path = Path(self.config.quarantine_path)
        if autoload:
            self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.records = []
            return

        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except Exception:
            LOGGER.exception('policy_memory.load_failed path=%s', self.path)
            self.records = []
            return

        records: List[PolicyMemoryRecord] = []
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            text = str(item.get('text', '') or '').strip()
            if not text:
                continue
            sanitized_slots = self._stable_slots(dict(item.get('slots', {}) or {}))
            record = PolicyMemoryRecord(
                intent=str(item.get('intent', 'unknown') or 'unknown'),
                slots=sanitized_slots,
                text=text,
                source=str(item.get('source', 'teacher_target') or 'teacher_target'),
                template_id=str(item.get('template_id', '') or ''),
                count=max(1, int(item.get('count', 1) or 1)),
                weight=max(0.0, min(1.0, float(item.get('weight', 0.0) or 0.0))),
                last_reward_total=max(0.0, min(1.0, float(item.get('last_reward_total', 0.0) or 0.0))),
                last_internal=max(0.0, min(1.0, float(item.get('last_internal', 0.0) or 0.0))),
                last_external=max(0.0, min(1.0, float(item.get('last_external', 0.0) or 0.0))),
                created_at=str(item.get('created_at', '') or ''),
                updated_at=str(item.get('updated_at', '') or ''),
            )
            if not record.slots:
                continue
            if self._is_generic_low_value_text(record.text):
                continue
            if not self._has_meaningful_anchor(record.slots):
                continue
            if self._count_anchor_mentions(record.text, record.slots) < int(self.config.min_anchor_mentions_in_text) and record.last_external < float(self.config.anchor_free_external_threshold):
                continue
            if record.weight < float(self.config.min_suggest_weight) and record.last_external < float(self.config.min_suggest_external):
                continue
            risk = assess_policy_memory_record(
                intent=record.intent,
                slots=record.slots,
                text=record.text,
                source=record.source,
                reward_total=record.last_reward_total,
                external_score=record.last_external,
                watch_threshold=float(self.config.contamination_watch_threshold),
                danger_threshold=float(self.config.contamination_danger_threshold),
            )
            if risk.status != 'safe':
                continue
            records.append(record)
        self.records = records

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(record) for record in self.records]
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        return self.path

    def remember(
        self,
        *,
        intent: str,
        slots: Mapping[str, str],
        text: str,
        reward_total: float,
        internal_score: float,
        external_score: float,
        source: str,
        template_id: str = '',
    ) -> bool:
        cleaned_text = self._normalize_text(text)
        if not cleaned_text:
            return False

        slot_signature = self._stable_slots(slots)
        if not slot_signature:
            LOGGER.info('policy_memory.remember_skipped source=%s reason=no_meaningful_slots text=%s', source, cleaned_text)
            return False
        reward_total = max(0.0, min(1.0, float(reward_total)))
        internal_score = max(0.0, min(1.0, float(internal_score)))
        external_score = max(0.0, min(1.0, float(external_score)))

        source_scale = self.config.teacher_reward_scale if source == 'teacher_target' else self.config.response_reward_scale
        blended_weight = max(0.0, min(1.0, reward_total * source_scale))

        anchor_mentions = self._count_anchor_mentions(cleaned_text, slot_signature)

        if source == 'teacher_target':
            if external_score <= float(self.config.teacher_hard_reject_external) and reward_total < float(self.config.teacher_min_reward_total):
                LOGGER.info(
                    'policy_memory.remember_skipped source=%s reason=teacher_low_quality reward_total=%.4f external=%.4f text=%s',
                    source,
                    reward_total,
                    external_score,
                    cleaned_text,
                )
                return False
            if anchor_mentions < int(self.config.min_anchor_mentions_in_text) and external_score < float(self.config.anchor_free_external_threshold):
                LOGGER.info(
                    'policy_memory.remember_skipped source=%s reason=no_anchor_mention external=%.4f text=%s slots=%s',
                    source,
                    external_score,
                    cleaned_text,
                    slot_signature,
                )
                return False

        if source == 'selected_response':
            if external_score <= float(self.config.selected_response_hard_reject_external):
                LOGGER.info(
                    'policy_memory.remember_skipped source=%s reason=hard_low_external external=%.4f text=%s',
                    source,
                    external_score,
                    cleaned_text,
                )
                return False
            if reward_total < float(self.config.selected_response_min_reward_total):
                LOGGER.info(
                    'policy_memory.remember_skipped source=%s reason=low_reward_total reward_total=%.4f text=%s',
                    source,
                    reward_total,
                    cleaned_text,
                )
                return False
            if external_score < float(self.config.selected_response_low_external_threshold):
                blended_weight *= max(0.0, min(1.0, float(self.config.selected_response_low_external_scale)))
            if anchor_mentions < int(self.config.min_anchor_mentions_in_text) and external_score < float(self.config.anchor_free_external_threshold):
                LOGGER.info(
                    'policy_memory.remember_skipped source=%s reason=no_anchor_mention external=%.4f text=%s slots=%s',
                    source,
                    external_score,
                    cleaned_text,
                    slot_signature,
                )
                return False

        risk = assess_policy_memory_record(
            intent=intent,
            slots=slot_signature,
            text=cleaned_text,
            source=source,
            reward_total=reward_total,
            external_score=external_score,
            watch_threshold=float(self.config.contamination_watch_threshold),
            danger_threshold=float(self.config.contamination_danger_threshold),
        )
        if risk.status != 'safe':
            self._append_quarantine_record(
                {
                    'intent': str(intent or 'unknown'),
                    'slots': dict(slot_signature),
                    'text': cleaned_text,
                    'source': source,
                    'template_id': template_id,
                    'reward_total': reward_total,
                    'internal_score': internal_score,
                    'external_score': external_score,
                    'risk_assessment': risk.to_dict(),
                }
            )
            LOGGER.info(
                'policy_memory.remember_skipped source=%s reason=contamination_risk status=%s risk=%.4f text=%s',
                source,
                risk.status,
                risk.risk_score,
                cleaned_text,
            )
            return False

        now = datetime.now(JST).isoformat(timespec='seconds')
        existing = self._find_record(intent=intent, slots=slot_signature, text=cleaned_text, source=source)
        if existing is None:
            self.records.append(
                PolicyMemoryRecord(
                    intent=str(intent or 'unknown'),
                    slots=slot_signature,
                    text=cleaned_text,
                    source=source,
                    template_id=template_id,
                    count=1,
                    weight=blended_weight,
                    last_reward_total=reward_total,
                    last_internal=internal_score,
                    last_external=external_score,
                    created_at=now,
                    updated_at=now,
                )
            )
        else:
            total_count = existing.count + 1
            existing.weight = ((existing.weight * existing.count) + blended_weight) / float(total_count)
            existing.count = total_count
            existing.last_reward_total = reward_total
            existing.last_internal = internal_score
            existing.last_external = external_score
            existing.updated_at = now
            if template_id:
                existing.template_id = template_id

        self._trim_records(intent=intent, slots=slot_signature)
        return True

    def suggest(
        self,
        *,
        intent_plan: IntentPlan,
        filled_slots: FilledSlots,
        existing_texts: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> tuple[List[RealizationCandidate], List[Dict[str, object]]]:
        if not self.records:
            return [], []

        current_slots = self._filled_slots_to_dict(filled_slots)
        if not current_slots:
            return [], []
        existing_normalized = {self._normalize_text(text) for text in (existing_texts or []) if self._normalize_text(text)}
        matches = self._rank_matches(
            intent=intent_plan.intent,
            slots=current_slots,
        )
        candidates: List[RealizationCandidate] = []
        debug: List[Dict[str, object]] = []
        max_items = max(1, int(limit or self.config.max_suggestions))

        for index, match in enumerate(matches[:max_items], start=1):
            text = self._normalize_text(match.record.text)
            if not text or text in existing_normalized:
                continue

            slot_coverage = self._estimate_slot_coverage(match, filled_slots)
            semantic_score = self._estimate_semantic_score(match)
            candidate = RealizationCandidate(
                text=text,
                token_sequence=self._simple_tokenize(text),
                template_id=f'policy_memory_{match.record.source}_{index}',
                grammar_violations=self._quick_grammar_checks(text),
                slot_coverage=slot_coverage,
                semantic_score=semantic_score,
                final_score=0.0,
            )
            candidates.append(candidate)
            existing_normalized.add(text)
            debug.append(
                {
                    'text': text,
                    'source': match.record.source,
                    'match_score': round(match.match_score, 6),
                    'slot_hits': match.slot_hits,
                    'slot_total': match.slot_total,
                    'text_slot_hits': match.text_slot_hits,
                    'weight': round(match.record.weight, 6),
                    'count': match.record.count,
                }
            )

        return candidates, debug

    def recent_texts(self, *, limit: int = 8, source: str = 'selected_response') -> List[str]:
        records = [record for record in self.records if record.source == source and record.text]
        records.sort(key=lambda item: (item.updated_at, item.created_at, item.count), reverse=True)
        results: List[str] = []
        for record in records:
            results.append(self._normalize_text(record.text))
            if len(results) >= max(1, int(limit)):
                break
        return results

    def _rank_matches(self, *, intent: str, slots: Mapping[str, str]) -> List[PolicyMemoryMatch]:
        matches: List[PolicyMemoryMatch] = []
        if not slots:
            return matches
        for record in self.records:
            if record.intent != intent:
                continue
            if record.weight < float(self.config.min_suggest_weight):
                continue
            if record.last_external < float(self.config.min_suggest_external):
                continue
            if self._is_generic_low_value_text(record.text):
                continue
            slot_total = len(record.slots)
            slot_hits = 0
            anchor_hits = 0
            for key, value in record.slots.items():
                if slots.get(key, '') == value:
                    slot_hits += 1
                    if key in _ANCHOR_SLOT_KEYS:
                        anchor_hits += 1

            text_slot_hits = sum(1 for value in slots.values() if value and value in record.text)
            exact_intent = record.intent == intent
            structural_score = (slot_hits / float(slot_total)) if slot_total > 0 else 0.0
            anchor_bonus = min(0.30, 0.15 * anchor_hits)
            text_bonus = min(0.18, 0.04 * text_slot_hits)
            confidence_bonus = min(0.18, 0.16 * record.weight)
            source_bonus = self.config.teacher_source_bonus if record.source == 'teacher_target' else self.config.response_source_bonus
            match_score = structural_score + anchor_bonus + text_bonus + confidence_bonus + source_bonus

            if slot_hits < int(self.config.min_exact_slot_hits):
                continue
            if self.config.require_anchor_slot_match and anchor_hits < int(self.config.min_anchor_slot_hits):
                continue
            if slot_total == 0 and text_slot_hits == 0:
                continue
            if match_score < self.config.min_match_score:
                continue
            matches.append(
                PolicyMemoryMatch(
                    record=record,
                    match_score=match_score,
                    slot_hits=slot_hits,
                    slot_total=slot_total,
                    exact_intent=exact_intent,
                    text_slot_hits=text_slot_hits,
                )
            )

        matches.sort(
            key=lambda item: (
                item.match_score,
                item.record.last_external,
                item.record.weight,
                item.record.last_reward_total,
                item.record.count,
            ),
            reverse=True,
        )
        return matches

    def _estimate_slot_coverage(self, match: PolicyMemoryMatch, filled_slots: FilledSlots) -> float:
        current_total = max(1, len([value for value in filled_slots.values.values() if value.value]))
        signature_ratio = (match.slot_hits / float(match.slot_total)) if match.slot_total > 0 else 0.0
        text_ratio = min(1.0, match.text_slot_hits / float(current_total))
        coverage = (signature_ratio * 0.55) + (text_ratio * 0.45)
        if match.record.source == 'teacher_target':
            coverage += 0.05
        return max(0.0, min(1.0, coverage))

    def _estimate_semantic_score(self, match: PolicyMemoryMatch) -> float:
        base = 0.52
        base += min(0.20, match.match_score * 0.18)
        base += min(0.18, match.record.weight * 0.22)
        base += min(0.08, match.record.last_external * 0.10)
        if match.record.source == 'teacher_target':
            base += 0.06
        elif match.record.source == 'selected_response':
            base += 0.02
        return max(0.0, min(1.0, base))

    def _is_generic_low_value_text(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        return any(normalized == self._normalize_text(item) for item in _GENERIC_LOW_VALUE_TEXTS)

    def _find_record(self, *, intent: str, slots: Mapping[str, str], text: str, source: str) -> PolicyMemoryRecord | None:
        for record in self.records:
            if record.intent != intent:
                continue
            if record.source != source:
                continue
            if record.slots != dict(slots):
                continue
            if self._normalize_text(record.text) == text:
                return record
        return None

    def _trim_records(self, *, intent: str, slots: Mapping[str, str]) -> None:
        grouped: List[PolicyMemoryRecord] = [
            record
            for record in self.records
            if record.intent == intent and record.slots == dict(slots)
        ]
        if len(grouped) <= self.config.max_records_per_key:
            return
        grouped.sort(
            key=lambda item: (item.weight, item.last_reward_total, item.count),
            reverse=True,
        )
        keep = set(id(item) for item in grouped[: self.config.max_records_per_key])
        self.records = [record for record in self.records if (record.intent != intent or record.slots != dict(slots) or id(record) in keep)]

    def _filled_slots_to_dict(self, filled_slots: FilledSlots) -> Dict[str, str]:
        return self._stable_slots(
            {
                str(name): str(value.value).strip()
                for name, value in filled_slots.values.items()
                if str(value.value).strip()
            }
        )

    def _has_meaningful_anchor(self, slots: Mapping[str, str]) -> bool:
        return any(str(slots.get(key, '')).strip() for key in _ANCHOR_SLOT_KEYS)

    def _count_anchor_mentions(self, text: str, slots: Mapping[str, str]) -> int:
        normalized_text = self._normalize_text(text)
        return sum(1 for key in _ANCHOR_SLOT_KEYS if slots.get(key) and str(slots[key]) in normalized_text)

    def _is_low_value_slot_value(self, value: str) -> bool:
        text = str(value or '').strip()
        if not text:
            return True
        if text in _LOW_VALUE_SLOT_WORDS:
            return True
        if text.startswith('〇〇'):
            return True
        if len(text) <= 2 and all('ぁ' <= ch <= 'ん' or ch == 'ー' for ch in text):
            return True
        if len(text) <= 3 and any(text.endswith(ending) for ending in _LOW_VALUE_SLOT_ENDINGS):
            return True
        return False

    def _stable_slots(self, slots: Mapping[str, str]) -> Dict[str, str]:
        stable: Dict[str, str] = {}
        raw = {str(k): str(v).strip() for k, v in dict(slots or {}).items() if str(v).strip()}
        for key in _SLOT_PRIORITY:
            value = raw.get(key, '')
            if value and not self._is_low_value_slot_value(value):
                stable[key] = value
        for key in sorted(raw):
            if key not in stable and not self._is_low_value_slot_value(raw[key]):
                stable[key] = raw[key]
        return stable

    def _normalize_text(self, text: str) -> str:
        value = str(text or '').strip()
        if not value:
            return ''
        if not value.endswith(('。', '？', '!', '！')):
            value += '。'
        return value

    def _simple_tokenize(self, text: str) -> List[str]:
        return [chunk for chunk in _PUNCT_RE.sub(' ', text).split() if chunk]

    def _append_quarantine_record(self, record: Mapping[str, object]) -> None:
        path = self.quarantine_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False, separators=(',', ':')))
            handle.write('\n')

    def _quick_grammar_checks(self, text: str) -> List[str]:
        violations: List[str] = []
        if '。。' in text:
            violations.append('double_period')
        if '、、' in text:
            violations.append('duplicated_punctuation')
        if 'です。です' in text:
            violations.append('duplicated_copula')
        return violations
