from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

_RISK_FRAGMENT_SURFACE_RE = re.compile(r'^(?:[ぁ-んー]{1,2}|[ぁ-んー]{1,4}(?:んだ|かな|よね|っけ)|〇〇.*)$')
_RISK_PUNCT_RE = re.compile(r'[\s、。？！!?,，．・「」『』（）()\[\]{}]+')
_GENERIC_LOW_VALUE_TEXTS = {
    '確認できる範囲ではそのように見えます。',
    '何かについては面白いことが考えられます。',
    '予定については多いことが考えられます。',
    'それについては考えられます。',
}
_LOW_VALUE_SLOT_WORDS = {
    'この', 'その', 'あの', 'それ', 'これ', 'あれ', 'ここ', 'そこ', 'あそこ',
    '何か', '何も', 'もの', 'こと', '一番', '色々', '〇〇', 'どれ', 'どんな',
}
_LOW_VALUE_SLOT_ENDINGS = ('は', 'が', 'を', 'に', 'で', 'の', 'も', 'って', 'んだ', 'かな', 'よね', 'っけ')
_ANCHOR_SLOT_KEYS = ('topic', 'predicate', 'target', 'state')
_NON_PROPER_KATAKANA = {'リフレッシュ', 'メニュー', 'ランチ', 'レシピ', 'チケット', 'クリア', 'イタリアン'}


@dataclass(slots=True)
class RiskAssessment:
    status: str
    risk_score: float
    reasons: List[str] = field(default_factory=list)
    suggested_action: str = ''
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def _normalize_text(text: str) -> str:
    value = str(text or '').strip()
    if not value:
        return ''
    return _RISK_PUNCT_RE.sub('', value).strip().lower()


def _is_low_value_slot_value(value: str) -> bool:
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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def assess_policy_memory_record(
    *,
    intent: str,
    slots: Mapping[str, str],
    text: str,
    source: str,
    reward_total: float,
    external_score: float,
    danger_threshold: float = 0.74,
    watch_threshold: float = 0.45,
) -> RiskAssessment:
    risk = 0.0
    reasons: List[str] = []
    normalized_text = _normalize_text(text)
    stable_slots = {str(k): str(v).strip() for k, v in dict(slots or {}).items() if str(v).strip()}
    anchor_slots = {k: v for k, v in stable_slots.items() if k in _ANCHOR_SLOT_KEYS and not _is_low_value_slot_value(v)}
    low_value_slots = [k for k, v in stable_slots.items() if _is_low_value_slot_value(v)]
    anchor_mentions = sum(1 for value in anchor_slots.values() if _normalize_text(value) and _normalize_text(value) in normalized_text)

    if not stable_slots:
        risk += 0.85
        reasons.append('no_slots')
    if normalized_text in {_normalize_text(item) for item in _GENERIC_LOW_VALUE_TEXTS}:
        risk += 0.55
        reasons.append('generic_low_value_text')
    if len(normalized_text) <= 4:
        risk += 0.20
        reasons.append('too_short_text')
    if low_value_slots:
        ratio = len(low_value_slots) / float(max(1, len(stable_slots)))
        risk += 0.32 * ratio
        reasons.append('contains_low_value_slots')
    if not anchor_slots:
        risk += 0.32
        reasons.append('missing_anchor_slots')
    if anchor_slots and anchor_mentions == 0:
        risk += 0.18
        reasons.append('anchor_not_grounded_in_text')
    if float(reward_total) < 0.46:
        risk += 0.12
        reasons.append('low_reward_total')
    if source == 'selected_response' and float(external_score) < 0.50:
        risk += 0.20
        reasons.append('selected_response_low_external')
    if source == 'teacher_target' and float(external_score) < 0.20:
        risk += 0.12
        reasons.append('teacher_target_low_external')

    risk = _clamp01(risk)
    if risk >= float(danger_threshold):
        status = 'danger'
        suggested_action = 'quarantine_and_rebuild'
    elif risk >= float(watch_threshold):
        status = 'watch'
        suggested_action = 'quarantine_and_review'
    else:
        status = 'safe'
        suggested_action = 'accept'

    return RiskAssessment(
        status=status,
        risk_score=risk,
        reasons=reasons,
        suggested_action=suggested_action,
        evidence={
            'intent': str(intent or 'unknown'),
            'slot_count': len(stable_slots),
            'anchor_slots': sorted(anchor_slots.keys()),
            'low_value_slots': low_value_slots,
            'anchor_mentions': anchor_mentions,
        },
    )


def assess_dict_entry(
    *,
    word: str,
    entry_payload: Mapping[str, Any],
    surface: str = '',
    occurrence_count: int = 1,
    min_overlay_confidence: float = 0.78,
    danger_threshold: float = 0.74,
    watch_threshold: float = 0.45,
) -> RiskAssessment:
    text = str(word or surface or '').strip()
    risk = 0.0
    reasons: List[str] = []
    grammar = dict(entry_payload.get('grammar', {}) or {})
    meta = dict(entry_payload.get('meta', {}) or {})
    surface_forms = [item for item in list(entry_payload.get('surface_forms', []) or []) if isinstance(item, Mapping)]
    confidence = float(meta.get('confidence', 1.0) or 1.0)
    generated_by = str(meta.get('generated_by', '') or '').strip()
    named_entity_type = str(grammar.get('named_entity_type', '') or meta.get('named_entity_type_hint', '') or '').strip()
    proper_noun = bool(grammar.get('proper_noun', False))
    pos = str(grammar.get('pos', 'unknown') or 'unknown').strip()
    source_surface = str(meta.get('source_surface', surface or text) or surface or text).strip()

    if not text:
        risk += 0.95
        reasons.append('empty_word')
    if _RISK_FRAGMENT_SURFACE_RE.match(text):
        risk += 0.70
        reasons.append('fragment_surface')
    if len(text) <= 3 and any(text.endswith(ending) for ending in _LOW_VALUE_SLOT_ENDINGS):
        risk += 0.25
        reasons.append('particle_like_surface')
    if pos == 'unknown':
        risk += 0.35
        reasons.append('unknown_pos')
    if bool(grammar.get('function_word', False)):
        risk += 0.45
        reasons.append('function_word_entry')
    if confidence < float(min_overlay_confidence):
        risk += 0.28
        reasons.append('low_confidence')
    if generated_by == 'fallback_unknown_word':
        risk += 0.18
        reasons.append('fallback_generated_entry')
    if source_surface and text != source_surface and source_surface not in {str(item.get('surface', '')).strip() for item in surface_forms}:
        risk += 0.14
        reasons.append('word_surface_mismatch')
    multi_token = any(len([str(tok).strip() for tok in list(item.get('tokens', []) or []) if str(tok).strip()]) > 1 for item in surface_forms)
    if multi_token and not proper_noun:
        risk += 0.20
        reasons.append('multi_token_surface')
    if proper_noun and not named_entity_type and text in _NON_PROPER_KATAKANA:
        risk += 0.35
        reasons.append('weak_proper_noun_evidence')
    if str(entry_payload.get('category', '') or '').strip() == 'surface' and pos == 'unknown':
        risk += 0.12
        reasons.append('surface_category_unknown_pos')
    if occurrence_count <= 1:
        risk += 0.08
        reasons.append('single_observation')
    else:
        risk -= min(0.18, 0.05 * float(max(0, occurrence_count - 1)))

    risk = _clamp01(risk)
    if risk >= float(danger_threshold):
        status = 'danger'
        suggested_action = 'quarantine_and_relearn'
    elif risk >= float(watch_threshold):
        status = 'watch'
        suggested_action = 'relearn_then_promote_if_safe'
    else:
        status = 'safe'
        suggested_action = 'promote'

    return RiskAssessment(
        status=status,
        risk_score=risk,
        reasons=reasons,
        suggested_action=suggested_action,
        evidence={
            'word': text,
            'surface': source_surface,
            'pos': pos,
            'proper_noun': proper_noun,
            'named_entity_type': named_entity_type,
            'generated_by': generated_by,
            'occurrence_count': int(max(1, occurrence_count)),
            'surface_form_count': len(surface_forms),
            'multi_token_surface': multi_token,
            'confidence': confidence,
        },
    )
