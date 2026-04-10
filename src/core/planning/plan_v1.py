from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


QUESTION_WORDS = (
    "なぜ",
    "どうして",
    "なんで",
    "どうやって",
    "何",
    "どこ",
    "いつ",
    "誰",
    "どれ",
    "どっち",
    "どう",
)
EMPATHY_HINTS = (
    "つらい",
    "しんどい",
    "不安",
    "怖い",
    "悲しい",
    "疲れた",
    "苦しい",
)
COMPARE_HINTS = ("比較", "違い", "どっち", "どちら", "比べ")
PROCEDURE_HINTS = ("手順", "やり方", "方法", "設定", "実装")
REASON_HINTS = ("なぜ", "どうして", "理由", "なんで")
GREETING_HINTS = (
    "おはよう",
    "こんにちは",
    "こんばんは",
    "やあ",
    "もしもし",
    "よろしく",
)
THANKS_HINTS = ("ありがとう", "ありがとうございます", "助かった", "感謝")
CHECKIN_HINTS = (
    "元気?",
    "元気？",
    "元気か",
    "元気かい",
    "元気ですか",
    "調子どう",
    "調子はどう",
)
DEFINITION_PATTERNS = (
    re.compile(r'^\s*[「『"\']?(?P<term>.+?)[」』"\']?\s*(?:って|とは)\s*(?:何|なに|なん(?:ですか)?|どういう意味)\s*[？?]?$'),
    re.compile(r'^\s*[「『"\'](?P<term>.+?)[」』"\']\s*(?:って|とは)?\s*(?:何|なに|なん(?:ですか)?|どういう意味)?\s*[？?]?$'),
)
ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-./]*")
KATAKANA_WORD_RE = re.compile(r"[ァ-ヶヴー]{2,}")
QUOTE_TRIM = " \t\n\r\"'「」『』()（）[]{}.,!?！？:：;；"


@dataclass
class PlanV1:
    intent: str
    response_mode: str
    required_slots: List[str]
    relation_type_priority: List[str]
    tone: str
    needs_clarification: bool = False
    fallback_reason: str | None = None
    unknown_focus: str | None = None
    wants_definition: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _clean_focus_term(term: str) -> str:
    cleaned = str(term or "").strip(QUOTE_TRIM)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(?:って|とは)$", "", cleaned).strip(QUOTE_TRIM)
    return cleaned


def extract_unknown_focus_term(text: str) -> str | None:
    normalized = str(text).strip()
    for pattern in DEFINITION_PATTERNS:
        match = pattern.match(normalized)
        if not match:
            continue
        term = _clean_focus_term(match.group("term"))
        if len(term) >= 2:
            return term

    ascii_words = ASCII_WORD_RE.findall(normalized)
    if ascii_words and ("?" in normalized or "？" in normalized or _contains_any(normalized, QUESTION_WORDS)):
        return max(ascii_words, key=len)

    katakana_words = [word for word in KATAKANA_WORD_RE.findall(normalized) if word not in {"ナニ", "ドウ", "ナゼ"}]
    if katakana_words and ("?" in normalized or "？" in normalized or _contains_any(normalized, QUESTION_WORDS)):
        return max(katakana_words, key=len)

    return None


def build_plan_v1(
    text: str,
    *,
    topic_count: int = 0,
    unknown_words: int = 0,
    unknown_focus: str | None = None,
) -> PlanV1:
    normalized = str(text).strip()
    unknown_focus = _clean_focus_term(unknown_focus) if unknown_focus else None
    wants_definition = unknown_focus is not None and (
        "?" in normalized or "？" in normalized or _contains_any(normalized, QUESTION_WORDS) or "って" in normalized or "とは" in normalized
    )

    intent = "answer"
    response_mode = "brief_explanation"
    required_slots: List[str] = ["topic"]
    relation_type_priority: List[str] = ["related_to", "hypernym", "hyponym", "paraphrase"]
    tone = "neutral"
    fallback_reason: str | None = None
    needs_clarification = False

    if _contains_any(normalized, THANKS_HINTS):
        intent = "thanks_reply"
        response_mode = "social"
        required_slots = []
        relation_type_priority = ["style_variant", "related_to", "paraphrase"]
        tone = "gentle"
    elif _contains_any(normalized, GREETING_HINTS) or _contains_any(normalized, CHECKIN_HINTS):
        intent = "greeting"
        response_mode = "social"
        required_slots = []
        relation_type_priority = ["style_variant", "related_to", "paraphrase"]
        tone = "gentle"
    elif _contains_any(normalized, EMPATHY_HINTS):
        intent = "empathy"
        response_mode = "supportive"
        relation_type_priority = ["related_to", "style_variant", "paraphrase"]
        tone = "gentle"
    elif _contains_any(normalized, REASON_HINTS):
        intent = "explain_reason"
        response_mode = "reasoned"
        required_slots = ["topic", "reason"]
        relation_type_priority = ["cause_of", "caused_by", "hypernym", "related_to"]
    elif _contains_any(normalized, COMPARE_HINTS):
        intent = "compare"
        response_mode = "contrastive"
        required_slots = ["topic", "comparison"]
        relation_type_priority = ["antonym", "hyponym", "hypernym", "related_to"]
    elif _contains_any(normalized, PROCEDURE_HINTS):
        intent = "procedure"
        response_mode = "steps"
        required_slots = ["topic", "support"]
        relation_type_priority = ["predicate_slot", "argument_role", "related_to", "collocation"]
    elif wants_definition:
        intent = "define"
        response_mode = "brief_definition"
        required_slots = ["topic", "support"]
        relation_type_priority = ["hypernym", "paraphrase", "related_to", "hyponym"]
    elif "?" in normalized or "？" in normalized or _contains_any(normalized, QUESTION_WORDS):
        intent = "answer"
        response_mode = "brief_explanation"
        required_slots = ["topic", "support"]
        relation_type_priority = ["hypernym", "hyponym", "related_to", "paraphrase"]
    else:
        intent = "explain"
        response_mode = "summary"
        required_slots = ["topic", "support"]
        relation_type_priority = ["related_to", "hypernym", "collocation", "paraphrase"]

    if topic_count == 0 and intent not in {"greeting", "thanks_reply"}:
        needs_clarification = True
        required_slots = ["topic"]
        relation_type_priority = ["related_to", "paraphrase"]
        if unknown_focus:
            fallback_reason = "unknown_focus_term"
        else:
            fallback_reason = "topic_not_detected"
    elif unknown_words > max(topic_count, 2):
        fallback_reason = "unknown_words_dominant"

    return PlanV1(
        intent=intent,
        response_mode=response_mode,
        required_slots=required_slots,
        relation_type_priority=relation_type_priority,
        tone=tone,
        needs_clarification=needs_clarification,
        fallback_reason=fallback_reason,
        unknown_focus=unknown_focus,
        wants_definition=wants_definition,
    )


__all__ = ["PlanV1", "build_plan_v1", "extract_unknown_focus_term"]
