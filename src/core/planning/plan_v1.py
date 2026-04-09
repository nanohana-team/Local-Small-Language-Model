from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping


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


@dataclass
class PlanV1:
    intent: str
    response_mode: str
    required_slots: List[str]
    relation_type_priority: List[str]
    tone: str
    needs_clarification: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)



def build_plan_v1(text: str, *, topic_count: int = 0, unknown_words: int = 0) -> PlanV1:
    normalized = str(text).strip()

    intent = "answer"
    response_mode = "brief_explanation"
    required_slots: List[str] = ["topic"]
    relation_type_priority: List[str] = ["related_to", "hypernym", "hyponym", "paraphrase"]
    tone = "neutral"
    fallback_reason: str | None = None
    needs_clarification = False

    if _contains_any(normalized, EMPATHY_HINTS):
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

    if topic_count == 0:
        needs_clarification = True
        fallback_reason = "topic_not_detected"
        required_slots = ["topic"]
        relation_type_priority = ["related_to", "paraphrase"]
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
    )


__all__ = ["PlanV1", "build_plan_v1"]
