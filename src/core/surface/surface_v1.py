from __future__ import annotations

from typing import Any, Dict, Mapping

from src.core.planning.plan_v1 import PlanV1
from src.core.slotting.slot_v1 import SlotResult


RELATION_JA = {
    "hypernym": "上位概念",
    "hyponym": "下位概念",
    "cause_of": "因果",
    "caused_by": "因果",
    "related_to": "関連",
    "paraphrase": "言い換え",
    "antonym": "対比",
    "predicate_slot": "述語スロット",
    "argument_role": "役割",
    "collocation": "共起",
}


def _mostly_ascii(text: str | None) -> bool:
    if not text:
        return False
    printable = [ch for ch in text if not ch.isspace()]
    if not printable:
        return False
    ascii_count = sum(1 for ch in printable if ord(ch) < 128)
    return ascii_count / max(1, len(printable)) >= 0.7


def _label(slot: Mapping[str, Any] | None) -> str | None:
    if not isinstance(slot, Mapping):
        return None
    label = slot.get("label")
    return str(label) if label else None



def render_surface_v1(plan: PlanV1, slots: SlotResult, *, accepted_relations: list[dict[str, Any]] | None = None) -> Dict[str, Any]:
    topic_slot = slots.filled_slots.get("topic")
    support_slot = slots.filled_slots.get("support")
    reason_slot = slots.filled_slots.get("reason")
    comparison_slot = slots.filled_slots.get("comparison")
    topic = _label(topic_slot)
    support = _label(support_slot)
    reason = _label(reason_slot)
    comparison = _label(comparison_slot)
    topic_description = topic_slot.get("description") if isinstance(topic_slot, Mapping) else None

    accepted_relations = accepted_relations or []
    relation_hint = None
    if accepted_relations:
        relation_type = str(accepted_relations[0].get("type", "related_to"))
        relation_hint = RELATION_JA.get(relation_type, relation_type)

    topic_support_relation = slots.slot_evidence.get("topic_support_relation")
    inverse_topic_support_relation = slots.slot_evidence.get("inverse_topic_support_relation")
    topic_support_hypernym_path = slots.slot_evidence.get("topic_support_hypernym_path")
    inverse_topic_support_hypernym_path = slots.slot_evidence.get("inverse_topic_support_hypernym_path")

    if plan.needs_clarification or topic is None:
        text = "まだ主題の手がかりが少ないので、知りたい対象を一語か短文で教えてください。"
        return {"sentence_plan": {"mode": "clarification"}, "final_text": text, "style_choice": "neutral"}

    if plan.intent == "empathy":
        if support and support != topic:
            text = f"かなりしんどそうです。いまは{topic}に対して{support}が強い焦点になっています。"
        else:
            text = f"かなりしんどそうです。まずは{topic}という気持ちをそのまま置いて、負荷を増やさない形で整えるのがよさそうです。"
    elif plan.intent == "explain_reason":
        if reason and support and reason != support:
            text = f"{topic}については、{reason}との接続が強いです。補助線としては{support}を見るとまとまりやすいです。"
        elif reason:
            text = f"{topic}については、{reason}との接続が理由の中心です。"
        else:
            text = f"{topic}については、周辺の接続をもう少し増やすと理由を詰めやすいです。"
    elif plan.intent == "compare":
        if comparison:
            text = f"{topic}を見ると、{comparison}との違いが比較の軸になります。"
        elif support:
            text = f"{topic}を見ると、{support}との並びで比べるのが自然です。"
        else:
            text = f"{topic}は比較相手を一つ決めると整理しやすいです。"
    elif plan.intent == "procedure":
        if support:
            text = f"{topic}は、まず{support}から触るのが自然です。細部はそのあとで詰める形が安定します。"
        else:
            text = f"{topic}は、まず主題を一つに絞ってから順に接続を増やすのが安定します。"
    else:
        if support and (topic_support_relation == "hypernym" or topic_support_hypernym_path):
            text = f"{topic}は{support}の一種です。"
        elif support and (inverse_topic_support_relation == "hypernym" or inverse_topic_support_hypernym_path):
            text = f"{topic}は{support}の一種です。"
        elif support and topic_support_relation == "hyponym":
            text = f"{support}は{topic}の一種です。"
        elif support and inverse_topic_support_relation == "hyponym":
            text = f"{support}は{topic}の一種です。"
        elif topic_description and not _mostly_ascii(topic_description) and (not support or support == topic or support in {"何", "質問"}):
            text = f"{topic}は、{topic_description}"
        elif topic_description and not _mostly_ascii(topic_description) and relation_hint not in {"上位概念", "下位概念"}:
            text = f"{topic}は、{topic_description}"
        elif support and relation_hint:
            text = f"{topic}については、{support}が近い手がかりです。relation で見ると{relation_hint}の線が強めです。"
        elif support:
            text = f"{topic}については、{support}が近い手がかりです。"
        else:
            text = f"{topic}を中心に見るのが、いまの入力ではいちばん自然です。"

    return {
        "sentence_plan": {"mode": plan.response_mode, "relation_hint": relation_hint},
        "final_text": text,
        "style_choice": plan.tone,
    }


__all__ = ["render_surface_v1"]
