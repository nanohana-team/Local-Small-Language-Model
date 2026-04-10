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



def _frame_slot_names(slots: SlotResult) -> list[str]:
    names = slots.slot_evidence.get("frame_slot_names")
    if isinstance(names, list):
        return [str(name) for name in names if name]
    return []



def _uses_slot_relations(slots: SlotResult) -> bool:
    for key in ("predicate_slot_targets", "support_slot_targets"):
        value = slots.slot_evidence.get(key)
        if isinstance(value, list) and value:
            return True
    return False



def render_surface_v1(plan: PlanV1, slots: SlotResult, *, accepted_relations: list[dict[str, Any]] | None = None) -> Dict[str, Any]:
    topic_slot = slots.filled_slots.get("topic")
    support_slot = slots.filled_slots.get("support")
    reason_slot = slots.filled_slots.get("reason")
    comparison_slot = slots.filled_slots.get("comparison")
    common_hypernym_slot = slots.filled_slots.get("common_hypernym")
    topic = _label(topic_slot)
    support = _label(support_slot)
    reason = _label(reason_slot)
    comparison = _label(comparison_slot)
    common_hypernym = _label(common_hypernym_slot)
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
    frame_slot_names = _frame_slot_names(slots)
    slot_relation_used = _uses_slot_relations(slots)

    if plan.intent == "greeting":
        return {
            "sentence_plan": {"mode": "social", "kind": "greeting"},
            "final_text": "やあ。元気だよ。そっちはどう？",
            "style_choice": "gentle",
        }

    if plan.intent == "thanks_reply":
        return {
            "sentence_plan": {"mode": "social", "kind": "thanks_reply"},
            "final_text": "どういたしまして。必要ならこのまま続きも見ていけます。",
            "style_choice": "gentle",
        }

    if plan.needs_clarification or topic is None:
        if plan.fallback_reason == "unknown_focus_term" and plan.unknown_focus:
            text = (
                f"「{plan.unknown_focus}」という語は、いまの辞書だけだと手がかりがまだ足りません。"
                "分野や前後の文脈を少し足してもらえると、意味をかなり絞りやすくなります。"
            )
            return {
                "sentence_plan": {"mode": "clarification", "kind": "unknown_focus_term"},
                "final_text": text,
                "style_choice": "neutral",
            }
        text = "まだ主題の手がかりが少ないので、知りたい対象を一語か短文で教えてください。"
        return {"sentence_plan": {"mode": "clarification"}, "final_text": text, "style_choice": "neutral"}

    if plan.intent == "empathy":
        if support and support != topic:
            text = f"かなりしんどそうです。いまは{topic}に対して{support}が強い焦点になっています。"
        else:
            text = f"かなりしんどそうです。まずは{topic}という気持ちをそのまま置いて、負荷を増やさない形で整えるのがよさそうです。"
    elif plan.intent == "explain_reason":
        if reason:
            text = f"{topic}については、理由候補として{reason}がいちばん近いです。"
            if support and support != reason:
                text += f" 補助線としては{support}も見ておくと整理しやすいです。"
        elif slot_relation_used or frame_slot_names:
            text = f"{topic}については、理由を組み立てる骨格は見えていますが、原因を一つに確定できるほどの接続はまだ足りません。"
        else:
            text = f"{topic}については、いまの辞書だけでは理由を一つに絞り切れません。関連候補をもう少し足すと詰めやすいです。"
    elif plan.intent == "compare":
        if comparison and common_hypernym:
            text = f"{topic}と{comparison}は、どちらも{common_hypernym}ですが、同じ概念ではありません。分類や性質の違いを見ると整理しやすいです。"
        elif comparison:
            text = f"{topic}と{comparison}は別の概念です。いま見えている範囲では、分類や性質の違いを軸に比べるのが自然です。"
        else:
            text = f"{topic}は比較相手を一つ決めると整理しやすいです。"
    elif plan.intent == "procedure":
        if slot_relation_used and frame_slot_names:
            text = f"{topic}については、{', '.join(frame_slot_names[:3])} などの役割枠は見えています。具体手順はこの骨格に沿って順番化すると自然です。"
        elif support:
            text = f"{topic}は、まず{support}から触るのが自然です。具体手順はまだ粗いので、必要なら対象をもう少し絞ると詰めやすいです。"
        else:
            text = f"{topic}は、まず対象を一つに絞ってから順番を立てるのが安定します。"
    elif plan.intent == "define" and isinstance(topic_slot, Mapping) and topic_slot.get("category") == "unknown":
        text = f"「{topic}」は、いまの辞書にはまだ十分な知識がありません。分野や前後の文脈を足してもらえると、定義をかなり絞りやすくなります。"
    else:
        if support and (topic_support_relation == "hypernym" or topic_support_hypernym_path):
            text = f"{topic}は{support}の一種です。"
        elif support and (inverse_topic_support_relation == "hypernym" or inverse_topic_support_hypernym_path):
            text = f"{topic}は{support}の一種です。"
        elif support and topic_support_relation == "hyponym":
            text = f"{support}は{topic}の一種です。"
        elif support and inverse_topic_support_relation == "hyponym":
            text = f"{support}は{topic}の一種です。"
        elif topic_description and not _mostly_ascii(topic_description):
            description = str(topic_description).rstrip("。")
            text = f"{topic}は、{description}です。"
        elif support:
            if relation_hint and (topic_support_relation or inverse_topic_support_relation or topic_support_hypernym_path or inverse_topic_support_hypernym_path):
                text = f"{topic}については、{support}が近い手がかりです。relation では{relation_hint}の線が見えています。"
            else:
                text = f"{topic}については、{support}が近い手がかりです。"
        else:
            text = f"{topic}については、まだ補助線が少ないので、文脈を少し足すとかなり自然に返しやすくなります。"

    return {
        "sentence_plan": {
            "mode": plan.response_mode,
            "relation_hint": relation_hint,
            "slot_frame": slots.selected_slot_frame,
            "slot_frame_names": frame_slot_names,
        },
        "final_text": text,
        "style_choice": plan.tone,
    }


__all__ = ["render_surface_v1"]
