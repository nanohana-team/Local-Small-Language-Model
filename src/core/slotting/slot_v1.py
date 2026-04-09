from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping

from src.core.convergence.convergence_v1 import ConvergenceResult
from src.core.divergence.divergence_v1 import DivergenceResult
from src.core.planning.plan_v1 import PlanV1
from src.core.relation.index import RelationIndex


@dataclass
class SlotResult:
    selected_slot_frame: str | None
    filled_slots: Dict[str, Any]
    missing_slots: List[str]
    slot_evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def _seed_concepts_in_order(divergence: DivergenceResult) -> List[tuple[str, str]]:
    ordered: List[tuple[str, str]] = []
    seen: set[str] = set()
    seen_spans: set[tuple[int, int, str]] = set()
    sorted_matches = sorted(
        divergence.seed_matches,
        key=lambda item: (item.start if item.start >= 0 else 10**9, -(item.end - item.start), -item.score),
    )
    for match in sorted_matches:
        span_key = (match.start, match.end, match.surface)
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        for concept_id in match.concept_ids[:1]:
            if concept_id not in seen:
                seen.add(concept_id)
                ordered.append((concept_id, match.surface))
    return ordered


def _find_relation_path(
    index: RelationIndex,
    source_id: str | None,
    target_id: str | None,
    *,
    allowed_types: tuple[str, ...] = ("hypernym",),
    max_depth: int = 8,
) -> List[str] | None:
    if not source_id or not target_id:
        return None
    if source_id == target_id:
        return [source_id]
    queue: List[tuple[str, List[str], int]] = [(source_id, [source_id], 0)]
    visited: set[str] = {source_id}
    while queue:
        current, path, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for relation in index.get_outbound(current):
            if str(relation.get("type")) not in allowed_types:
                continue
            nxt = str(relation.get("target"))
            if nxt in visited:
                continue
            next_path = path + [nxt]
            if nxt == target_id:
                return next_path
            visited.add(nxt)
            queue.append((nxt, next_path, depth + 1))
    return None


def _find_direct_relation(index: RelationIndex, source_id: str | None, target_id: str | None) -> str | None:
    if not source_id or not target_id:
        return None
    for relation in index.get_outbound(source_id):
        if str(relation.get("target")) == target_id:
            return str(relation.get("type"))
    return None


def _concept_record(index: RelationIndex, concept_id: str | None, *, override_label: str | None = None) -> Dict[str, Any] | None:
    if not concept_id:
        return None
    concept = index.get_concept(concept_id)
    if not concept:
        return None
    return {
        "id": concept_id,
        "label": str(override_label or index.concept_label(concept_id)),
        "category": str(concept.get("category") or ""),
        "description": str(concept.get("description") or ""),
    }



def fill_slots_v1(
    plan: PlanV1,
    divergence: DivergenceResult,
    convergence: ConvergenceResult,
    index: RelationIndex,
) -> SlotResult:
    accepted = convergence.accepted_concepts
    ordered_seed_concepts = _seed_concepts_in_order(divergence)

    topic = None
    if ordered_seed_concepts:
        topic = _concept_record(index, ordered_seed_concepts[0][0], override_label=ordered_seed_concepts[0][1])
    elif accepted:
        topic = _concept_record(index, accepted[0]["concept_id"])

    support = None
    if len(ordered_seed_concepts) > 1:
        support = _concept_record(index, ordered_seed_concepts[1][0], override_label=ordered_seed_concepts[1][1])
    elif len(accepted) > 1:
        topic_label = topic["label"] if topic else None
        for candidate in accepted[1:]:
            candidate_record = _concept_record(index, candidate["concept_id"])
            if candidate_record is None:
                continue
            if topic_label and candidate_record["label"] == topic_label:
                continue
            support = candidate_record
            break
        if support is None:
            support = _concept_record(index, accepted[1]["concept_id"])

    reason = None
    comparison = None
    for concept in accepted[1:]:
        relation_types = set(concept.get("via_relation_types", []))
        if reason is None and relation_types & {"cause_of", "caused_by"}:
            reason = _concept_record(index, concept["concept_id"])
        if comparison is None and relation_types & {"antonym", "hyponym", "hypernym"}:
            comparison = _concept_record(index, concept["concept_id"])

    if plan.intent == "compare" and comparison is None:
        comparison = support
    if plan.intent == "explain_reason" and reason is None:
        reason = support

    selected_slot_frame = None
    if topic is not None:
        raw_topic = index.get_concept(topic["id"]) or {}
        selected_slot_frame = raw_topic.get("default_slot_frame_id") if isinstance(raw_topic, Mapping) else None

    filled_slots: Dict[str, Any] = {
        "topic": topic,
        "support": support,
        "reason": reason,
        "comparison": comparison,
    }
    missing_slots = [slot_name for slot_name in plan.required_slots if filled_slots.get(slot_name) is None]
    topic_support_relation = None
    inverse_topic_support_relation = None
    topic_support_hypernym_path = None
    inverse_topic_support_hypernym_path = None
    if topic and support:
        topic_support_relation = _find_direct_relation(index, topic["id"], support["id"])
        inverse_topic_support_relation = _find_direct_relation(index, support["id"], topic["id"])
        topic_support_hypernym_path = _find_relation_path(index, topic["id"], support["id"], allowed_types=("hypernym",), max_depth=8)
        inverse_topic_support_hypernym_path = _find_relation_path(index, support["id"], topic["id"], allowed_types=("hypernym",), max_depth=8)

    slot_evidence = {
        "seed_matches": [match.to_dict() for match in divergence.seed_matches[:4]],
        "accepted_concepts": accepted[:4],
        "topic_support_relation": topic_support_relation,
        "inverse_topic_support_relation": inverse_topic_support_relation,
        "topic_support_hypernym_path": topic_support_hypernym_path,
        "inverse_topic_support_hypernym_path": inverse_topic_support_hypernym_path,
    }
    return SlotResult(
        selected_slot_frame=str(selected_slot_frame) if selected_slot_frame else None,
        filled_slots=filled_slots,
        missing_slots=missing_slots,
        slot_evidence=slot_evidence,
    )


__all__ = ["SlotResult", "fill_slots_v1"]
