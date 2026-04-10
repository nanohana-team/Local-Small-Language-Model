from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence

from src.core.convergence.convergence_v1 import ConvergenceResult
from src.core.divergence.divergence_v1 import DivergenceResult
from src.core.planning.plan_v1 import PlanV1
from src.core.relation.index import RelationIndex

MIN_STRONG_SEED_SCORE = 0.25
LOW_SIGNAL_CONCEPT_CATEGORIES = {
    "grammar",
    "discourse",
    "prefix",
    "suffix",
    "adnominal",
}


@dataclass
class SlotResult:
    selected_slot_frame: str | None
    filled_slots: Dict[str, Any]
    missing_slots: List[str]
    slot_evidence: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def _is_low_signal_concept(index: RelationIndex, concept_id: str | None) -> bool:
    if not concept_id:
        return False
    concept = index.get_concept(concept_id) or {}
    category = str(concept.get("category") or "")
    return category in LOW_SIGNAL_CONCEPT_CATEGORIES



def _seed_concepts_in_order(divergence: DivergenceResult, index: RelationIndex) -> List[tuple[str, str]]:
    ordered_all: List[tuple[str, str]] = []
    ordered_preferred: List[tuple[str, str]] = []
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
            if concept_id in seen:
                continue
            seen.add(concept_id)
            pair = (concept_id, match.surface)
            ordered_all.append(pair)
            if match.score >= MIN_STRONG_SEED_SCORE and not _is_low_signal_concept(index, concept_id):
                ordered_preferred.append(pair)
    return ordered_preferred or ordered_all



def _find_relation_path(
    index: RelationIndex,
    source_id: str | None,
    target_id: str | None,
    *,
    allowed_types: Sequence[str] = ("hypernym",),
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


def _support_has_meaningful_bridge(index: RelationIndex, topic: Mapping[str, Any] | None, support: Mapping[str, Any] | None) -> bool:
    if not isinstance(topic, Mapping) or not isinstance(support, Mapping):
        return False
    topic_id = topic.get("id")
    support_id = support.get("id")
    if not topic_id or not support_id:
        return False
    direct = _find_direct_relation(index, topic_id, support_id)
    inverse = _find_direct_relation(index, support_id, topic_id)
    if direct or inverse:
        return True
    if _find_relation_path(index, topic_id, support_id, allowed_types=("hypernym",), max_depth=6):
        return True
    if _find_relation_path(index, support_id, topic_id, allowed_types=("hypernym",), max_depth=6):
        return True
    return False



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



def _unknown_topic_record(label: str) -> Dict[str, Any]:
    return {
        "id": None,
        "label": str(label),
        "category": "unknown",
        "description": "",
    }



def _find_common_hypernym(index: RelationIndex, left_id: str | None, right_id: str | None, *, max_depth: int = 6) -> Dict[str, Any] | None:
    if not left_id or not right_id:
        return None

    def walk(start_id: str) -> Dict[str, int]:
        distances: Dict[str, int] = {start_id: 0}
        queue = deque([(start_id, 0)])
        while queue:
            current, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for relation in index.get_outbound(current):
                if str(relation.get("type")) != "hypernym":
                    continue
                target = str(relation.get("target"))
                if target in distances:
                    continue
                distances[target] = depth + 1
                queue.append((target, depth + 1))
        return distances

    left_map = walk(left_id)
    right_map = walk(right_id)
    shared = set(left_map) & set(right_map)
    shared.discard(left_id)
    shared.discard(right_id)
    if not shared:
        return None
    best_id = min(shared, key=lambda cid: (left_map[cid] + right_map[cid], max(left_map[cid], right_map[cid]), index.concept_label(cid)))
    return _concept_record(index, best_id)



def _slot_frame_from_candidate(index: RelationIndex, concept_id: str | None) -> tuple[str | None, Mapping[str, Any] | None]:
    if not concept_id:
        return None, None
    concept = index.get_concept(concept_id) or {}
    slot_frame_id = concept.get("default_slot_frame_id")
    if not slot_frame_id:
        return None, None
    slot_frame = index.get_slot_frame(str(slot_frame_id))
    if not isinstance(slot_frame, Mapping):
        return None, None
    return str(slot_frame_id), slot_frame



def _pick_selected_slot_frame(
    index: RelationIndex,
    topic: Mapping[str, Any] | None,
    support: Mapping[str, Any] | None,
    accepted: Sequence[Mapping[str, Any]],
) -> tuple[str | None, Mapping[str, Any] | None]:
    for candidate in (topic, support):
        if isinstance(candidate, Mapping):
            slot_frame_id, slot_frame = _slot_frame_from_candidate(index, candidate.get("id"))
            if slot_frame_id:
                return slot_frame_id, slot_frame
    for concept in accepted:
        slot_frame_id, slot_frame = _slot_frame_from_candidate(index, concept.get("concept_id"))
        if slot_frame_id:
            return slot_frame_id, slot_frame
    return None, None



def _relation_targets(index: RelationIndex, concept_id: str | None, relation_types: Sequence[str]) -> List[Dict[str, Any]]:
    if not concept_id:
        return []
    targets: List[Dict[str, Any]] = []
    seen: set[str] = set()
    allowed = set(relation_types)
    for relation in index.get_outbound(concept_id):
        if str(relation.get("type")) not in allowed:
            continue
        target_id = str(relation.get("target"))
        if target_id in seen:
            continue
        seen.add(target_id)
        record = _concept_record(index, target_id)
        if record is not None:
            record["relation_type"] = str(relation.get("type"))
            targets.append(record)
    return targets



def _fill_frame_slots(
    plan: PlanV1,
    slot_frame: Mapping[str, Any] | None,
    *,
    topic: Mapping[str, Any] | None,
    support: Mapping[str, Any] | None,
    reason: Mapping[str, Any] | None,
    comparison: Mapping[str, Any] | None,
    common_hypernym: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    if not isinstance(slot_frame, Mapping):
        return {}

    mapping: Dict[str, Any] = {}
    slot_defs = slot_frame.get("slots", []) if isinstance(slot_frame.get("slots"), list) else []
    for slot_def in slot_defs:
        if not isinstance(slot_def, Mapping):
            continue
        name = str(slot_def.get("name") or "").strip()
        if not name:
            continue
        value = None
        if name in {"subject", "actor", "content"}:
            value = topic
        elif name in {"target", "object"}:
            value = comparison if plan.intent == "compare" and comparison is not None else support
        elif name == "cause":
            value = reason
        elif name in {"result", "goal"}:
            value = support or comparison
        elif name == "degree":
            value = common_hypernym if plan.intent == "compare" else None
        mapping[name] = value
    return mapping



def fill_slots_v1(
    plan: PlanV1,
    divergence: DivergenceResult,
    convergence: ConvergenceResult,
    index: RelationIndex,
) -> SlotResult:
    accepted = convergence.accepted_concepts
    ordered_seed_concepts = _seed_concepts_in_order(divergence, index)
    explicit_pair = ordered_seed_concepts[:2]

    topic = None
    if plan.intent == "define" and plan.unknown_focus:
        if ordered_seed_concepts:
            seed_surface = ordered_seed_concepts[0][1]
            if seed_surface and seed_surface != plan.unknown_focus and seed_surface in plan.unknown_focus:
                topic = _unknown_topic_record(plan.unknown_focus)
        if topic is None and not ordered_seed_concepts:
            topic = _unknown_topic_record(plan.unknown_focus)

    if topic is None:
        if ordered_seed_concepts:
            topic = _concept_record(index, ordered_seed_concepts[0][0], override_label=ordered_seed_concepts[0][1])
        elif accepted:
            topic = _concept_record(index, accepted[0]["concept_id"])

    support = None
    if plan.intent in {"greeting", "thanks_reply"}:
        support = None
    elif plan.intent == "define" and isinstance(topic, Mapping) and topic.get("category") == "unknown":
        support = None
    elif len(ordered_seed_concepts) > 1:
        candidate_support = _concept_record(index, ordered_seed_concepts[1][0], override_label=ordered_seed_concepts[1][1])
        if _support_has_meaningful_bridge(index, topic, candidate_support):
            support = candidate_support
    elif len(accepted) > 1:
        topic_label = topic["label"] if topic else None
        for candidate in accepted[1:]:
            candidate_record = _concept_record(index, candidate["concept_id"])
            if candidate_record is None:
                continue
            if topic_label and candidate_record["label"] == topic_label:
                continue
            if _is_low_signal_concept(index, candidate["concept_id"]):
                continue
            if not _support_has_meaningful_bridge(index, topic, candidate_record):
                continue
            support = candidate_record
            break

    reason = None
    comparison = None
    for concept in accepted[1:]:
        relation_types = set(concept.get("via_relation_types", [])) | {
            str(edge.get("type")) for edge in concept.get("constraint_relations", []) if isinstance(edge, Mapping)
        }
        if plan.intent not in {"greeting", "thanks_reply"}:
            if reason is None and relation_types & {"cause_of", "caused_by"}:
                reason = _concept_record(index, concept["concept_id"])
            if comparison is None and relation_types & {"antonym", "hyponym", "hypernym"}:
                comparison = _concept_record(index, concept["concept_id"])

    if plan.intent == "compare":
        if len(explicit_pair) > 1:
            comparison = _concept_record(index, explicit_pair[1][0], override_label=explicit_pair[1][1])
        elif comparison is None:
            comparison = support
    if plan.intent == "explain_reason" and reason is None:
        reason = support

    common_hypernym = None
    if plan.intent == "compare" and topic and comparison and topic.get("id") and comparison.get("id"):
        common_hypernym = _find_common_hypernym(index, topic.get("id"), comparison.get("id"))

    selected_slot_frame, slot_frame = _pick_selected_slot_frame(index, topic, support, accepted)
    frame_slots = _fill_frame_slots(
        plan,
        slot_frame,
        topic=topic,
        support=support,
        reason=reason,
        comparison=comparison,
        common_hypernym=common_hypernym,
    )

    predicate_slot_targets = _relation_targets(index, topic.get("id") if isinstance(topic, Mapping) else None, ("predicate_slot", "argument_role", "subject_predicate"))
    support_targets = _relation_targets(index, support.get("id") if isinstance(support, Mapping) else None, ("predicate_slot", "argument_role", "subject_predicate"))

    filled_slots: Dict[str, Any] = {
        "topic": topic,
        "support": support,
        "reason": reason,
        "comparison": comparison,
        "common_hypernym": common_hypernym,
        "frame_slots": frame_slots,
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
        "explicit_seed_pair": [surface for _, surface in explicit_pair],
        "topic_support_relation": topic_support_relation,
        "inverse_topic_support_relation": inverse_topic_support_relation,
        "topic_support_hypernym_path": topic_support_hypernym_path,
        "inverse_topic_support_hypernym_path": inverse_topic_support_hypernym_path,
        "slot_frame": dict(slot_frame) if isinstance(slot_frame, Mapping) else None,
        "frame_slot_names": sorted(frame_slots.keys()),
        "predicate_slot_targets": predicate_slot_targets[:6],
        "support_slot_targets": support_targets[:6],
    }
    return SlotResult(
        selected_slot_frame=str(selected_slot_frame) if selected_slot_frame else None,
        filled_slots=filled_slots,
        missing_slots=missing_slots,
        slot_evidence=slot_evidence,
    )


__all__ = ["SlotResult", "fill_slots_v1"]
