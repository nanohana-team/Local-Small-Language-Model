from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence

from src.core.divergence.divergence_v1 import CandidateConcept, DivergenceResult, SeedMatch
from src.core.planning.plan_v1 import PlanV1
from src.core.relation.index import RelationIndex


@dataclass
class ConvergenceResult:
    accepted_concepts: List[Dict[str, Any]]
    rejected_concepts: List[Dict[str, Any]]
    accepted_relations: List[Dict[str, Any]]
    rejected_relations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


REVERSE_SUPPORT_TYPES = {
    "define": {"hypernym", "paraphrase", "related_to", "hyponym"},
    "answer": {"hypernym", "hyponym", "related_to", "paraphrase"},
    "explain": {"related_to", "hypernym", "collocation", "paraphrase"},
    "explain_reason": {"cause_of", "caused_by", "related_to", "hypernym"},
    "compare": {"antonym", "hypernym", "hyponym", "related_to"},
    "procedure": {"predicate_slot", "argument_role", "collocation", "related_to"},
}


def _sorted_seed_matches(seed_matches: Sequence[SeedMatch]) -> List[SeedMatch]:
    return sorted(
        seed_matches,
        key=lambda item: (item.start if item.start >= 0 else 10**9, -(item.end - item.start), -item.score),
    )



def _explicit_seed_concepts(divergence: DivergenceResult) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for match in _sorted_seed_matches(divergence.seed_matches):
        if not match.concept_ids:
            continue
        concept_id = str(match.concept_ids[0])
        if concept_id not in seen:
            seen.add(concept_id)
            ordered.append(concept_id)
    return ordered



def _plan_bonus(candidate: CandidateConcept, plan: PlanV1) -> float:
    bonus = 0.0
    relation_types = set(candidate.via_relation_types)
    if candidate.depth == 0:
        bonus += 0.25
    if plan.intent == "explain_reason" and relation_types & {"cause_of", "caused_by"}:
        bonus += 0.30
    if plan.intent == "compare" and relation_types & {"antonym", "hyponym", "hypernym"}:
        bonus += 0.24
    if plan.intent == "procedure" and relation_types & {"predicate_slot", "argument_role", "collocation"}:
        bonus += 0.22
    if plan.intent in {"answer", "explain", "define"} and relation_types & {"hypernym", "hyponym", "related_to", "paraphrase"}:
        bonus += 0.18
    overlap = len([relation_type for relation_type in candidate.via_relation_types if relation_type in plan.relation_type_priority])
    bonus += min(0.20, 0.05 * overlap)
    return bonus



def _constraint_relations(
    candidate: CandidateConcept,
    plan: PlanV1,
    index: RelationIndex,
    anchor_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed_types = REVERSE_SUPPORT_TYPES.get(plan.intent, set(plan.relation_type_priority or []))
    collected: List[Dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for relation in index.get_inbound(candidate.concept_id):
        source = str(relation.get("source_concept"))
        relation_type = str(relation.get("type"))
        if source not in anchor_ids or relation_type not in allowed_types:
            continue
        key = (source, relation_type, candidate.concept_id)
        if key in seen:
            continue
        seen.add(key)
        collected.append(
            {
                "from": source,
                "type": relation_type,
                "to": candidate.concept_id,
                "weight": round(float(relation.get("weight", 1.0)), 6),
                "reason": f"constraint:inbound:{plan.intent}",
                "depth": 0,
                "seed": source,
            }
        )

    for anchor_id in anchor_ids:
        if anchor_id == candidate.concept_id:
            continue
        for relation in index.get_outbound(candidate.concept_id):
            target = str(relation.get("target"))
            relation_type = str(relation.get("type"))
            if target != anchor_id or relation_type not in allowed_types:
                continue
            key = (candidate.concept_id, relation_type, target)
            if key in seen:
                continue
            seen.add(key)
            collected.append(
                {
                    "from": candidate.concept_id,
                    "type": relation_type,
                    "to": target,
                    "weight": round(float(relation.get("weight", 1.0)), 6),
                    "reason": f"constraint:outbound:{plan.intent}",
                    "depth": 0,
                    "seed": candidate.seed_concept,
                }
            )
    return collected



def _slot_frame_bonus(candidate_id: str, plan: PlanV1, index: RelationIndex) -> float:
    concept = index.get_concept(candidate_id) or {}
    slot_frame_id = concept.get("default_slot_frame_id")
    slot_frame = index.get_slot_frame(str(slot_frame_id)) if slot_frame_id else None
    if not isinstance(slot_frame, Mapping):
        return 0.0
    slots = slot_frame.get("slots", [])
    slot_names = {str(slot.get("name")) for slot in slots if isinstance(slot, Mapping) and slot.get("name")}
    if not slot_names:
        return 0.0
    if plan.intent == "procedure" and slot_names & {"actor", "target", "content"}:
        return 0.12
    if plan.intent == "explain_reason" and slot_names & {"cause", "result", "subject", "target"}:
        return 0.10
    if plan.intent == "define":
        return 0.04
    return 0.0



def _compare_focus_bonus(candidate: CandidateConcept, explicit_seed_ids: Sequence[str], index: RelationIndex) -> float:
    if candidate.concept_id in explicit_seed_ids[:2]:
        return 0.55
    if len(explicit_seed_ids) < 2:
        return 0.0

    bonus = 0.0
    left_id, right_id = explicit_seed_ids[:2]
    connected_to_pair = False
    for relation in index.get_outbound(candidate.concept_id):
        target = str(relation.get("target"))
        relation_type = str(relation.get("type"))
        if target in {left_id, right_id} and relation_type in {"hypernym", "hyponym", "antonym", "related_to"}:
            connected_to_pair = True
            bonus += 0.08
    for relation in index.get_inbound(candidate.concept_id):
        source = str(relation.get("source_concept"))
        relation_type = str(relation.get("type"))
        if source in {left_id, right_id} and relation_type in {"hypernym", "hyponym", "antonym", "related_to"}:
            connected_to_pair = True
            bonus += 0.08
    if not connected_to_pair and candidate.depth > 0:
        bonus -= 0.10
    return bonus



def _record_candidate(
    candidate: CandidateConcept,
    final_score: float,
    constraint_relations: Sequence[Dict[str, Any]],
    *,
    preserve_reason: str | None = None,
) -> Dict[str, Any]:
    record = candidate.to_dict()
    record["final_score"] = final_score
    record["constraint_relations"] = list(constraint_relations)
    if preserve_reason:
        record["preserve_reason"] = preserve_reason
    return record



def _append_unique_relation(bucket: List[Dict[str, Any]], relation: Mapping[str, Any]) -> None:
    if relation not in bucket:
        bucket.append(dict(relation))



def run_convergence_v1(
    divergence: DivergenceResult,
    plan: PlanV1,
    index: RelationIndex,
    *,
    max_accept: int = 5,
) -> ConvergenceResult:
    explicit_seed_ids = _explicit_seed_concepts(divergence)
    anchor_ids = explicit_seed_ids[:2] if plan.intent == "compare" else explicit_seed_ids[:1] or explicit_seed_ids

    scored: List[tuple[float, CandidateConcept, List[Dict[str, Any]], str | None]] = []
    for candidate in divergence.candidate_concepts:
        constraint_relations = _constraint_relations(candidate, plan, index, anchor_ids)
        score = float(candidate.score) + _plan_bonus(candidate, plan)
        concept = index.get_concept(candidate.concept_id) or {}
        category = str(concept.get("category", ""))
        if category in {"entity", "event", "state", "quality"}:
            score += 0.04
        score += _slot_frame_bonus(candidate.concept_id, plan, index)
        score += min(0.24, 0.08 * len(constraint_relations))

        preserve_reason = None
        if candidate.concept_id in explicit_seed_ids:
            score += 0.18
            preserve_reason = "explicit_seed"
        if plan.intent == "define" and plan.unknown_focus and candidate.concept_id in explicit_seed_ids[:1]:
            score += 0.24
            preserve_reason = "definition_focus"
        if plan.intent == "compare":
            score += _compare_focus_bonus(candidate, explicit_seed_ids, index)
            if candidate.concept_id in explicit_seed_ids[:2]:
                preserve_reason = "compare_focus"

        scored.append((round(score, 6), candidate, constraint_relations, preserve_reason))

    scored.sort(key=lambda item: (item[0], -item[1].depth, item[1].label), reverse=True)
    accepted_concepts: List[Dict[str, Any]] = []
    rejected_concepts: List[Dict[str, Any]] = []
    accepted_ids: set[str] = set()
    accepted_relations: List[Dict[str, Any]] = []
    rejected_relations: List[Dict[str, Any]] = []

    preferred_ids: List[str] = []
    if plan.intent == "compare":
        preferred_ids.extend(explicit_seed_ids[:2])
    elif plan.intent == "define":
        preferred_ids.extend(explicit_seed_ids[:1])

    scored_by_id = {candidate.concept_id: (final_score, candidate, constraints, preserve_reason) for final_score, candidate, constraints, preserve_reason in scored}
    for concept_id in preferred_ids:
        item = scored_by_id.get(concept_id)
        if item is None or concept_id in accepted_ids or len(accepted_concepts) >= max_accept:
            continue
        final_score, candidate, constraints, preserve_reason = item
        accepted_ids.add(concept_id)
        accepted_concepts.append(_record_candidate(candidate, final_score, constraints, preserve_reason=preserve_reason))
        for edge in candidate.path:
            _append_unique_relation(accepted_relations, edge)
        for edge in constraints:
            _append_unique_relation(accepted_relations, edge)

    for final_score, candidate, constraints, preserve_reason in scored:
        record = _record_candidate(candidate, final_score, constraints, preserve_reason=preserve_reason)
        if len(accepted_concepts) < max_accept and candidate.concept_id not in accepted_ids:
            accepted_concepts.append(record)
            accepted_ids.add(candidate.concept_id)
            for edge in candidate.path:
                _append_unique_relation(accepted_relations, edge)
            for edge in constraints:
                _append_unique_relation(accepted_relations, edge)
        else:
            rejected_concepts.append(record)
            for edge in candidate.path:
                if edge not in accepted_relations and edge not in rejected_relations:
                    rejected_relations.append(edge)
            for edge in constraints:
                if edge not in accepted_relations and edge not in rejected_relations:
                    rejected_relations.append(edge)

    return ConvergenceResult(
        accepted_concepts=accepted_concepts,
        rejected_concepts=rejected_concepts,
        accepted_relations=accepted_relations,
        rejected_relations=rejected_relations,
    )


__all__ = ["ConvergenceResult", "run_convergence_v1"]
