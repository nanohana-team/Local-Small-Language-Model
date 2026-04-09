from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from src.core.divergence.divergence_v1 import CandidateConcept, DivergenceResult
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
    if plan.intent in {"answer", "explain"} and relation_types & {"hypernym", "hyponym", "related_to", "paraphrase"}:
        bonus += 0.18
    overlap = len([relation_type for relation_type in candidate.via_relation_types if relation_type in plan.relation_type_priority])
    bonus += min(0.20, 0.05 * overlap)
    return bonus



def run_convergence_v1(
    divergence: DivergenceResult,
    plan: PlanV1,
    index: RelationIndex,
    *,
    max_accept: int = 5,
) -> ConvergenceResult:
    scored: List[tuple[float, CandidateConcept]] = []
    for candidate in divergence.candidate_concepts:
        score = float(candidate.score) + _plan_bonus(candidate, plan)
        concept = index.get_concept(candidate.concept_id) or {}
        category = str(concept.get("category", ""))
        if category in {"entity", "event", "state", "quality"}:
            score += 0.04
        scored.append((round(score, 6), candidate))

    scored.sort(key=lambda item: (item[0], -item[1].depth, item[1].label), reverse=True)
    accepted_concepts: List[Dict[str, Any]] = []
    rejected_concepts: List[Dict[str, Any]] = []
    accepted_ids: set[str] = set()
    accepted_relations: List[Dict[str, Any]] = []
    rejected_relations: List[Dict[str, Any]] = []

    for final_score, candidate in scored:
        record = candidate.to_dict()
        record["final_score"] = final_score
        if len(accepted_concepts) < max_accept and candidate.concept_id not in accepted_ids:
            accepted_concepts.append(record)
            accepted_ids.add(candidate.concept_id)
            for edge in candidate.path:
                if edge not in accepted_relations:
                    accepted_relations.append(edge)
        else:
            rejected_concepts.append(record)
            for edge in candidate.path:
                if edge not in accepted_relations and edge not in rejected_relations:
                    rejected_relations.append(edge)

    return ConvergenceResult(
        accepted_concepts=accepted_concepts,
        rejected_concepts=rejected_concepts,
        accepted_relations=accepted_relations,
        rejected_relations=rejected_relations,
    )


__all__ = ["ConvergenceResult", "run_convergence_v1"]
