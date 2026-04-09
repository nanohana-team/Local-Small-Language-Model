from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from src.core.planning.plan_v1 import PlanV1
from src.core.relation.index import RelationIndex

PUNCT_RE = re.compile(r"[\s、。,.!?！？()（）\[\]{}「」『』]+")


@dataclass
class SeedMatch:
    surface: str
    entry_id: str | None
    concept_ids: List[str]
    reason: str
    score: float
    start: int = -1
    end: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateConcept:
    concept_id: str
    label: str
    score: float
    depth: int
    seed_concept: str
    via_relation_types: List[str] = field(default_factory=list)
    path: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DivergenceResult:
    input_features: Dict[str, Any]
    seed_matches: List[SeedMatch]
    candidate_concepts: List[CandidateConcept]
    explored_relations: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_features": self.input_features,
            "seed_matches": [match.to_dict() for match in self.seed_matches],
            "candidate_concepts": [candidate.to_dict() for candidate in self.candidate_concepts],
            "explored_relations": self.explored_relations,
        }



def _append_seed(seeds: Dict[tuple[str, str | None, int, int], SeedMatch], seed: SeedMatch) -> None:
    key = (seed.surface, seed.entry_id, seed.start, seed.end)
    existing = seeds.get(key)
    if existing is None or seed.score > existing.score:
        seeds[key] = seed



def _collect_entry_concepts(entry: Mapping[str, Any]) -> List[str]:
    concepts: List[str] = []
    if isinstance(entry.get("concept_ids"), list):
        concepts.extend(str(v) for v in entry.get("concept_ids", []))
    senses = entry.get("senses", [])
    if isinstance(senses, list):
        for sense in senses:
            if isinstance(sense, Mapping) and isinstance(sense.get("concept_ids"), list):
                concepts.extend(str(v) for v in sense.get("concept_ids", []))
    deduped: List[str] = []
    seen: set[str] = set()
    for concept_id in concepts:
        if concept_id not in seen:
            seen.add(concept_id)
            deduped.append(concept_id)
    return deduped



def _tokenize(text: str) -> List[str]:
    return [token for token in PUNCT_RE.split(text) if token]



def _match_seeds(text: str, index: RelationIndex, *, max_matches: int = 16) -> List[SeedMatch]:
    seeds: Dict[tuple[str, str | None, int, int], SeedMatch] = {}
    normalized = text.strip()

    def add_entry_seed(surface: str, entry_id: str, reason: str, score: float, start: int, end: int) -> None:
        entry = index.lexical_entries.get(entry_id, {})
        concepts = _collect_entry_concepts(entry)
        if not concepts:
            return
        grammar = entry.get("grammar", {}) if isinstance(entry, Mapping) else {}
        pos = str(grammar.get("pos", ""))
        if len(surface) == 1 and pos in {"particle_case", "particle_binding", "particle_conjunctive", "particle_sentence_final", "auxiliary", "copula"}:
            return
        if surface in {"何", "なに", "どこ", "誰", "いつ", "どう", "どれ"} and reason != "exact_surface":
            return
        _append_seed(
            seeds,
            SeedMatch(
                surface=surface,
                entry_id=entry_id,
                concept_ids=concepts,
                reason=reason,
                score=score,
                start=start,
                end=end,
            ),
        )

    if normalized in index.surface_to_entries:
        for entry_id in index.surface_to_entries[normalized]:
            add_entry_seed(normalized, entry_id, "exact_surface", 1.0, 0, len(normalized))

    for token in _tokenize(normalized):
        if token in index.surface_to_entries:
            start = normalized.find(token)
            if start >= 0:
                end = start + len(token)
                for entry_id in index.surface_to_entries[token]:
                    add_entry_seed(token, entry_id, "token_surface", min(0.96, 0.70 + 0.03 * len(token)), start, end)

    inspected_surfaces: set[str] = set()
    for char in sorted(set(normalized)):
        for surface in index.surface_first_char.get(char, []):
            if surface in inspected_surfaces:
                continue
            inspected_surfaces.add(surface)
            search_from = 0
            while True:
                start = normalized.find(surface, search_from)
                if start < 0:
                    break
                end = start + len(surface)
                search_from = start + 1
                for entry_id in index.surface_to_entries.get(surface, []):
                    add_entry_seed(surface, entry_id, "substring_surface", min(0.94, 0.50 + 0.04 * len(surface)), start, end)

    for label, concept_ids in index.label_to_concepts.items():
        if not label or (len(label) <= 1 and label != normalized):
            continue
        search_from = 0
        while True:
            start = normalized.find(label, search_from)
            if start < 0:
                break
            end = start + len(label)
            search_from = start + 1
            for concept_id in concept_ids:
                _append_seed(
                    seeds,
                    SeedMatch(
                        surface=label,
                        entry_id=None,
                        concept_ids=[concept_id],
                        reason="concept_label",
                        score=min(0.92, 0.52 + 0.04 * len(label)),
                        start=start,
                        end=end,
                    ),
                )

    ordered = sorted(
        seeds.values(),
        key=lambda item: (item.score, len(item.surface), len(item.concept_ids)),
        reverse=True,
    )

    chosen: List[SeedMatch] = []
    occupied: List[tuple[int, int]] = []
    for item in ordered:
        overlaps = any(not (item.end <= start or item.start >= end) for start, end in occupied if item.start >= 0 and item.end >= 0)
        if overlaps:
            continue
        chosen.append(item)
        if item.start >= 0 and item.end >= 0:
            occupied.append((item.start, item.end))
        if len(chosen) >= max_matches:
            break

    return chosen



def _priority_weight(relation_type: str, priority: Sequence[str]) -> float:
    try:
        position = list(priority).index(relation_type)
    except ValueError:
        return 0.62
    return max(0.78, 1.20 - 0.08 * position)



def run_divergence_v1(
    text: str,
    plan: PlanV1,
    index: RelationIndex,
    *,
    depth_budget: int = 2,
    branching_budget: int = 6,
) -> DivergenceResult:
    seed_matches = _match_seeds(text, index)
    unknown_words = [] if seed_matches else _tokenize(text)

    seed_concepts: List[str] = []
    for match in seed_matches:
        for concept_id in match.concept_ids:
            if concept_id in index.concepts and concept_id not in seed_concepts:
                seed_concepts.append(concept_id)

    candidate_map: Dict[str, CandidateConcept] = {}
    explored_relations: List[Dict[str, Any]] = []

    for seed_concept in seed_concepts:
        label = index.concept_label(seed_concept)
        candidate_map[seed_concept] = CandidateConcept(
            concept_id=seed_concept,
            label=label,
            score=1.0,
            depth=0,
            seed_concept=seed_concept,
        )

    frontier = [(seed_concept, seed_concept, 0, 1.0, []) for seed_concept in seed_concepts]
    while frontier:
        current_concept, root_seed, depth, parent_score, path = frontier.pop(0)
        if depth >= depth_budget:
            continue
        outbound = index.get_outbound(current_concept)
        scored_relations: List[tuple[float, Dict[str, Any]]] = []
        for relation in outbound:
            weight = float(relation.get("weight", 1.0))
            if "divergence" not in relation.get("usage_stage", []):
                weight *= 0.85
            weight *= _priority_weight(str(relation.get("type")), plan.relation_type_priority)
            scored_relations.append((weight, relation))

        scored_relations.sort(key=lambda item: item[0], reverse=True)
        for relation_score, relation in scored_relations[:branching_budget]:
            target = str(relation["target"])
            if target not in index.concepts:
                continue
            next_depth = depth + 1
            candidate_score = parent_score * relation_score * (0.90 ** depth)
            relation_record = {
                "from": current_concept,
                "type": str(relation["type"]),
                "to": target,
                "weight": round(float(relation.get("weight", 1.0)), 6),
                "reason": f"priority:{str(relation['type'])}",
                "depth": next_depth,
                "seed": root_seed,
            }
            explored_relations.append(relation_record)

            next_path = list(path) + [relation_record]
            label = index.concept_label(target)
            existing = candidate_map.get(target)
            via_types = [str(edge["type"]) for edge in next_path]
            candidate = CandidateConcept(
                concept_id=target,
                label=label,
                score=round(candidate_score, 6),
                depth=next_depth,
                seed_concept=root_seed,
                via_relation_types=via_types,
                path=next_path,
            )
            if existing is None or candidate.score > existing.score:
                candidate_map[target] = candidate
                frontier.append((target, root_seed, next_depth, candidate.score, next_path))

    ordered_candidates = sorted(
        candidate_map.values(),
        key=lambda candidate: (candidate.score, -candidate.depth, candidate.label),
        reverse=True,
    )

    input_features = {
        "raw_input": text,
        "normalized_input": text.strip(),
        "unknown_words": unknown_words,
        "detected_topics": [candidate.label for candidate in ordered_candidates[:5]],
        "seed_concepts": seed_concepts,
    }

    return DivergenceResult(
        input_features=input_features,
        seed_matches=seed_matches,
        candidate_concepts=ordered_candidates,
        explored_relations=explored_relations,
    )


__all__ = [
    "CandidateConcept",
    "DivergenceResult",
    "SeedMatch",
    "run_divergence_v1",
]
