from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from src.core.planning.plan_v1 import PlanV1, extract_unknown_focus_term
from src.core.relation.index import RelationIndex

PUNCT_RE = re.compile(r"[\s、。,.!?！？()（）\[\]{}「」『』]+")
FUNCTION_LIKE_POS_PREFIXES = (
    "particle",
    "auxiliary",
    "copula",
)
LOW_SIGNAL_CONCEPT_CATEGORIES = {
    "grammar",
    "discourse",
    "prefix",
    "suffix",
    "adnominal",
}
QUESTION_SURFACES = {"何", "なに", "どこ", "誰", "いつ", "どう", "どれ"}
ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-./]*")
KATAKANA_WORD_RE = re.compile(r"[ァ-ヶヴー]{2,}")
QUOTED_TERM_RE = re.compile(r"[「『\"'](?P<term>.+?)[」』\"']")
QUOTE_TRIM = " \t\n\r\"'「」『』()（）[]{}.,!?！？:：;；"


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
class InputAnalysisResult:
    input_features: Dict[str, Any]
    seed_matches: List[SeedMatch]
    unknown_terms: List[str]
    unknown_focus: str | None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_features": self.input_features,
            "seed_matches": [match.to_dict() for match in self.seed_matches],
            "unknown_terms": list(self.unknown_terms),
            "unknown_focus": self.unknown_focus,
        }


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



def _surface_is_quoted(text: str, start: int, end: int) -> bool:
    if start < 0 or end <= start:
        return False
    left = text[start - 1] if start > 0 else ""
    right = text[end] if end < len(text) else ""
    return (left, right) in {("「", "」"), ("『", "』"), ('"', '"'), ("'", "'")}



def _is_single_kana(surface: str) -> bool:
    if len(surface) != 1:
        return False
    code = ord(surface)
    return (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF)



def _entry_seed_signal(
    surface: str,
    entry: Mapping[str, Any],
    concept_ids: Sequence[str],
    index: RelationIndex,
    *,
    reason: str,
    start: int,
    end: int,
    text: str,
) -> float:
    score = 1.0
    grammar = entry.get("grammar", {}) if isinstance(entry.get("grammar"), Mapping) else {}
    pos = str(grammar.get("pos", ""))
    if len(surface) == 1:
        score *= 0.88
    if pos and any(pos.startswith(prefix) for prefix in FUNCTION_LIKE_POS_PREFIXES):
        score *= 0.25
    elif grammar.get("function_word") is True:
        score *= 0.35
    elif grammar.get("content_word") is False:
        score *= 0.45

    concept_categories = {
        str((index.get_concept(concept_id) or {}).get("category", ""))
        for concept_id in concept_ids
        if concept_id in index.concepts
    }
    concept_categories.discard("")
    if concept_categories and concept_categories <= LOW_SIGNAL_CONCEPT_CATEGORIES:
        score *= 0.18
    elif "grammar" in concept_categories:
        score *= 0.30

    if len(surface) <= 2 and all(
        str((index.get_concept(concept_id) or {}).get("meta", {}).get("seed_tier", "")) == "synthetic"
        for concept_id in concept_ids
        if concept_id in index.concepts
    ):
        score *= 0.75

    if _surface_is_quoted(text, start, end):
        score = max(score, 0.60)

    if reason == "exact_surface":
        score = max(score, 0.60)

    return max(0.0, min(score, 1.0))



def _match_seeds(text: str, index: RelationIndex, *, max_matches: int = 16) -> List[SeedMatch]:
    seeds: Dict[tuple[str, str | None, int, int], SeedMatch] = {}
    normalized = text.strip()

    def add_entry_seed(surface: str, entry_id: str, reason: str, base_score: float, start: int, end: int) -> None:
        entry = index.lexical_entries.get(entry_id, {})
        concepts = _collect_entry_concepts(entry)
        if not concepts:
            return
        quoted = _surface_is_quoted(normalized, start, end)
        if surface in QUESTION_SURFACES and reason != "exact_surface":
            return
        if _is_single_kana(surface) and reason != "exact_surface" and not quoted:
            return
        signal = _entry_seed_signal(
            surface,
            entry,
            concepts,
            index,
            reason=reason,
            start=start,
            end=end,
            text=normalized,
        )
        score = round(base_score * signal, 6)
        if score <= 0.0:
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
        concept_categories = {
            str((index.get_concept(concept_id) or {}).get("category", ""))
            for concept_id in concept_ids
            if concept_id in index.concepts
        }
        concept_categories.discard("")
        if concept_categories and concept_categories <= LOW_SIGNAL_CONCEPT_CATEGORIES and label != normalized:
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



def _span_overlaps_seed(start: int, end: int, seed_matches: Sequence[SeedMatch]) -> bool:
    return any(not (end <= seed.start or start >= seed.end) for seed in seed_matches if seed.start >= 0 and seed.end >= 0)



def _normalize_unknown_term(term: str) -> str:
    cleaned = str(term or "").strip(QUOTE_TRIM)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned



def _collect_unknown_terms(text: str, seed_matches: Sequence[SeedMatch]) -> List[str]:
    normalized = str(text).strip()
    candidates: List[tuple[int, str]] = []
    explicit_focus = extract_unknown_focus_term(normalized)
    if explicit_focus:
        candidates.append((0, explicit_focus))

    for match in QUOTED_TERM_RE.finditer(normalized):
        term = _normalize_unknown_term(match.group("term"))
        if len(term) >= 2 and not _span_overlaps_seed(match.start("term"), match.end("term"), seed_matches):
            candidates.append((match.start("term"), term))

    for regex in (ASCII_WORD_RE, KATAKANA_WORD_RE):
        for match in regex.finditer(normalized):
            term = _normalize_unknown_term(match.group(0))
            if len(term) < 2:
                continue
            if term in {"ナニ", "ドウ", "ナゼ"}:
                continue
            if _span_overlaps_seed(match.start(), match.end(), seed_matches):
                continue
            candidates.append((match.start(), term))

    ordered: List[str] = []
    seen: set[str] = set()
    for _, term in sorted(candidates, key=lambda item: (item[0], -len(item[1]))):
        if term not in seen:
            seen.add(term)
            ordered.append(term)
    return ordered[:8]



def analyze_input_v1(text: str, index: RelationIndex) -> InputAnalysisResult:
    seed_matches = _match_seeds(text, index)
    unknown_terms = _collect_unknown_terms(text, seed_matches)
    unknown_focus = unknown_terms[0] if unknown_terms else None

    if seed_matches:
        unknown_words = unknown_terms
    else:
        unknown_words = unknown_terms or _tokenize(text)

    seed_concepts: List[str] = []
    for match in seed_matches:
        for concept_id in match.concept_ids:
            if concept_id in index.concepts and concept_id not in seed_concepts:
                seed_concepts.append(concept_id)

    input_features = {
        "raw_input": text,
        "normalized_input": text.strip(),
        "unknown_words": unknown_words,
        "unknown_focus": unknown_focus,
        "unknown_term_candidates": unknown_terms,
        "detected_topics": [],
        "seed_concepts": seed_concepts,
        "seed_surfaces": [
            {"surface": match.surface, "reason": match.reason, "score": match.score}
            for match in seed_matches[:8]
        ],
    }
    return InputAnalysisResult(
        input_features=input_features,
        seed_matches=seed_matches,
        unknown_terms=unknown_terms,
        unknown_focus=unknown_focus,
    )


def run_divergence_v1(
    text: str,
    plan: PlanV1,
    index: RelationIndex,
    *,
    depth_budget: int = 2,
    branching_budget: int = 6,
    input_analysis: InputAnalysisResult | None = None,
) -> DivergenceResult:
    analysis = input_analysis or analyze_input_v1(text, index)
    seed_matches = list(analysis.seed_matches)

    seed_concepts: List[str] = []
    for concept_id in analysis.input_features.get("seed_concepts", []):
        concept_id = str(concept_id)
        if concept_id in index.concepts and concept_id not in seed_concepts:
            seed_concepts.append(concept_id)

    candidate_map: Dict[str, CandidateConcept] = {}
    explored_relations: List[Dict[str, Any]] = []

    for seed_match in seed_matches:
        for seed_concept in seed_match.concept_ids:
            if seed_concept not in index.concepts:
                continue
            label = seed_match.surface or index.concept_label(seed_concept)
            existing = candidate_map.get(seed_concept)
            seed_score = round(float(seed_match.score), 6)
            if existing is None or seed_score > existing.score:
                candidate_map[seed_concept] = CandidateConcept(
                    concept_id=seed_concept,
                    label=label,
                    score=seed_score,
                    depth=0,
                    seed_concept=seed_concept,
                )

    frontier = [
        (seed_concept, seed_concept, 0, candidate.score, [])
        for seed_concept, candidate in candidate_map.items()
        if candidate.depth == 0
    ]
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

    input_features = dict(analysis.input_features)
    input_features["detected_topics"] = [candidate.label for candidate in ordered_candidates[:5]]

    return DivergenceResult(
        input_features=input_features,
        seed_matches=seed_matches,
        candidate_concepts=ordered_candidates,
        explored_relations=explored_relations,
    )


__all__ = [
    "CandidateConcept",
    "DivergenceResult",
    "InputAnalysisResult",
    "SeedMatch",
    "analyze_input_v1",
    "run_divergence_v1",
]
