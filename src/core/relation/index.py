from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from .schema import canonicalize_relation


@dataclass
class RelationIndex:
    """Indexes for concept relations and light lexical lookup."""

    concepts: Dict[str, Dict[str, Any]]
    lexical_entries: Dict[str, Dict[str, Any]]
    slot_frames: Dict[str, Dict[str, Any]]
    by_source: Dict[str, List[Dict[str, Any]]]
    by_type: Dict[str, List[Dict[str, Any]]]
    target_to_sources: Dict[str, List[str]]
    label_to_concepts: Dict[str, List[str]]
    surface_to_entries: Dict[str, List[str]]
    surface_first_char: Dict[str, List[str]]
    concept_to_entries: Dict[str, List[str]]

    def get_outbound(self, concept_id: str) -> List[Dict[str, Any]]:
        return list(self.by_source.get(str(concept_id), []))

    def get_inbound_sources(self, concept_id: str) -> List[str]:
        return list(self.target_to_sources.get(str(concept_id), []))

    def get_by_type(self, relation_type: str) -> List[Dict[str, Any]]:
        return list(self.by_type.get(str(relation_type), []))

    def get_concept(self, concept_id: str) -> Dict[str, Any] | None:
        return self.concepts.get(str(concept_id))

    def concept_label(self, concept_id: str) -> str:
        entries = self.concept_to_entries.get(str(concept_id), [])
        if entries:
            ranked = sorted(
                entries,
                key=lambda value: (value.startswith("synthetic:"), len(value) < 2, len(value), value),
            )
            preferred = ranked[0]
            if preferred in self.lexical_entries:
                entry = self.lexical_entries[preferred]
                lemma = str(entry.get("word") or entry.get("lemma") or preferred)
                if lemma:
                    return lemma
            return preferred
        concept = self.get_concept(concept_id)
        if not concept:
            return str(concept_id)
        return str(concept.get("label") or concept_id)



def _append_unique(mapping: Dict[str, List[str]], key: str | None, value: str) -> None:
    if not key:
        return
    bucket = mapping.setdefault(str(key), [])
    if value not in bucket:
        bucket.append(value)



def _collect_surface_to_entries(
    indexes: Mapping[str, Any],
    lexical_entries: Mapping[str, Dict[str, Any]],
) -> Dict[str, List[str]]:
    provided = indexes.get("surface_to_entry")
    if isinstance(provided, Mapping):
        return {str(k): [str(v) for v in values] for k, values in provided.items() if isinstance(values, list)}

    generated: Dict[str, List[str]] = {}
    for entry_id, entry in lexical_entries.items():
        lemma = str(entry.get("lemma", entry_id))
        _append_unique(generated, lemma, entry_id)
        reading = entry.get("reading")
        if isinstance(reading, str) and reading.strip():
            _append_unique(generated, reading.strip(), entry_id)
        surface_forms = entry.get("surface_forms", [])
        if isinstance(surface_forms, list):
            for form in surface_forms:
                if isinstance(form, Mapping):
                    text = str(form.get("text", "")).strip()
                    if text:
                        _append_unique(generated, text, entry_id)
    return generated



def _build_surface_first_char(surface_to_entries: Mapping[str, List[str]]) -> Dict[str, List[str]]:
    first_char_map: Dict[str, List[str]] = {}
    for surface in surface_to_entries.keys():
        if not surface:
            continue
        _append_unique(first_char_map, surface[0], surface)
    for bucket in first_char_map.values():
        bucket.sort(key=len, reverse=True)
    return first_char_map



def build_relation_index(container: Mapping[str, Any]) -> RelationIndex:
    concepts = {
        str(key): value
        for key, value in dict(container.get("concepts", {})).items()
        if isinstance(value, Mapping)
    }
    lexical_source = container.get("lexical_entries")
    if not isinstance(lexical_source, Mapping):
        lexical_source = container.get("entries", {})
    lexical_entries = {
        str(key): value
        for key, value in dict(lexical_source).items()
        if isinstance(value, Mapping)
    }
    slot_frames = {
        str(key): value
        for key, value in dict(container.get("slot_frames", {})).items()
        if isinstance(value, Mapping)
    }
    indexes = dict(container.get("indexes", {})) if isinstance(container.get("indexes"), Mapping) else {}
    raw_concept_to_entries = indexes.get("concept_to_entries") if isinstance(indexes.get("concept_to_entries"), Mapping) else {}
    concept_to_entries: Dict[str, List[str]] = {
        str(concept_id): [str(entry_id) for entry_id in entry_ids]
        for concept_id, entry_ids in raw_concept_to_entries.items()
        if isinstance(entry_ids, list)
    }

    by_source: Dict[str, List[Dict[str, Any]]] = {}
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    target_to_sources: Dict[str, List[str]] = {}
    label_to_concepts: Dict[str, List[str]] = {}

    for concept_id, concept in concepts.items():
        label = str(concept.get("label", "")).strip()
        _append_unique(label_to_concepts, label, concept_id)
        relations = concept.get("relations", [])
        if not isinstance(relations, list):
            continue
        for raw_relation in relations:
            if not isinstance(raw_relation, Mapping):
                continue
            relation = canonicalize_relation(raw_relation)
            relation_with_source = dict(relation)
            relation_with_source["source_concept"] = concept_id
            by_source.setdefault(concept_id, []).append(relation_with_source)
            by_type.setdefault(str(relation["type"]), []).append(relation_with_source)
            _append_unique(target_to_sources, str(relation["target"]), concept_id)

    if not concept_to_entries:
        for entry_id, entry in lexical_entries.items():
            concept_ids = entry.get("concept_ids", [])
            if isinstance(concept_ids, list):
                for concept_id in concept_ids:
                    _append_unique(concept_to_entries, str(concept_id), entry_id)
    surface_to_entries = _collect_surface_to_entries(indexes, lexical_entries)
    surface_first_char = _build_surface_first_char(surface_to_entries)

    return RelationIndex(
        concepts=concepts,
        lexical_entries=lexical_entries,
        slot_frames=slot_frames,
        by_source=by_source,
        by_type=by_type,
        target_to_sources=target_to_sources,
        label_to_concepts=label_to_concepts,
        surface_to_entries=surface_to_entries,
        surface_first_char=surface_first_char,
        concept_to_entries=concept_to_entries,
    )


__all__ = ["RelationIndex", "build_relation_index"]
