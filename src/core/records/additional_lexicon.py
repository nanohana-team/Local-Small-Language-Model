from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.core.io.lsd_lexicon import load_lexicon_container, save_lexicon_container
from src.core.relation.index import RelationIndex, build_relation_index

try:
    JST = ZoneInfo("Asia/Tokyo")
except ZoneInfoNotFoundError:
    JST = timezone(timedelta(hours=9))

_ALLOWED_POS = {
    "noun": "noun",
    "名詞": "noun",
    "verb": "verb",
    "動詞": "verb",
    "adjective_i": "adjective_i",
    "形容詞": "adjective_i",
    "i_adjective": "adjective_i",
    "adjective_na": "adjective_stem",
    "形容動詞": "adjective_stem",
    "na_adjective": "adjective_stem",
    "adverb": "adverb",
    "副詞": "adverb",
    "interjection": "interjection",
    "感動詞": "interjection",
    "prefix": "prefix",
    "接頭辞": "prefix",
    "suffix": "suffix",
    "接尾辞": "suffix",
    "adnominal": "adnominal",
    "連体詞": "adnominal",
}

_ALLOWED_CATEGORIES = {
    "abstract": "abstract",
    "概念": "abstract",
    "term": "abstract",
    "technical_term": "abstract",
    "entity": "entity",
    "物": "entity",
    "object": "entity",
    "event": "event",
    "出来事": "event",
    "action": "event",
    "state": "state",
    "状態": "state",
    "quality": "quality",
    "性質": "quality",
    "generated": "generated",
}


@dataclass
class AdditionalLexiconUpdate:
    requested_terms: List[str]
    added_terms: List[str]
    skipped_terms: List[str]
    persisted_path: str | None
    provider: str | None = None
    model: str | None = None
    error: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requested_terms": list(self.requested_terms),
            "added_terms": list(self.added_terms),
            "skipped_terms": list(self.skipped_terms),
            "persisted_path": self.persisted_path,
            "provider": self.provider,
            "model": self.model,
            "error": self.error,
        }


class AdditionalLexiconStore:
    """Runtime overlay lexicon for LLM-enriched unknown words.

    The store intentionally keeps generated entries separate from the main
    lexicon so loop-learning can mine unknown terms without contaminating the
    base dictionary. The generated overlay can still be merged into the in-memory
    relation index for subsequent episodes in the same run.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.container = self._load_or_init(self.path)
        self.index = build_relation_index(self.container)

    @staticmethod
    def _load_or_init(path: Path) -> Dict[str, Any]:
        if path.exists():
            return load_lexicon_container(path)
        return {
            "meta": {
                "schema_version": "v4_additional_dict_v1",
                "version": "v4_additional_dict_v1",
                "description": "Runtime overlay lexicon generated from loop-learning unknown words.",
                "created_at_jst": datetime.now(JST).isoformat(),
                "semantic_axes": [],
            },
            "concepts": {},
            "slot_frames": {},
            "lexical_entries": {},
            "indexes": {},
        }

    def known_surfaces(self) -> set[str]:
        return set(self.index.surface_to_entries.keys())

    def has_surface(self, surface: str) -> bool:
        normalized = _normalize_term(surface)
        return bool(normalized) and normalized in self.index.surface_to_entries

    def unseen_terms(self, terms: Iterable[str], *, base_index: RelationIndex | None = None) -> List[str]:
        ordered: List[str] = []
        seen: set[str] = set()
        base_surfaces = set(base_index.surface_to_entries.keys()) if base_index is not None else set()
        overlay_surfaces = self.known_surfaces()
        for raw_term in terms:
            term = _normalize_term(raw_term)
            if not term or term in seen:
                continue
            seen.add(term)
            if term in base_surfaces or term in overlay_surfaces:
                continue
            ordered.append(term)
        return ordered

    def apply_llm_entries(
        self,
        *,
        requested_terms: Sequence[str],
        parsed_payload: Mapping[str, Any] | None,
        provider: str | None = None,
        model: str | None = None,
        prompt_version: str | None = None,
        raw_text: str | None = None,
        error: str | None = None,
    ) -> AdditionalLexiconUpdate:
        requested = [term for term in (_normalize_term(term) for term in requested_terms) if term]
        if not requested:
            return AdditionalLexiconUpdate([], [], [], None, provider=provider, model=model, error=error)

        requested_set = set(requested)
        parsed_entries = _extract_generated_entries(parsed_payload)
        added_terms: List[str] = []
        skipped_terms: List[str] = []
        now = datetime.now(JST).isoformat()

        concepts = self.container.setdefault("concepts", {})
        lexical_entries = self.container.setdefault("lexical_entries", {})
        meta = self.container.setdefault("meta", {})
        if not isinstance(concepts, dict) or not isinstance(lexical_entries, dict) or not isinstance(meta, dict):
            raise TypeError("Additional lexicon container is malformed")

        for item in parsed_entries:
            surface = _normalize_term(item.get("surface") or item.get("term") or item.get("word"))
            if not surface or surface not in requested_set:
                continue
            if self.has_surface(surface):
                skipped_terms.append(surface)
                continue

            entry_id = _entry_id_for_term(surface)
            concept_id = _concept_id_for_term(surface)
            if entry_id in lexical_entries:
                skipped_terms.append(surface)
                continue

            reading = _normalize_optional_text(item.get("reading"))
            pos = _normalize_pos(item.get("pos"))
            category = _normalize_category(item.get("category"), pos=pos)
            definition = _normalize_optional_text(
                item.get("short_definition")
                or item.get("definition")
                or item.get("gloss")
                or item.get("description")
            ) or f"{surface} を指す生成語彙"
            surface_forms = _build_surface_forms(surface, reading, item.get("surface_forms"))
            related_terms = _normalize_text_list(item.get("related_terms") or item.get("aliases") or item.get("synonyms"))

            concept = {
                "id": concept_id,
                "label": surface,
                "category": category,
                "description": definition,
                "relations": [],
                "meta": {
                    "source": "loop_learning_unknown_llm",
                    "provider": provider,
                    "model": model,
                    "prompt_version": prompt_version,
                    "created_at_jst": now,
                    "related_terms": related_terms,
                },
            }
            entry = {
                "id": entry_id,
                "word": surface,
                "lemma": surface,
                "reading": reading,
                "category": _entry_category_from_pos(pos),
                "hierarchy": _hierarchy_for_pos(pos),
                "vector": {},
                "grammar": _grammar_for_pos(pos),
                "surface_forms": surface_forms,
                "senses": [
                    {
                        "id": f"sense:{entry_id}",
                        "gloss": definition,
                        "concept_ids": [concept_id],
                        "priority": 1.0,
                        "usage_notes": "loop-learning unknown word enrichment",
                    }
                ],
                "style_tags": ["generated", "llm_enriched"],
                "frequency": 0.0,
                "meta": {
                    "source": "loop_learning_unknown_llm",
                    "provider": provider,
                    "model": model,
                    "prompt_version": prompt_version,
                    "created_at_jst": now,
                    "requested_surface": surface,
                },
            }
            concepts[concept_id] = concept
            lexical_entries[entry_id] = entry
            added_terms.append(surface)

        meta["updated_at_jst"] = now
        if added_terms:
            save_lexicon_container(self.path, self.container)
            self.container = load_lexicon_container(self.path)
            self.index = build_relation_index(self.container)
        return AdditionalLexiconUpdate(
            requested_terms=requested,
            added_terms=added_terms,
            skipped_terms=_dedupe_keep_order(skipped_terms),
            persisted_path=str(self.path) if added_terms else None,
            provider=provider,
            model=model,
            error=error,
        )

    def merge_into_engine(self, engine: Any) -> int:
        target_index = getattr(engine, "index", None)
        target_container = getattr(engine, "container", None)
        if not isinstance(target_index, RelationIndex) or not isinstance(target_container, dict):
            return 0

        merged = 0
        target_container.setdefault("concepts", {})
        target_container.setdefault("lexical_entries", {})
        target_container.setdefault("indexes", {})
        target_indexes = target_container["indexes"]
        if not isinstance(target_indexes, dict):
            target_indexes = {}
            target_container["indexes"] = target_indexes
        target_indexes.setdefault("surface_to_entry", {})
        target_indexes.setdefault("concept_to_entries", {})

        for concept_id, concept in self.index.concepts.items():
            if concept_id not in target_index.concepts:
                target_index.concepts[concept_id] = concept
                if isinstance(target_container.get("concepts"), dict):
                    target_container["concepts"][concept_id] = concept
                merged += 1
            label = str(concept.get("label") or "").strip()
            if label:
                _append_unique(target_index.label_to_concepts, label, concept_id)

        for entry_id, entry in self.index.lexical_entries.items():
            if entry_id not in target_index.lexical_entries:
                target_index.lexical_entries[entry_id] = entry
                if isinstance(target_container.get("lexical_entries"), dict):
                    target_container["lexical_entries"][entry_id] = entry
                merged += 1
            for surface in _collect_entry_surfaces(entry_id, entry):
                _append_unique(target_index.surface_to_entries, surface, entry_id)
                _append_unique(target_indexes["surface_to_entry"], surface, entry_id)
                _append_unique(target_index.surface_first_char, surface[:1] if surface else None, surface)

        for concept_id, entry_ids in self.index.concept_to_entries.items():
            for entry_id in entry_ids:
                _append_unique(target_index.concept_to_entries, concept_id, entry_id)
                _append_unique(target_indexes["concept_to_entries"], concept_id, entry_id)

        for bucket in target_index.surface_first_char.values():
            bucket.sort(key=len, reverse=True)
        return merged


def _append_unique(mapping: Dict[str, List[str]], key: str | None, value: str) -> None:
    if not key:
        return
    bucket = mapping.setdefault(str(key), [])
    if value not in bucket:
        bucket.append(value)



def _normalize_term(value: Any) -> str:
    return str(value or "").strip()



def _normalize_optional_text(value: Any) -> str | None:
    text = _normalize_term(value)
    return text or None



def _normalize_text_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return _dedupe_keep_order(_normalize_term(item) for item in values if _normalize_term(item))



def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered



def _entry_id_for_term(term: str) -> str:
    digest = hashlib.sha1(term.encode("utf-8")).hexdigest()[:12]
    return f"generated:{digest}"



def _concept_id_for_term(term: str) -> str:
    digest = hashlib.sha1(term.encode("utf-8")).hexdigest()[:12]
    return f"concept:generated:{digest}"



def _normalize_pos(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return "noun"
    return _ALLOWED_POS.get(normalized, _ALLOWED_POS.get(str(value or "").strip(), "noun"))



def _normalize_category(value: Any, *, pos: str) -> str:
    if value is not None:
        normalized = str(value).strip().lower()
        if normalized:
            return _ALLOWED_CATEGORIES.get(normalized, _ALLOWED_CATEGORIES.get(str(value).strip(), _default_category_from_pos(pos)))
    return _default_category_from_pos(pos)



def _default_category_from_pos(pos: str) -> str:
    if pos == "verb":
        return "event"
    if pos in {"adjective_i", "adjective_stem", "adverb"}:
        return "quality"
    if pos == "interjection":
        return "state"
    return "abstract"



def _entry_category_from_pos(pos: str) -> str:
    if pos == "verb":
        return "verb"
    if pos in {"adjective_i", "adjective_stem"}:
        return "adjective"
    if pos == "adverb":
        return "adverb"
    if pos == "interjection":
        return "interjection"
    if pos == "prefix":
        return "prefix"
    if pos == "suffix":
        return "suffix"
    if pos == "adnominal":
        return "adnominal"
    return "noun"



def _hierarchy_for_pos(pos: str) -> List[str]:
    if pos == "verb":
        return ["content_words", "verbs", "stems", "generated", "oov"]
    if pos == "adjective_i":
        return ["content_words", "adjectives", "i", "generated", "oov"]
    if pos == "adjective_stem":
        return ["content_words", "adjectives", "na", "stems", "generated", "oov"]
    if pos == "adverb":
        return ["content_words", "adverbs", "generated", "oov"]
    if pos == "interjection":
        return ["content_words", "interjections"]
    if pos == "prefix":
        return ["content_words", "prefixes"]
    if pos == "suffix":
        return ["content_words", "suffixes"]
    if pos == "adnominal":
        return ["content_words", "adnominals"]
    return ["content_words", "nouns", "generated", "oov"]



def _grammar_for_pos(pos: str) -> Dict[str, Any]:
    base = {
        "pos": pos,
        "sub_pos": "generated",
        "conjugation_type": "none",
        "conjugation_slot": "dictionary",
        "connectability": 0.8,
        "independent": True,
        "can_start": True,
        "can_end": True,
        "content_word": True,
        "function_word": False,
        "requires_prev": [],
        "requires_next": [],
        "forbid_prev": [],
        "forbid_next": [],
        "roles": ["topic"] if pos == "noun" else ["predicate"],
    }
    if pos == "verb":
        base["conjugation_type"] = "ichidan"
        base["roles"] = ["predicate"]
        base["can_start"] = False
    elif pos == "adjective_i":
        base["conjugation_type"] = "adjective_i"
        base["roles"] = ["modifier", "predicate"]
    elif pos == "adjective_stem":
        base["conjugation_type"] = "adjective_na"
        base["roles"] = ["modifier", "predicate"]
    elif pos == "adverb":
        base["roles"] = ["modifier"]
    elif pos == "interjection":
        base["roles"] = ["discourse"]
    return base



def _build_surface_forms(surface: str, reading: str | None, raw_forms: Any) -> List[Dict[str, str]]:
    forms: List[Dict[str, str]] = [{"text": surface, "kind": "lemma"}]
    if reading and reading != surface:
        forms.append({"text": reading, "kind": "reading"})
    if isinstance(raw_forms, list):
        for item in raw_forms:
            text = None
            kind = "variant"
            if isinstance(item, Mapping):
                text = _normalize_optional_text(item.get("text") or item.get("surface") or item.get("value"))
                kind = _normalize_optional_text(item.get("kind") or item.get("type")) or "variant"
            else:
                text = _normalize_optional_text(item)
            if text:
                forms.append({"text": text, "kind": kind})
    deduped: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in forms:
        key = (item["text"], item["kind"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped



def _collect_entry_surfaces(entry_id: str, entry: Mapping[str, Any]) -> List[str]:
    surfaces = [str(entry.get("word") or entry.get("lemma") or entry_id)]
    reading = _normalize_optional_text(entry.get("reading"))
    if reading:
        surfaces.append(reading)
    for form in entry.get("surface_forms", []) if isinstance(entry.get("surface_forms"), list) else []:
        if isinstance(form, Mapping):
            text = _normalize_optional_text(form.get("text") or form.get("surface") or form.get("value"))
            if text:
                surfaces.append(text)
    return _dedupe_keep_order(surfaces)



def _extract_generated_entries(parsed_payload: Mapping[str, Any] | None) -> List[Mapping[str, Any]]:
    if not isinstance(parsed_payload, Mapping):
        return []
    for key in ("entries", "items", "terms", "words"):
        value = parsed_payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


__all__ = ["AdditionalLexiconStore", "AdditionalLexiconUpdate"]
