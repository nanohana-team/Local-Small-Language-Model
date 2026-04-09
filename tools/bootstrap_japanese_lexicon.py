from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.io.lsd_lexicon import (
    RELATION_TYPE_RULES,
    export_lexical_entries_lexicon_container,
    load_lexicon_container,
    normalize_lexicon_container,
    save_lexicon_container,
    stable_json_dumps,
    validate_lexicon_container,
    validate_raw_lexicon_container,
)

JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f々〆ヵヶー]")
WHITESPACE_RE = re.compile(r"\s+")
ASCII_RE = re.compile(r"^[\x00-\x7F]+$")
ONLY_SYMBOL_RE = re.compile(r"^[\W_]+$", re.UNICODE)
NUMERIC_RE = re.compile(r"^[0-9０-９.,]+$")

DEFAULT_SEMANTIC_AXES = [
    "valence",
    "arousal",
    "abstractness",
    "sociality",
    "temporality",
    "agency",
    "causality",
    "certainty",
    "deixis",
    "discourse_force",
]

WORDNET_POS_TO_CATEGORY = {"n": "entity", "v": "event", "a": "attribute", "s": "attribute", "r": "modifier"}
WORDNET_POS_TO_GRAMMAR = {"n": "noun", "v": "verb", "a": "adjective_i", "s": "adjective_i", "r": "adverb"}

DEFAULT_SLOT_FRAMES: Dict[str, Dict[str, Any]] = {
    "slot_frame:event_basic": {
        "id": "slot_frame:event_basic",
        "slots": [
            {"name": "actor", "required": False},
            {"name": "target", "required": False},
            {"name": "location", "required": False},
            {"name": "time", "required": False},
            {"name": "cause", "required": False},
        ],
    },
    "slot_frame:state_basic": {
        "id": "slot_frame:state_basic",
        "slots": [
            {"name": "subject", "required": False},
            {"name": "target", "required": False},
            {"name": "degree", "required": False},
            {"name": "time", "required": False},
        ],
    },
}

@dataclass
class BuildStats:
    wordnet_synsets: int = 0
    wordnet_lemma_forms: int = 0
    sudachi_tokens: int = 0
    unidic_tokens: int = 0
    corpus_files: int = 0
    seed_entries: int = 0
    accepted_entries: int = 0
    review_entries: int = 0
    rejected_entries: int = 0
    relation_count: int = 0
    dangling_relations: int = 0
    pruned_relations: int = 0
    default_seed_loaded: bool = False

@dataclass
class PromotionPolicy:
    name: str
    min_source_support: int
    min_candidate_hits: int
    min_corpus_hits: int
    allow_function_word_auto: bool
    allow_single_char_auto: bool
    allow_ascii_auto: bool
    promote_corpus_only_entries: bool
    max_surface_forms: int
    review_limit: int

@dataclass
class PromotionDecision:
    state: str
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"state": self.state, "reasons": list(self.reasons)}

@dataclass
class EntryAccumulator:
    lemma: str
    entry_id: str
    category: str
    grammar: Dict[str, Any]
    reading: str | None = None
    slot_frame_id: str | None = None
    surface_forms: "OrderedDict[tuple[str, str], Dict[str, Any]]" = field(default_factory=OrderedDict)
    senses: "OrderedDict[str, Dict[str, Any]]" = field(default_factory=OrderedDict)
    concept_ids: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=lambda: ["neutral"])
    frequency: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    source_counts: Dict[str, int] = field(default_factory=dict)
    analysis_candidate_hits: int = 0
    corpus_hits: int = 0
    seed_locked: bool = False
    wordnet_backed: bool = False

    def add_surface_form(self, text: str | None, kind: str, **extras: Any) -> None:
        normalized = normalize_surface(text)
        if not normalized:
            return
        key = (normalized, kind)
        if key not in self.surface_forms:
            payload: Dict[str, Any] = {"text": normalized, "kind": kind}
            payload.update({k: v for k, v in extras.items() if v not in (None, "", [], {})})
            self.surface_forms[key] = payload

    def add_concept(self, concept_id: str | None) -> None:
        if concept_id and concept_id not in self.concept_ids:
            self.concept_ids.append(concept_id)

    def add_sense(self, sense: Mapping[str, Any]) -> None:
        sense_id = str(sense.get("id", "")).strip()
        if not sense_id:
            return
        if sense_id not in self.senses:
            self.senses[sense_id] = dict(sense)
        for concept_id in sense.get("concept_ids", []):
            self.add_concept(str(concept_id))

    def merge_grammar(self, incoming: Mapping[str, Any]) -> None:
        for key, value in incoming.items():
            if key not in self.grammar or self.grammar.get(key) in (None, "", [], {}, False):
                self.grammar[key] = value
        if incoming.get("content_word"):
            self.grammar["content_word"] = True
            self.grammar["function_word"] = False

    def record_source(self, source: str, evidence_kind: str | None = None) -> None:
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        if evidence_kind == "candidate":
            self.analysis_candidate_hits += 1
        elif evidence_kind == "corpus":
            self.corpus_hits += 1

    @property
    def source_support(self) -> int:
        return len([name for name, count in self.source_counts.items() if count > 0 and name != "seed"])

    @property
    def is_function_word(self) -> bool:
        return bool(self.grammar.get("function_word"))

    @property
    def evidence_score(self) -> float:
        score = float(self.frequency)
        score += float(self.analysis_candidate_hits) * 0.5
        score += float(self.corpus_hits) * 1.5
        if self.seed_locked:
            score += 2.0
        if self.wordnet_backed:
            score += 1.0
        return score

    def to_payload(self, decision: PromotionDecision, *, max_evidence_score: float, max_surface_forms: int) -> Dict[str, Any]:
        normalized_frequency = 0.0
        if max_evidence_score > 0.0:
            normalized_frequency = min(1.0, self.evidence_score / max_evidence_score)
        normalized_frequency = max(normalized_frequency, min(1.0, float(self.frequency)))
        meta = dict(self.meta)
        meta["bootstrap"] = {
            "sources": sorted(self.source_counts.keys()),
            "source_counts": dict(sorted(self.source_counts.items())),
            "source_support": self.source_support,
            "analysis_candidate_hits": self.analysis_candidate_hits,
            "corpus_hits": self.corpus_hits,
            "seed_locked": self.seed_locked,
            "wordnet_backed": self.wordnet_backed,
            "promotion": decision.as_dict(),
        }
        payload: Dict[str, Any] = {
            "id": self.entry_id,
            "lemma": self.lemma,
            "surface_forms": list(self.surface_forms.values())[:max_surface_forms] or [{"text": self.lemma, "kind": "lemma"}],
            "grammar": self.grammar,
            "senses": list(self.senses.values()),
            "style_tags": list(dict.fromkeys(self.style_tags)),
            "frequency": round(normalized_frequency, 6),
            "meta": meta,
            "category": self.category,
        }
        if self.reading:
            payload["reading"] = self.reading
        if self.slot_frame_id:
            payload["slot_frame_id"] = self.slot_frame_id
        if self.concept_ids:
            payload["concept_ids"] = list(self.concept_ids)
        return payload

class LexiconBuilder:
    def __init__(self, policy: PromotionPolicy) -> None:
        self.policy = policy
        self.entries: "OrderedDict[str, EntryAccumulator]" = OrderedDict()
        self.concepts: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.slot_frames: Dict[str, Dict[str, Any]] = dict(DEFAULT_SLOT_FRAMES)
        self.stats = BuildStats()

    def _entry_key(self, lemma: str, grammar_pos: str) -> str:
        return f"lex:{lemma}:{grammar_pos}"

    def get_or_create_entry(self, *, lemma: str, category: str, grammar: Mapping[str, Any], reading: str | None = None, slot_frame_id: str | None = None) -> EntryAccumulator:
        grammar_pos = str(grammar.get("pos", "unknown"))
        entry_key = self._entry_key(lemma, grammar_pos)
        if entry_key not in self.entries:
            entry = EntryAccumulator(lemma=lemma, entry_id=entry_key, category=category, grammar=dict(grammar), reading=reading, slot_frame_id=slot_frame_id, meta={"sources": []})
            entry.add_surface_form(lemma, "lemma")
            self.entries[entry_key] = entry
        entry = self.entries[entry_key]
        if reading and not entry.reading:
            entry.reading = reading
        if slot_frame_id and not entry.slot_frame_id:
            entry.slot_frame_id = slot_frame_id
        entry.merge_grammar(grammar)
        return entry

    def register_seed_entry(self, payload: Mapping[str, Any]) -> None:
        lemma = normalize_surface(payload.get("lemma") or payload.get("word"))
        if not lemma:
            return
        grammar = build_base_grammar(str(payload.get("grammar", {}).get("pos", payload.get("category", "noun"))))
        if isinstance(payload.get("grammar"), Mapping):
            grammar.update(dict(payload.get("grammar", {})))
        entry = self.get_or_create_entry(
            lemma=lemma,
            category=str(payload.get("category", grammar.get("pos", "noun"))),
            grammar=grammar,
            reading=normalize_surface(payload.get("reading")),
            slot_frame_id=normalize_identifier(payload.get("slot_frame_id")),
        )
        for form in payload.get("surface_forms", []) if isinstance(payload.get("surface_forms"), list) else []:
            if isinstance(form, Mapping):
                entry.add_surface_form(form.get("text"), str(form.get("kind", "variant")))
        for sense in payload.get("senses", []) if isinstance(payload.get("senses"), list) else []:
            if isinstance(sense, Mapping):
                entry.add_sense(sense)
        for concept_id in payload.get("concept_ids", []) if isinstance(payload.get("concept_ids"), list) else []:
            entry.add_concept(str(concept_id))
        entry.frequency = max(entry.frequency, safe_float(payload.get("frequency", 0.0)))
        entry.seed_locked = True
        entry.record_source("seed")
        entry.meta.setdefault("sources", []).append("seed")
        merge_meta(entry.meta, payload.get("meta"))
        self.stats.seed_entries += 1

    def register_concept(self, concept_id: str, payload: Mapping[str, Any]) -> None:
        if concept_id not in self.concepts:
            self.concepts[concept_id] = dict(payload)

    def add_wordnet_entry(self, lemma: str, pos: str, sense: Mapping[str, Any], reading: str | None = None) -> None:
        grammar_pos = WORDNET_POS_TO_GRAMMAR.get(pos, "noun")
        entry = self.get_or_create_entry(
            lemma=lemma,
            category=WORDNET_POS_TO_CATEGORY.get(pos, "entity"),
            grammar=build_base_grammar(grammar_pos),
            reading=reading,
            slot_frame_id=default_slot_frame_for_pos(grammar_pos),
        )
        entry.add_sense(sense)
        entry.add_surface_form(lemma, "wordnet_lemma")
        entry.record_source("wordnet")
        entry.wordnet_backed = True
        entry.meta.setdefault("sources", []).append("wordnet")
        entry.frequency += 1.0
        self.stats.wordnet_lemma_forms += 1

    def add_morpheme(self, *, lemma: str, surface: str, normalized_form: str | None, reading: str | None, pos_fields: Sequence[str], source: str, evidence_kind: str, concept_ids: Sequence[str] | None = None, extra_meta: Mapping[str, Any] | None = None) -> None:
        grammar_pos = map_pos_to_v4(pos_fields)
        grammar = build_base_grammar(grammar_pos)
        grammar["raw_pos"] = [str(v) for v in pos_fields]
        if len(pos_fields) >= 5 and pos_fields[4] not in {"", "*"}:
            grammar["conjugation_type"] = str(pos_fields[4])
        if len(pos_fields) >= 6 and pos_fields[5] not in {"", "*"}:
            grammar["conjugation_slot"] = str(pos_fields[5])
        entry = self.get_or_create_entry(lemma=lemma, category=grammar_pos, grammar=grammar, reading=reading, slot_frame_id=default_slot_frame_for_pos(grammar_pos))
        entry.add_surface_form(lemma, "lemma")
        entry.add_surface_form(surface, f"{source}_surface")
        if normalized_form and normalized_form != surface:
            entry.add_surface_form(normalized_form, f"{source}_normalized")
        if reading and not entry.reading:
            entry.reading = reading
        for concept_id in concept_ids or []:
            entry.add_concept(concept_id)
        entry.record_source(source, evidence_kind=evidence_kind)
        entry.frequency += 1.0 if evidence_kind == "corpus" else 0.35
        entry.meta.setdefault("sources", []).append(source)
        if extra_meta:
            merge_meta(entry.meta, extra_meta)
        if source == "sudachi":
            self.stats.sudachi_tokens += 1
        elif source == "unidic":
            self.stats.unidic_tokens += 1

    def decide_entry(self, entry: EntryAccumulator) -> PromotionDecision:
        if entry.seed_locked:
            return PromotionDecision("accepted", ["seed_locked"])
        if entry.wordnet_backed or entry.concept_ids or entry.senses:
            return PromotionDecision("accepted", ["concept_backed"])
        if looks_suspicious_surface(entry.lemma, allow_ascii=self.policy.allow_ascii_auto):
            return PromotionDecision("rejected", ["suspicious_surface"])
        reasons: List[str] = []
        if entry.is_function_word and not self.policy.allow_function_word_auto:
            reasons.append("function_word_requires_manual_review")
        if len(entry.lemma) == 1 and not self.policy.allow_single_char_auto:
            reasons.append("single_character_requires_manual_review")
        if entry.source_support < self.policy.min_source_support:
            reasons.append(f"low_source_support:{entry.source_support}")
        candidate_ok = entry.analysis_candidate_hits >= self.policy.min_candidate_hits
        corpus_ok = entry.corpus_hits >= self.policy.min_corpus_hits
        if not candidate_ok and not corpus_ok:
            reasons.append(f"insufficient_evidence:candidate={entry.analysis_candidate_hits},corpus={entry.corpus_hits}")
        if entry.corpus_hits > 0 and not self.policy.promote_corpus_only_entries and not candidate_ok:
            reasons.append("corpus_only_entries_disabled")
        if reasons:
            return PromotionDecision("review", reasons)
        return PromotionDecision("accepted", ["morphology_supported"])

    def _collect_report_item(self, entry: EntryAccumulator, decision: PromotionDecision) -> Dict[str, Any]:
        return {"entry_id": entry.entry_id, "lemma": entry.lemma, "pos": entry.grammar.get("pos", "unknown"), "source_support": entry.source_support, "analysis_candidate_hits": entry.analysis_candidate_hits, "corpus_hits": entry.corpus_hits, "sources": sorted(entry.source_counts.keys()), "reasons": list(decision.reasons)}

    def build_output(self, *, relation_target_policy: str = "prune") -> tuple[Dict[str, Any], Dict[str, Any]]:
        accepted: "OrderedDict[str, EntryAccumulator]" = OrderedDict()
        review_items: List[Dict[str, Any]] = []
        rejected_items: List[Dict[str, Any]] = []
        accepted_decisions: Dict[str, PromotionDecision] = {}
        for entry in self.entries.values():
            decision = self.decide_entry(entry)
            if decision.state == "accepted":
                accepted[entry.entry_id] = entry
                accepted_decisions[entry.entry_id] = decision
            elif decision.state == "review":
                review_items.append(self._collect_report_item(entry, decision))
            else:
                rejected_items.append(self._collect_report_item(entry, decision))
        for entry in accepted.values():
            if entry.concept_ids or entry.senses:
                continue
            concept_id = f"synthetic:{entry.grammar.get('pos', 'unknown')}:{stable_slug(entry.lemma)}"
            if concept_id not in self.concepts:
                self.concepts[concept_id] = {"id": concept_id, "label": entry.lemma, "category": "grammar" if entry.is_function_word else str(entry.category or entry.grammar.get('pos', 'entity')), "description": f"自動生成された補助 concept: {entry.lemma}", "relations": [], "meta": {"source": "bootstrap_synthetic", "generated_for_entry": entry.entry_id}}
            entry.add_concept(concept_id)
            entry.add_sense({"id": f"sense:{entry.lemma}:{stable_slug(entry.entry_id)}", "gloss": f"自動生成された補助語義: {entry.lemma}", "concept_ids": [concept_id], "priority": 0.5, "usage_notes": "source=bootstrap_synthetic"})
        referenced_concepts = set(); referenced_slot_frames = set()
        for entry in accepted.values():
            referenced_concepts.update(entry.concept_ids)
            if entry.slot_frame_id:
                referenced_slot_frames.add(entry.slot_frame_id)
            for sense in entry.senses.values():
                referenced_concepts.update(str(cid) for cid in sense.get("concept_ids", []))
        expanded_concepts = set(referenced_concepts)
        pending = list(referenced_concepts)
        while pending:
            concept_id = pending.pop()
            concept = self.concepts.get(concept_id)
            if not isinstance(concept, Mapping):
                continue
            default_slot = normalize_identifier(concept.get("default_slot_frame_id"))
            if default_slot:
                referenced_slot_frames.add(default_slot)
            for relation in concept.get("relations", []) if isinstance(concept.get("relations"), list) else []:
                if not isinstance(relation, Mapping):
                    continue
                target = normalize_identifier(relation.get("target"))
                if target and target in self.concepts and target not in expanded_concepts:
                    expanded_concepts.add(target)
                    pending.append(target)
        pruned_concepts = OrderedDict((cid, self.concepts[cid]) for cid in self.concepts.keys() if cid in expanded_concepts)
        pruned_concepts, relation_count, dangling_relations, pruned_relations = sanitize_concept_relations(
            pruned_concepts,
            relation_target_policy=relation_target_policy,
        )
        pruned_slot_frames = OrderedDict((sid, self.slot_frames[sid]) for sid in self.slot_frames.keys() if sid in referenced_slot_frames or sid in DEFAULT_SLOT_FRAMES)
        max_evidence_score = max((entry.evidence_score for entry in accepted.values()), default=1.0)
        lexical_entries = OrderedDict((eid, entry.to_payload(accepted_decisions[eid], max_evidence_score=max_evidence_score, max_surface_forms=self.policy.max_surface_forms)) for eid, entry in accepted.items())
        self.stats.accepted_entries = len(accepted); self.stats.review_entries = len(review_items); self.stats.rejected_entries = len(rejected_items)
        self.stats.relation_count = relation_count; self.stats.dangling_relations = dangling_relations; self.stats.pruned_relations = pruned_relations
        raw_container: Dict[str, Any] = {
            "meta": {"schema_version": "v4-bridge", "build_version": "0.0.0", "semantic_axes": list(DEFAULT_SEMANTIC_AXES), "builder": "tools/bootstrap_japanese_lexicon.py", "seed": {"default_seed_loaded": self.stats.default_seed_loaded}, "bootstrap_policy": {"name": self.policy.name, "min_source_support": self.policy.min_source_support, "min_candidate_hits": self.policy.min_candidate_hits, "min_corpus_hits": self.policy.min_corpus_hits, "allow_function_word_auto": self.policy.allow_function_word_auto, "allow_single_char_auto": self.policy.allow_single_char_auto, "allow_ascii_auto": self.policy.allow_ascii_auto, "promote_corpus_only_entries": self.policy.promote_corpus_only_entries, "max_surface_forms": self.policy.max_surface_forms, "relation_target_policy": relation_target_policy}, "source_licenses": {"wordnet": "Japanese WordNet / OMW license", "sudachi": "Apache-2.0", "fugashi": "MIT and BSD-3-Clause", "unidic_lite": "MIT + UniDic BSD"}, "stats": {"seed_entries": self.stats.seed_entries, "wordnet_synsets": self.stats.wordnet_synsets, "wordnet_lemma_forms": self.stats.wordnet_lemma_forms, "sudachi_tokens": self.stats.sudachi_tokens, "unidic_tokens": self.stats.unidic_tokens, "accepted_entries": self.stats.accepted_entries, "review_entries": self.stats.review_entries, "rejected_entries": self.stats.rejected_entries, "relation_count": self.stats.relation_count, "dangling_relations": self.stats.dangling_relations, "pruned_relations": self.stats.pruned_relations, "concept_count": len(pruned_concepts), "entry_count": len(lexical_entries)}} ,
            "concepts": pruned_concepts,
            "slot_frames": pruned_slot_frames,
            "lexical_entries": lexical_entries,
            "indexes": {},
        }
        container = export_lexical_entries_lexicon_container(raw_container)
        review_payload = {"policy": raw_container["meta"]["bootstrap_policy"], "stats": raw_container["meta"]["stats"], "review": sorted(review_items, key=lambda item: (-item["source_support"], item["lemma"]))[: self.policy.review_limit], "rejected": sorted(rejected_items, key=lambda item: (-item["source_support"], item["lemma"]))[: self.policy.review_limit]}
        return container, review_payload

def normalize_surface(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("_", " ").replace("\u3000", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text or None

def looks_japanese(text: str | None) -> bool:
    return bool(text and JAPANESE_CHAR_RE.search(text))

def stable_slug(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]

def normalize_identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def make_relation_payload(
    relation_type: str,
    target: str,
    *,
    weight: float = 1.0,
    direction: str,
    layer: str,
    usage_stage: Sequence[str],
    source: str,
    confidence: float = 1.0,
    inverse_type: str | None = None,
    extra_meta: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": str(relation_type),
        "target": str(target),
        "weight": max(0.0, min(1.0, float(weight))),
        "direction": str(direction),
        "layer": str(layer),
        "usage_stage": [str(value) for value in usage_stage],
        "confidence": max(0.0, min(1.0, float(confidence))),
        "meta": {"source": str(source)},
    }
    if inverse_type:
        payload["inverse_type"] = str(inverse_type)
    if isinstance(extra_meta, Mapping):
        payload["meta"].update(dict(extra_meta))
    return payload


def sanitize_concept_relations(
    concepts: Mapping[str, Any],
    *,
    relation_target_policy: str,
) -> tuple["OrderedDict[str, Dict[str, Any]]", int, int, int]:
    known_concepts = set(str(key) for key in concepts.keys())
    cleaned: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    relation_count = 0
    dangling_relations = 0
    pruned_relations = 0
    for concept_id, raw_concept in concepts.items():
        if not isinstance(raw_concept, Mapping):
            continue
        concept = dict(raw_concept)
        relations = concept.get("relations", [])
        cleaned_relations: List[Dict[str, Any]] = []
        if isinstance(relations, list):
            concept_meta = concept.get("meta") if isinstance(concept.get("meta"), Mapping) else {}
            concept_source = normalize_identifier(concept_meta.get("source"))
            for relation in relations:
                if not isinstance(relation, Mapping):
                    continue
                relation_count += 1
                normalized_relation = dict(relation)
                relation_type = normalize_identifier(normalized_relation.get("type"))
                relation_target = normalize_identifier(normalized_relation.get("target"))
                relation_rule = RELATION_TYPE_RULES.get(relation_type or "")
                if relation_type:
                    normalized_relation["type"] = relation_type
                if relation_target:
                    normalized_relation["target"] = relation_target
                if relation_rule is not None:
                    normalized_relation.setdefault("direction", relation_rule.get("direction"))
                    normalized_relation.setdefault("layer", relation_rule.get("layer"))
                    if relation_rule.get("usage_stage"):
                        normalized_relation.setdefault("usage_stage", list(relation_rule.get("usage_stage", [])))
                    if relation_rule.get("inverse_type"):
                        normalized_relation.setdefault("inverse_type", relation_rule.get("inverse_type"))
                if "confidence" not in normalized_relation:
                    normalized_relation["confidence"] = max(0.0, min(1.0, safe_float(normalized_relation.get("weight", 1.0), 1.0)))
                meta_value = normalized_relation.get("meta")
                if not isinstance(meta_value, Mapping):
                    meta_value = {}
                meta_value = dict(meta_value)
                if "source" not in meta_value:
                    meta_value["source"] = concept_source or "seed_inherited"
                normalized_relation["meta"] = meta_value
                if relation_target and relation_target not in known_concepts:
                    dangling_relations += 1
                    if relation_target_policy == "error":
                        raise RuntimeError(f"concept {concept_id!r} has relation target {relation_target!r} outside the closed graph")
                    if relation_target_policy == "prune":
                        pruned_relations += 1
                        continue
                cleaned_relations.append(normalized_relation)
        concept["relations"] = cleaned_relations
        cleaned[str(concept_id)] = concept
    return cleaned, relation_count, dangling_relations, pruned_relations


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def merge_meta(target: Dict[str, Any], incoming: Any) -> None:
    if not isinstance(incoming, Mapping):
        return
    for key, value in incoming.items():
        if key == "sources":
            existing = target.setdefault("sources", [])
            if isinstance(existing, list):
                for item in value if isinstance(value, list) else [value]:
                    if item not in existing:
                        existing.append(item)
            continue
        if key not in target:
            target[key] = value

def build_base_grammar(pos: str) -> Dict[str, Any]:
    function_word = pos.startswith("particle") or pos in {"auxiliary", "copula", "symbol", "conjunction"}
    return {"pos": pos or "unknown", "sub_pos": "independent" if not function_word else "function", "conjugation_type": "none", "conjugation_slot": "dictionary", "connectability": 1.0, "independent": not function_word, "can_start": not function_word, "can_end": True, "content_word": not function_word, "function_word": function_word, "roles": ["predicate"] if pos in {"verb", "adjective_i", "adjective_na", "adverb"} else [], "requires_prev": [], "requires_next": [], "forbid_prev": [], "forbid_next": []}

def default_slot_frame_for_pos(pos: str) -> str | None:
    if pos == "verb":
        return "slot_frame:event_basic"
    if pos in {"adjective_i", "adjective_na"}:
        return "slot_frame:state_basic"
    return None

def map_pos_to_v4(pos_fields: Sequence[str]) -> str:
    fields = [str(v) for v in pos_fields]
    p0 = fields[0] if len(fields) > 0 else ""; p1 = fields[1] if len(fields) > 1 else ""
    if p0 == "名詞":
        return "pronoun" if p1 == "代名詞" else "noun"
    if p0 == "動詞": return "verb"
    if p0 == "形容詞": return "adjective_i"
    if p0 == "形状詞": return "adjective_na"
    if p0 == "副詞": return "adverb"
    if p0 == "連体詞": return "adnominal"
    if p0 == "接続詞": return "conjunction"
    if p0 == "感動詞": return "interjection"
    if p0 == "接頭辞": return "prefix"
    if p0 == "接尾辞": return "suffix"
    if p0 == "助動詞": return "auxiliary"
    if p0 == "助詞":
        if p1 == "格助詞": return "particle_case"
        if p1 == "係助詞": return "particle_binding"
        if p1 in {"接続助詞", "副助詞"}: return "particle_conjunctive"
        if p1 == "終助詞": return "particle_sentence_final"
        return "particle_case"
    if p0 in {"補助記号", "記号", "空白"}: return "symbol"
    return "noun"

def looks_suspicious_surface(text: str, *, allow_ascii: bool) -> bool:
    return (not text) or bool(ONLY_SYMBOL_RE.fullmatch(text)) or bool(NUMERIC_RE.fullmatch(text)) or (bool(ASCII_RE.fullmatch(text)) and not allow_ascii)

def default_seed_path() -> Path:
    return PROJECT_ROOT / "examples" / "seed_core_japanese_dictionary.json"

def read_utf8_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_seed(builder: LexiconBuilder, seed_path: Path | None) -> None:
    if seed_path is None: return
    lexical = export_lexical_entries_lexicon_container(load_lexicon_container(seed_path))
    for concept_id, concept in lexical.get("concepts", {}).items():
        if isinstance(concept, Mapping): builder.register_concept(str(concept_id), concept)
    for slot_frame_id, slot_frame in lexical.get("slot_frames", {}).items():
        if isinstance(slot_frame, Mapping): builder.slot_frames[str(slot_frame_id)] = dict(slot_frame)
    for entry in lexical.get("lexical_entries", {}).values():
        if isinstance(entry, Mapping): builder.register_seed_entry(entry)

def ensure_nltk_wordnet(download: bool) -> Any:
    try:
        import nltk
        from nltk.corpus import wordnet as wn
        try:
            wn.synset("dog.n.01"); _ = wn.synset("dog.n.01").lemma_names("jpn"); return wn
        except LookupError:
            if not download: raise
            nltk.download("wordnet", quiet=False); nltk.download("omw-1.4", quiet=False)
            from nltk.corpus import wordnet as wn2
            return wn2
    except LookupError as exc:
        raise RuntimeError("NLTK WordNet / OMW data が見つかりません。--download-nltk-data を付けるか、nltk.download('wordnet'); nltk.download('omw-1.4') を実行してください。") from exc
    except ImportError as exc:
        raise RuntimeError("nltk がインストールされていません。requirements.txt の依存関係を入れてください。") from exc

def iter_wordnet_entries(*, wn: Any, max_synsets: int, min_surface_length: int, allow_non_japanese: bool) -> Iterator[tuple[str, str, Dict[str, Any], List[tuple[str, Dict[str, Any]]]]]:
    for index, synset in enumerate(wn.all_synsets(), start=1):
        if max_synsets > 0 and index > max_synsets: break
        lemmas: List[str] = []
        for lemma_name in synset.lemma_names("jpn"):
            normalized = normalize_surface(lemma_name)
            if not normalized or len(normalized) < min_surface_length: continue
            if not allow_non_japanese and not looks_japanese(normalized): continue
            lemmas.append(normalized)
        if not lemmas: continue
        unique_lemmas = list(dict.fromkeys(lemmas))
        pos = getattr(synset, "pos")() if callable(getattr(synset, "pos", None)) else str(getattr(synset, "pos", "n"))
        concept_id = f"wordnet:{synset.name()}"
        relations = [
            make_relation_payload(
                "hypernym",
                f"wordnet:{hyper.name()}",
                weight=1.0,
                direction="outbound",
                layer="semantic",
                usage_stage=["divergence", "convergence"],
                source="jpn-wordnet",
                confidence=1.0,
                inverse_type="hyponym",
                extra_meta={"source_synset": synset.name()},
            )
            for hyper in synset.hypernyms()[:8]
        ] + [
            make_relation_payload(
                "hyponym",
                f"wordnet:{hypo.name()}",
                weight=1.0,
                direction="outbound",
                layer="semantic",
                usage_stage=["divergence", "convergence"],
                source="jpn-wordnet",
                confidence=1.0,
                inverse_type="hypernym",
                extra_meta={"source_synset": synset.name()},
            )
            for hypo in synset.hyponyms()[:8]
        ]
        concept_payload: Dict[str, Any] = {"id": concept_id, "label": unique_lemmas[0], "category": WORDNET_POS_TO_CATEGORY.get(pos, "entity"), "description": synset.definition(), "relations": relations, "meta": {"source": "jpn-wordnet", "synset": synset.name(), "lemma_count": len(unique_lemmas)}}
        slot_frame_id = default_slot_frame_for_pos(WORDNET_POS_TO_GRAMMAR.get(pos, "noun"))
        if slot_frame_id: concept_payload["default_slot_frame_id"] = slot_frame_id
        sense_pairs = []
        for lemma in unique_lemmas:
            sense_pairs.append((lemma, {"id": f"sense:{synset.name()}:{stable_slug(lemma)}", "gloss": synset.definition(), "concept_ids": [concept_id], "priority": 1.0, "usage_notes": "source=jpn-wordnet"}))
        yield concept_id, pos, concept_payload, sense_pairs

def import_wordnet(builder: LexiconBuilder, *, download_nltk_data: bool, max_synsets: int, min_surface_length: int, allow_non_japanese: bool) -> List[str]:
    wn = ensure_nltk_wordnet(download_nltk_data)
    candidate_surfaces: List[str] = []
    for concept_id, pos, concept_payload, sense_pairs in iter_wordnet_entries(wn=wn, max_synsets=max_synsets, min_surface_length=min_surface_length, allow_non_japanese=allow_non_japanese):
        builder.register_concept(concept_id, concept_payload); builder.stats.wordnet_synsets += 1
        for lemma, sense_payload in sense_pairs:
            builder.add_wordnet_entry(lemma, pos, sense_payload); candidate_surfaces.append(lemma)
    return list(dict.fromkeys(candidate_surfaces))

def extract_sudachi_token_fields(token: Any) -> tuple[str | None, str | None, str | None, str | None, List[str]]:
    surface = normalize_surface(token.surface())
    lemma = normalize_surface(token.dictionary_form()) or surface
    normalized_form = normalize_surface(token.normalized_form()) or surface
    reading = normalize_surface(token.reading_form())
    if reading in {None, "*"}: reading = None
    return lemma, surface, normalized_form, reading, [str(v) for v in token.part_of_speech()]

def import_sudachi_candidates(builder: LexiconBuilder, *, surfaces: Sequence[str], dict_type: str, split_mode: str, analyze_corpus_paths: Sequence[Path], min_surface_length: int, allow_non_japanese: bool) -> None:
    try:
        from sudachipy import Dictionary, SplitMode
    except ImportError as exc:
        raise RuntimeError("SudachiPy がインストールされていません。requirements.txt の依存関係を入れてください。") from exc
    tokenizer = Dictionary(dict=dict_type).create(); split = getattr(SplitMode, split_mode)
    def consume_text(text: str, *, evidence_kind: str) -> None:
        for token in tokenizer.tokenize(text, split):
            lemma, surface, normalized_form, reading, pos_fields = extract_sudachi_token_fields(token)
            if not lemma or not surface: continue
            if len(lemma) < min_surface_length and len(surface) < min_surface_length: continue
            if not allow_non_japanese and not (looks_japanese(lemma) or looks_japanese(surface)): continue
            builder.add_morpheme(lemma=lemma, surface=surface, normalized_form=normalized_form, reading=reading, pos_fields=pos_fields, source="sudachi", evidence_kind=evidence_kind)
    for i, surface in enumerate(dict.fromkeys(surfaces), start=1):
        consume_text(surface, evidence_kind="candidate")
        if i % 10000 == 0: print(f"[INFO] sudachi analyzed candidates={i} entries={len(builder.entries)}")
    for path in analyze_corpus_paths:
        consume_text(read_utf8_text(path), evidence_kind="corpus"); builder.stats.corpus_files += 1; print(f"[INFO] sudachi analyzed corpus={path}")

def feature_get(feature: Any, *names: str) -> str | None:
    for name in names:
        if hasattr(feature, name):
            normalized = normalize_surface(getattr(feature, name))
            if normalized not in {None, "*"}: return normalized
    return None

def import_unidic_candidates(builder: LexiconBuilder, *, surfaces: Sequence[str], dictionary_mode: str, analyze_corpus_paths: Sequence[Path], min_surface_length: int, allow_non_japanese: bool) -> None:
    try:
        import fugashi
    except ImportError as exc:
        raise RuntimeError("fugashi がインストールされていません。requirements.txt の依存関係を入れてください。") from exc
    tagger_args = ""
    if dictionary_mode == "full":
        try:
            import unidic
        except ImportError as exc:
            raise RuntimeError("full UniDic を使うには unidic パッケージを追加で入れて、python -m unidic download を実行してください。") from exc
        tagger_args = f'-d "{unidic.DICDIR}"'
    tagger = fugashi.Tagger(tagger_args)
    def consume_text(text: str, *, evidence_kind: str) -> None:
        for token in tagger(text):
            surface = normalize_surface(getattr(token, "surface", str(token)))
            if not surface: continue
            feature = getattr(token, "feature", None)
            lemma = feature_get(feature, "lemma", "lemma_form", "lForm") or surface
            reading = feature_get(feature, "kana", "kanaBase", "pron", "pronBase")
            normalized_form = feature_get(feature, "orthBase", "orth") or surface
            pos_value = getattr(token, "pos", None)
            if isinstance(pos_value, str): pos_fields = pos_value.split(",")
            elif isinstance(pos_value, Sequence): pos_fields = [str(v) for v in pos_value]
            else: pos_fields = [feature_get(feature, "pos1") or "名詞", feature_get(feature, "pos2") or "一般", feature_get(feature, "pos3") or "*", feature_get(feature, "pos4") or "*", feature_get(feature, "cType") or "*", feature_get(feature, "cForm") or "*"]
            if len(lemma) < min_surface_length and len(surface) < min_surface_length: continue
            if not allow_non_japanese and not (looks_japanese(lemma) or looks_japanese(surface)): continue
            builder.add_morpheme(lemma=lemma, surface=surface, normalized_form=normalized_form, reading=reading, pos_fields=pos_fields, source="unidic", evidence_kind=evidence_kind)
    for i, surface in enumerate(dict.fromkeys(surfaces), start=1):
        consume_text(surface, evidence_kind="candidate")
        if i % 10000 == 0: print(f"[INFO] unidic analyzed candidates={i} entries={len(builder.entries)}")
    for path in analyze_corpus_paths:
        consume_text(read_utf8_text(path), evidence_kind="corpus"); print(f"[INFO] unidic analyzed corpus={path}")

def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)

def build_policy(args: argparse.Namespace) -> PromotionPolicy:
    presets = {"conservative": dict(min_source_support=2, min_candidate_hits=2, min_corpus_hits=2, allow_function_word_auto=False, allow_single_char_auto=False, allow_ascii_auto=False, promote_corpus_only_entries=False), "balanced": dict(min_source_support=2, min_candidate_hits=1, min_corpus_hits=2, allow_function_word_auto=False, allow_single_char_auto=False, allow_ascii_auto=False, promote_corpus_only_entries=True), "aggressive": dict(min_source_support=1, min_candidate_hits=1, min_corpus_hits=1, allow_function_word_auto=True, allow_single_char_auto=True, allow_ascii_auto=True, promote_corpus_only_entries=True)}[str(args.promotion_policy)]
    return PromotionPolicy(name=str(args.promotion_policy), min_source_support=args.min_source_support if args.min_source_support is not None else presets["min_source_support"], min_candidate_hits=args.min_candidate_hits if args.min_candidate_hits is not None else presets["min_candidate_hits"], min_corpus_hits=args.min_corpus_hits if args.min_corpus_hits is not None else presets["min_corpus_hits"], allow_function_word_auto=args.allow_function_word_auto or presets["allow_function_word_auto"], allow_single_char_auto=args.allow_single_char_auto or presets["allow_single_char_auto"], allow_ascii_auto=args.allow_ascii_auto or presets["allow_ascii_auto"], promote_corpus_only_entries=args.promote_corpus_only_entries or presets["promote_corpus_only_entries"], max_surface_forms=max(1, int(args.max_surface_forms)), review_limit=max(10, int(args.review_limit)))

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="日本語 WordNet + Sudachi + UniDic を使って v4 辞書を大規模ブートストラップします")
    parser.add_argument("output", type=Path, help="出力先 (.json / .lsd / .lsdx)")
    parser.add_argument("--seed", type=Path, default=None, help="既存 seed 辞書。未指定時は examples/seed_core_japanese_dictionary.json を優先")
    parser.add_argument("--no-default-seed", action="store_true", help="組み込み seed 辞書の自動読込を無効化する")
    parser.add_argument("--extra-text", type=Path, action="append", default=[], help="追加コーパスのテキストファイル。複数指定可")
    parser.add_argument("--download-nltk-data", action="store_true", help="NLTK の wordnet / omw-1.4 を必要なら自動取得する")
    parser.add_argument("--wordnet-max-synsets", type=int, default=0, help="WordNet の読み込み上限。0 は全件")
    parser.add_argument("--min-surface-length", type=int, default=1, help="取り込む表層形の最小長")
    parser.add_argument("--allow-non-japanese", action="store_true", help="非日本語表記も取り込む")
    parser.add_argument("--skip-wordnet", action="store_true", help="WordNet 取り込みを無効化")
    parser.add_argument("--skip-sudachi", action="store_true", help="Sudachi 取り込みを無効化")
    parser.add_argument("--skip-unidic", action="store_true", help="UniDic 取り込みを無効化")
    parser.add_argument("--sudachi-dict", choices=["small", "core", "full"], default="core", help="Sudachi の辞書エディション")
    parser.add_argument("--sudachi-split-mode", choices=["A", "B", "C"], default="C", help="Sudachi の分割モード")
    parser.add_argument("--unidic-dictionary", choices=["lite", "full"], default="lite", help="UniDic の利用方針。full は別途 unidic download が必要")
    parser.add_argument("--promotion-policy", choices=["conservative", "balanced", "aggressive"], default="conservative", help="自動昇格ポリシー")
    parser.add_argument("--min-source-support", type=int, default=None, help="自動昇格に必要な最小 source support")
    parser.add_argument("--min-candidate-hits", type=int, default=None, help="候補解析だけで自動昇格するための最小ヒット数")
    parser.add_argument("--min-corpus-hits", type=int, default=None, help="コーパス由来で自動昇格するための最小ヒット数")
    parser.add_argument("--allow-function-word-auto", action="store_true", help="function word も自動昇格対象にする")
    parser.add_argument("--allow-single-char-auto", action="store_true", help="1文字語も自動昇格対象にする")
    parser.add_argument("--allow-ascii-auto", action="store_true", help="ASCII 主体語も自動昇格対象にする")
    parser.add_argument("--promote-corpus-only-entries", action="store_true", help="コーパス由来のみの新規語も自動昇格対象にする")
    parser.add_argument("--max-surface-forms", type=int, default=12, help="1 entry に保持する surface_forms の最大数")
    parser.add_argument("--review-limit", type=int, default=500, help="review/rejected report の最大件数")
    parser.add_argument("--review-json", type=Path, default=None, help="review / rejected 候補のレポート JSON")
    parser.add_argument("--stats-json", type=Path, default=None, help="統計情報を別 JSON に保存")
    parser.add_argument("--verify-roundtrip", action="store_true", help="保存後に再ロードして strict schema 検証する")
    parser.add_argument("--relation-target-policy", choices=["preserve", "prune", "error"], default="prune", help="concept relation の target が辞書内にない場合の扱い")
    parser.add_argument("--force", action="store_true", help="出力先が存在していても上書きする")
    return parser.parse_args(argv)

def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv); policy = build_policy(args); output_path: Path = args.output
    if output_path.exists() and not args.force:
        print(f"[ERROR] output already exists: {output_path} (overwrite する場合は --force)", file=sys.stderr); return 1
    seed_path = args.seed
    if seed_path is None and not args.no_default_seed:
        default_seed = default_seed_path()
        if default_seed.exists(): seed_path = default_seed
    if seed_path and not seed_path.exists():
        print(f"[ERROR] seed not found: {seed_path}", file=sys.stderr); return 1
    missing_extra = [path for path in args.extra_text if not path.exists()]
    if missing_extra:
        print(f"[ERROR] extra text not found: {missing_extra}", file=sys.stderr); return 1
    builder = LexiconBuilder(policy)
    if seed_path is not None:
        builder.stats.default_seed_loaded = seed_path == default_seed_path(); load_seed(builder, seed_path); print(f"[INFO] loaded seed={seed_path} entries={builder.stats.seed_entries}")
    candidate_surfaces: List[str] = []
    if not args.skip_wordnet:
        candidate_surfaces.extend(import_wordnet(builder, download_nltk_data=args.download_nltk_data, max_synsets=args.wordnet_max_synsets, min_surface_length=args.min_surface_length, allow_non_japanese=args.allow_non_japanese))
        print(f"[INFO] wordnet complete synsets={builder.stats.wordnet_synsets} lemma_forms={builder.stats.wordnet_lemma_forms} entries={len(builder.entries)}")
    candidate_surfaces.extend(entry.lemma for entry in builder.entries.values()); candidate_surfaces = list(dict.fromkeys(surface for surface in candidate_surfaces if surface))
    if not args.skip_sudachi:
        import_sudachi_candidates(builder, surfaces=candidate_surfaces, dict_type=args.sudachi_dict, split_mode=args.sudachi_split_mode, analyze_corpus_paths=args.extra_text, min_surface_length=args.min_surface_length, allow_non_japanese=args.allow_non_japanese)
        print(f"[INFO] sudachi complete tokens={builder.stats.sudachi_tokens} entries={len(builder.entries)}")
    if not args.skip_unidic:
        import_unidic_candidates(builder, surfaces=candidate_surfaces, dictionary_mode=args.unidic_dictionary, analyze_corpus_paths=args.extra_text, min_surface_length=args.min_surface_length, allow_non_japanese=args.allow_non_japanese)
        print(f"[INFO] unidic complete tokens={builder.stats.unidic_tokens} entries={len(builder.entries)}")
    container, review_payload = builder.build_output(relation_target_policy=args.relation_target_policy)
    validate_raw_lexicon_container(
        container,
        strict_schema=True,
        strict_relations=True,
        require_closed_relations=args.relation_target_policy != "preserve",
    )
    validate_lexicon_container(
        normalize_lexicon_container(container),
        strict_schema=True,
        strict_relations=True,
        require_closed_relations=args.relation_target_policy != "preserve",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json": write_json(output_path, container)
    else: save_lexicon_container(output_path, container)
    if args.verify_roundtrip:
        roundtrip = load_lexicon_container(output_path)
        validate_lexicon_container(
            roundtrip,
            strict_schema=True,
            strict_relations=True,
            require_closed_relations=args.relation_target_policy != "preserve",
        )
        print(f"[INFO] roundtrip verified: {output_path}")
    stats_payload = {"seed_entries": builder.stats.seed_entries, "wordnet_synsets": builder.stats.wordnet_synsets, "wordnet_lemma_forms": builder.stats.wordnet_lemma_forms, "sudachi_tokens": builder.stats.sudachi_tokens, "unidic_tokens": builder.stats.unidic_tokens, "accepted_entries": builder.stats.accepted_entries, "review_entries": builder.stats.review_entries, "rejected_entries": builder.stats.rejected_entries, "relation_count": builder.stats.relation_count, "dangling_relations": builder.stats.dangling_relations, "pruned_relations": builder.stats.pruned_relations, "entry_count": len(container.get("lexical_entries", {})), "concept_count": len(container.get("concepts", {})), "seed": str(seed_path) if seed_path else None, "output": str(output_path), "relation_target_policy": args.relation_target_policy}
    if args.stats_json: write_json(args.stats_json, stats_payload)
    if args.review_json: write_json(args.review_json, review_payload)
    print(f"[DONE] entries={len(container.get('lexical_entries', {}))} concepts={len(container.get('concepts', {}))} relations={builder.stats.relation_count} pruned_relations={builder.stats.pruned_relations} review={builder.stats.review_entries} rejected={builder.stats.rejected_entries} output={output_path}")
    print(stable_json_dumps(stats_payload)); return 0

if __name__ == "__main__":
    raise SystemExit(main())
