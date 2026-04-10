from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.io.lsd_lexicon import (  # noqa: E402
    load_lexicon_container,
    normalize_lexicon_container,
    save_lexicon_container,
    stable_json_dumps,
    validate_lexicon_container,
)
from src.core.relation.schema import canonicalize_relation  # noqa: E402

DEFAULT_SEED_PATH = PROJECT_ROOT / "examples" / "seed_conversation_ja_core.json"


def _load_container(path: Path) -> Dict[str, Any]:
    container = load_lexicon_container(path)
    normalized = normalize_lexicon_container(container)
    validate_lexicon_container(normalized, require_closed_relations=True)
    return normalized


def _lexical_entries(container: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    lexical = container.get("entries")
    if isinstance(lexical, Mapping):
        return {str(k): dict(v) for k, v in lexical.items() if isinstance(v, Mapping)}
    lexical = container.get("lexical_entries")
    if isinstance(lexical, Mapping):
        return {str(k): dict(v) for k, v in lexical.items() if isinstance(v, Mapping)}
    return {}


def _unique_preserve(values: Iterable[Any]) -> List[Any]:
    ordered: List[Any] = []
    seen: set[str] = set()
    for value in values:
        key = stable_json_dumps(value) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(value)
    return ordered


def _merge_meta(base: Mapping[str, Any], incoming: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if key not in merged:
            merged[key] = deepcopy(value)
            continue
        if isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_meta(merged[key], value)
        elif isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = _unique_preserve(list(merged[key]) + list(value))
        elif merged[key] in (None, "", [], {}, False):
            merged[key] = deepcopy(value)
    return merged


def _relation_signature(relation: Mapping[str, Any]) -> str:
    return stable_json_dumps(canonicalize_relation(relation))


def _merge_relations(existing: Sequence[Mapping[str, Any]], incoming: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for source in list(existing) + list(incoming):
        relation = canonicalize_relation(source)
        signature = _relation_signature(relation)
        current = merged.get(signature)
        if current is None:
            merged[signature] = relation
            continue
        current["weight"] = max(float(current.get("weight", 1.0)), float(relation.get("weight", 1.0)))
        current["confidence"] = max(float(current.get("confidence", 1.0)), float(relation.get("confidence", 1.0)))
        current["usage_stage"] = _unique_preserve(list(current.get("usage_stage", [])) + list(relation.get("usage_stage", [])))
        current_meta = current.get("meta", {}) if isinstance(current.get("meta"), Mapping) else {}
        relation_meta = relation.get("meta", {}) if isinstance(relation.get("meta"), Mapping) else {}
        current["meta"] = _merge_meta(current_meta, relation_meta)
    return list(merged.values())


def _merge_senses(existing: Sequence[Mapping[str, Any]], incoming: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    anonymous: List[Dict[str, Any]] = []
    for source in list(existing) + list(incoming):
        if not isinstance(source, Mapping):
            continue
        sense = dict(source)
        sense_id = str(sense.get("id", "")).strip()
        if not sense_id:
            anonymous.append(sense)
            continue
        current = merged.get(sense_id)
        if current is None:
            merged[sense_id] = sense
            continue
        current["concept_ids"] = _unique_preserve(list(current.get("concept_ids", [])) + list(sense.get("concept_ids", [])))
        current["priority"] = max(float(current.get("priority", 0.0)), float(sense.get("priority", 0.0)))
        if not current.get("gloss") and sense.get("gloss"):
            current["gloss"] = sense["gloss"]
        if not current.get("usage_notes") and sense.get("usage_notes"):
            current["usage_notes"] = sense["usage_notes"]
        current_meta = current.get("meta", {}) if isinstance(current.get("meta"), Mapping) else {}
        sense_meta = sense.get("meta", {}) if isinstance(sense.get("meta"), Mapping) else {}
        if current_meta or sense_meta:
            current["meta"] = _merge_meta(current_meta, sense_meta)
    ordered = list(merged.values())
    ordered.extend(anonymous)
    return ordered


def _merge_surface_forms(existing: Sequence[Mapping[str, Any]], incoming: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[tuple[str, str], Dict[str, Any]] = {}
    for source in list(existing) + list(incoming):
        if not isinstance(source, Mapping):
            continue
        text = str(source.get("text", "")).strip()
        kind = str(source.get("kind", "lemma")).strip() or "lemma"
        if not text:
            continue
        key = (text, kind)
        current = merged.get(key)
        if current is None:
            merged[key] = dict(source)
            continue
        if not current.get("reading") and source.get("reading"):
            current["reading"] = source["reading"]
    return list(merged.values())


def _merge_entry(base_entry: Mapping[str, Any] | None, seed_entry: Mapping[str, Any]) -> Dict[str, Any]:
    current = dict(base_entry) if isinstance(base_entry, Mapping) else {}
    incoming = dict(seed_entry)
    for key in ("word", "lemma", "reading", "slot_frame_id", "category"):
        if current.get(key) in (None, "") and incoming.get(key) not in (None, ""):
            current[key] = deepcopy(incoming[key])
    if "grammar" in incoming or "grammar" in current:
        grammar = dict(current.get("grammar", {}))
        for gk, gv in dict(incoming.get("grammar", {})).items():
            if gk not in grammar or grammar.get(gk) in (None, "", [], {}, False):
                grammar[gk] = deepcopy(gv)
            elif isinstance(grammar.get(gk), list) and isinstance(gv, list):
                grammar[gk] = _unique_preserve(list(grammar[gk]) + list(gv))
        current["grammar"] = grammar
    current["surface_forms"] = _merge_surface_forms(current.get("surface_forms", []), incoming.get("surface_forms", []))
    current["senses"] = _merge_senses(current.get("senses", []), incoming.get("senses", []))
    current["style_tags"] = _unique_preserve(list(current.get("style_tags", [])) + list(incoming.get("style_tags", [])))
    current["concept_ids"] = _unique_preserve(list(current.get("concept_ids", [])) + list(incoming.get("concept_ids", [])))
    current["frequency"] = max(float(current.get("frequency", 0.0)), float(incoming.get("frequency", 0.0)))
    current_meta = current.get("meta", {}) if isinstance(current.get("meta"), Mapping) else {}
    incoming_meta = incoming.get("meta", {}) if isinstance(incoming.get("meta"), Mapping) else {}
    current["meta"] = _merge_meta(current_meta, incoming_meta)
    return current


def _merge_concept(base_concept: Mapping[str, Any] | None, seed_concept: Mapping[str, Any]) -> Dict[str, Any]:
    current = dict(base_concept) if isinstance(base_concept, Mapping) else {}
    incoming = dict(seed_concept)
    for key in ("id", "label", "category", "description", "default_slot_frame_id"):
        if current.get(key) in (None, "") and incoming.get(key) not in (None, ""):
            current[key] = deepcopy(incoming[key])
    axes = dict(current.get("axes", {}))
    axes.update(dict(incoming.get("axes", {})))
    if axes:
        current["axes"] = axes
    current["relations"] = _merge_relations(current.get("relations", []), incoming.get("relations", []))
    current_meta = current.get("meta", {}) if isinstance(current.get("meta"), Mapping) else {}
    incoming_meta = incoming.get("meta", {}) if isinstance(incoming.get("meta"), Mapping) else {}
    current["meta"] = _merge_meta(current_meta, incoming_meta)
    return current


def _rebuild_indexes(container: MutableMapping[str, Any]) -> None:
    lexical_entries = _lexical_entries(container)
    concepts = container.get("concepts", {}) if isinstance(container.get("concepts"), Mapping) else {}
    indexes = container.get("indexes", {}) if isinstance(container.get("indexes"), Mapping) else {}

    surface_to_entry: Dict[str, List[str]] = {}
    concept_to_entries: Dict[str, List[str]] = {}
    relation_by_type: Dict[str, List[str]] = {}
    relation_target_to_sources: Dict[str, List[str]] = {}

    def append_unique(mapping: Dict[str, List[str]], key: str, value: str) -> None:
        bucket = mapping.setdefault(key, [])
        if value not in bucket:
            bucket.append(value)

    for entry_id, entry in lexical_entries.items():
        lemma = str(entry.get("word") or entry.get("lemma") or "").strip()
        if lemma:
            append_unique(surface_to_entry, lemma, str(entry_id))
        reading = str(entry.get("reading", "")).strip()
        if reading:
            append_unique(surface_to_entry, reading, str(entry_id))
        for form in entry.get("surface_forms", []):
            if isinstance(form, Mapping):
                text = str(form.get("text", "")).strip()
                if text:
                    append_unique(surface_to_entry, text, str(entry_id))
        for concept_id in entry.get("concept_ids", []):
            append_unique(concept_to_entries, str(concept_id), str(entry_id))
        for sense in entry.get("senses", []):
            if isinstance(sense, Mapping):
                for concept_id in sense.get("concept_ids", []):
                    append_unique(concept_to_entries, str(concept_id), str(entry_id))

    for concept_id, concept in concepts.items():
        for relation in concept.get("relations", []):
            if not isinstance(relation, Mapping):
                continue
            canonical = canonicalize_relation(relation)
            append_unique(relation_by_type, str(canonical.get("type")), str(concept_id))
            append_unique(relation_target_to_sources, str(canonical.get("target")), str(concept_id))

    indexes.update(
        {
            "surface_to_entry": surface_to_entry,
            "concept_to_entries": concept_to_entries,
            "relation_by_type": relation_by_type,
            "relation_target_to_sources": relation_target_to_sources,
        }
    )
    container["indexes"] = indexes


def merge_containers(base: Mapping[str, Any], seed_containers: Sequence[Mapping[str, Any]], seed_paths: Sequence[Path] | None = None) -> Dict[str, Any]:
    merged = deepcopy(normalize_lexicon_container(base))
    merged.setdefault("concepts", {})
    merged.setdefault("entries", {})
    merged.setdefault("slot_frames", {})
    merged.setdefault("indexes", {})

    for seed in seed_containers:
        normalized_seed = deepcopy(normalize_lexicon_container(seed))
        merged_meta = dict(merged.get("meta", {}))
        seed_meta = dict(normalized_seed.get("meta", {}))
        merged["meta"] = _merge_meta(merged_meta, seed_meta)

        for slot_id, slot_frame in normalized_seed.get("slot_frames", {}).items():
            if slot_id not in merged["slot_frames"]:
                merged["slot_frames"][slot_id] = deepcopy(slot_frame)

        for concept_id, concept in normalized_seed.get("concepts", {}).items():
            merged["concepts"][concept_id] = _merge_concept(merged["concepts"].get(concept_id), concept)

        for entry_id, entry in _lexical_entries(normalized_seed).items():
            merged["entries"][entry_id] = _merge_entry(merged["entries"].get(entry_id), entry)

    merged_meta = dict(merged.get("meta", {}))
    augmentation_meta = merged_meta.get("augmentation", {}) if isinstance(merged_meta.get("augmentation"), Mapping) else {}
    augmentation_meta["conversation_seed_applied"] = True
    augmentation_meta["seed_files"] = [str(path) for path in (seed_paths or [DEFAULT_SEED_PATH])]
    merged_meta["augmentation"] = augmentation_meta
    merged["meta"] = merged_meta

    _rebuild_indexes(merged)
    merged = normalize_lexicon_container(merged)
    validate_lexicon_container(merged, require_closed_relations=True)
    return merged


def _count_relations(concepts: Mapping[str, Any]) -> int:
    total = 0
    for concept in concepts.values():
        if isinstance(concept, Mapping) and isinstance(concept.get("relations"), list):
            total += len(concept.get("relations", []))
    return total


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a conversation-focused seed lexicon into an existing LSLM v4 lexicon.")
    parser.add_argument("--base", required=True, type=Path, help="Base lexicon (.json/.lsd/.lsdx)")
    parser.add_argument("--output", required=True, type=Path, help="Merged lexicon output path")
    parser.add_argument(
        "--seed",
        action="append",
        type=Path,
        help="Additional seed lexicon(s). If omitted, uses examples/seed_conversation_ja_core.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print summary without writing output")
    parser.add_argument("--print-summary", action="store_true", help="Print summary JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    base_path = args.base.resolve()
    output_path = args.output.resolve()
    seed_paths = [path.resolve() for path in (args.seed or [DEFAULT_SEED_PATH])]

    if not base_path.exists():
        parser.error(f"base lexicon not found: {base_path}")
    for seed_path in seed_paths:
        if not seed_path.exists():
            parser.error(f"seed lexicon not found: {seed_path}")

    base = _load_container(base_path)
    seeds = [_load_container(path) for path in seed_paths]
    before_concepts = len(base.get("concepts", {}))
    before_entries = len(_lexical_entries(base))
    before_relations = _count_relations(base.get("concepts", {}))

    merged = merge_containers(base, seeds, seed_paths=seed_paths)

    summary = {
        "base": str(base_path),
        "output": str(output_path),
        "seed_files": [str(path) for path in seed_paths],
        "concepts_before": before_concepts,
        "concepts_after": len(merged.get("concepts", {})),
        "lexical_entries_before": before_entries,
        "lexical_entries_after": len(_lexical_entries(merged)),
        "relations_before": before_relations,
        "relations_after": _count_relations(merged.get("concepts", {})),
        "surface_index_after": len((merged.get("indexes", {}) or {}).get("surface_to_entry", {})),
    }

    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_lexicon_container(output_path, merged)

    if args.print_summary or args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            f"[OK] merged conversation seed into lexicon: concepts {summary['concepts_before']} -> {summary['concepts_after']}, "
            f"lexical_entries {summary['lexical_entries_before']} -> {summary['lexical_entries_after']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
