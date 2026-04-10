from __future__ import annotations

import io
import json
import mmap
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping

MAGIC = b"LSLMDICT"
VERSION = 1

INDEXED_MAGIC = b"LSLMDX2"
INDEXED_VERSION = 2

CORE_BOOL_FIELDS = ["independent", "can_start", "can_end", "content_word", "function_word"]
CORE_LIST_FIELDS = ["roles", "requires_prev", "requires_next", "forbid_prev", "forbid_next"]
CORE_SCALAR_FIELDS = ["pos", "sub_pos", "conjugation_type", "conjugation_slot", "connectability"]
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

RELATION_DIRECTIONS = {"outbound", "inbound", "bidirectional"}
RELATION_USAGE_STAGES = {"planning", "divergence", "convergence", "slot", "surface"}
RELATION_LAYERS = {"semantic", "syntactic", "expressive"}
RELATION_TYPE_RULES: Dict[str, Dict[str, Any]] = {
    "synonym": {"layer": "semantic", "direction": "bidirectional", "inverse_type": "synonym", "usage_stage": ["divergence", "convergence", "surface"]},
    "antonym": {"layer": "semantic", "direction": "bidirectional", "inverse_type": "antonym", "usage_stage": ["divergence", "convergence"]},
    "hypernym": {"layer": "semantic", "direction": "outbound", "inverse_type": "hyponym", "usage_stage": ["divergence", "convergence"]},
    "hyponym": {"layer": "semantic", "direction": "outbound", "inverse_type": "hypernym", "usage_stage": ["divergence", "convergence"]},
    "cause_of": {"layer": "semantic", "direction": "outbound", "inverse_type": "caused_by", "usage_stage": ["divergence", "convergence"]},
    "caused_by": {"layer": "semantic", "direction": "outbound", "inverse_type": "cause_of", "usage_stage": ["divergence", "convergence"]},
    "part_of": {"layer": "semantic", "direction": "outbound", "inverse_type": "has_part", "usage_stage": ["divergence", "convergence"]},
    "has_part": {"layer": "semantic", "direction": "outbound", "inverse_type": "part_of", "usage_stage": ["divergence", "convergence"]},
    "often_with": {"layer": "semantic", "direction": "bidirectional", "inverse_type": "often_with", "usage_stage": ["divergence", "surface"]},
    "related_to": {"layer": "semantic", "direction": "bidirectional", "inverse_type": "related_to", "usage_stage": ["divergence", "convergence", "surface"]},
    "target_domain": {"layer": "semantic", "direction": "outbound", "usage_stage": ["divergence", "convergence"]},
    "predicate_slot": {"layer": "syntactic", "direction": "outbound", "usage_stage": ["slot", "convergence"]},
    "modifier_head": {"layer": "syntactic", "direction": "outbound", "usage_stage": ["convergence", "surface"]},
    "connective_sequence": {"layer": "syntactic", "direction": "outbound", "usage_stage": ["surface"]},
    "subject_predicate": {"layer": "syntactic", "direction": "outbound", "usage_stage": ["slot", "convergence"]},
    "argument_role": {"layer": "syntactic", "direction": "outbound", "usage_stage": ["slot", "convergence"]},
    "style_variant": {"layer": "expressive", "direction": "bidirectional", "inverse_type": "style_variant", "usage_stage": ["surface", "convergence"]},
    "politeness_variant": {"layer": "expressive", "direction": "bidirectional", "inverse_type": "politeness_variant", "usage_stage": ["surface", "convergence"]},
    "paraphrase": {"layer": "expressive", "direction": "bidirectional", "inverse_type": "paraphrase", "usage_stage": ["surface", "convergence"]},
    "collocation": {"layer": "expressive", "direction": "bidirectional", "inverse_type": "collocation", "usage_stage": ["divergence", "surface"]},
    "register_variant": {"layer": "expressive", "direction": "bidirectional", "inverse_type": "register_variant", "usage_stage": ["surface", "convergence"]},
}

TOP_LEVEL_BINARY_META_KEY = "__lslm_top_level__"


class ConsoleProgressBar:
    def __init__(self, total: int, title: str = "Loading", width: int = 28, enabled: bool | None = None) -> None:
        self.total = max(int(total), 1)
        self.title = title
        self.width = max(int(width), 10)
        self.enabled = sys.stderr.isatty() if enabled is None else bool(enabled)
        self.current = 0
        self._last_render = 0.0
        self._started = time.perf_counter()
        if self.enabled:
            self._render(force=True)

    def update(self, step: int = 1) -> None:
        self.current = min(self.total, self.current + max(int(step), 0))
        if self.enabled:
            self._render()

    def set(self, value: int) -> None:
        self.current = max(0, min(self.total, int(value)))
        if self.enabled:
            self._render()

    def _render(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and (now - self._last_render) < 0.03 and self.current < self.total:
            return
        self._last_render = now
        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self._started
        msg = f"\r{self.title} [{bar}] {self.current}/{self.total} {ratio * 100:5.1f}% {elapsed:5.1f}s"
        print(msg, end="", file=sys.stderr, flush=True)

    def close(self) -> None:
        if self.enabled:
            self.current = self.total
            self._render(force=True)
            print(file=sys.stderr, flush=True)


def should_show_progress(path: str | Path | None = None, *, min_bytes: int = 64 * 1024) -> bool:
    if not sys.stderr.isatty():
        return False
    if path is None:
        return True
    try:
        return Path(path).stat().st_size >= min_bytes
    except OSError:
        return True


def write_uvarint(n: int) -> bytes:
    if n < 0:
        raise ValueError("uvarint cannot encode negative values")
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def read_uvarint(buf: io.BytesIO) -> int:
    shift = 0
    result = 0
    while True:
        b_raw = buf.read(1)
        if not b_raw:
            raise EOFError("Unexpected EOF while reading uvarint")
        b = b_raw[0]
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result
        shift += 7
        if shift > 63:
            raise ValueError("uvarint too large")


def write_bytes_with_len(data: bytes) -> bytes:
    return write_uvarint(len(data)) + data


def read_bytes_with_len(buf: io.BytesIO) -> bytes:
    n = read_uvarint(buf)
    data = buf.read(n)
    if len(data) != n:
        raise EOFError("Unexpected EOF while reading length-prefixed bytes")
    return data


def skip_bytes_with_len(buf: io.BytesIO) -> None:
    n = read_uvarint(buf)
    data = buf.read(n)
    if len(data) != n:
        raise EOFError("Unexpected EOF while skipping length-prefixed bytes")


def write_str(s: str) -> bytes:
    return write_bytes_with_len(s.encode("utf-8"))


def read_str(buf: io.BytesIO) -> str:
    return read_bytes_with_len(buf).decode("utf-8")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def quantize_unit_float_to_i16(v: float) -> int:
    v = clamp(float(v), -1.0, 1.0)
    return int(round(v * 32767.0))


def dequantize_i16_to_unit_float(v: int) -> float:
    return max(-1.0, min(1.0, float(v) / 32767.0))


def pack_i16_list(values: List[int]) -> bytes:
    if not values:
        return b""
    return struct.pack("<" + "h" * len(values), *values)


def unpack_i16_list(data: bytes, count: int) -> List[int]:
    if count == 0:
        return []
    need = count * 2
    if len(data) != need:
        raise ValueError(f"Invalid int16 payload size: expected {need}, got {len(data)}")
    return list(struct.unpack("<" + "h" * count, data))


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _unique_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _to_str_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return []
    return [str(value)]


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _canonicalize_surface_forms(value: Any, fallback_word: str | None = None) -> List[Dict[str, Any]]:
    items: List[Any]
    if isinstance(value, list):
        items = list(value)
    elif value is None:
        items = []
    else:
        items = [value]

    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def append_form(text: str | None, kind: str = "variant", extras: Mapping[str, Any] | None = None) -> None:
        normalized_text = _to_optional_str(text)
        if not normalized_text:
            return
        normalized_kind = _to_optional_str(kind) or "variant"
        key = (normalized_text, normalized_kind)
        if key in seen:
            return
        seen.add(key)
        form: Dict[str, Any] = {"text": normalized_text, "kind": normalized_kind}
        if isinstance(extras, Mapping):
            for extra_key, extra_value in extras.items():
                if extra_key not in {"text", "surface", "value", "kind", "type"}:
                    form[str(extra_key)] = extra_value
        out.append(form)

    append_form(fallback_word, kind="lemma")

    for item in items:
        if isinstance(item, Mapping):
            extras = dict(item)
            text = extras.get("text", extras.get("surface", extras.get("value")))
            kind = extras.get("kind", extras.get("type", "variant"))
            append_form(text, kind=str(kind), extras=extras)
        else:
            append_form(str(item), kind="variant")

    return out


def _canonicalize_senses(value: Any) -> List[Dict[str, Any]]:
    items: List[Any]
    if isinstance(value, list):
        items = list(value)
    elif value is None:
        items = []
    else:
        items = [value]

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        raw = dict(item) if isinstance(item, Mapping) else {"gloss": str(item)}
        sense_id = _to_optional_str(raw.get("id")) or f"sense:{i}"
        gloss = _to_optional_str(raw.get("gloss", raw.get("label", raw.get("description"))))
        concept_ids = _unique_keep_order(
            _to_str_list(raw.get("concept_ids", raw.get("concept_id", [])))
        )
        slot_frame_override = _to_optional_str(
            raw.get("slot_frame_override", raw.get("slot_frame_id"))
        )

        normalized: Dict[str, Any] = {"id": sense_id}
        if gloss is not None:
            normalized["gloss"] = gloss
        if concept_ids:
            normalized["concept_ids"] = concept_ids
        if slot_frame_override is not None:
            normalized["slot_frame_override"] = slot_frame_override
        if raw.get("priority") is not None:
            normalized["priority"] = round(_to_float(raw.get("priority"), 1.0), 6)

        for key, field in raw.items():
            if key in {
                "id",
                "gloss",
                "label",
                "description",
                "concept_ids",
                "concept_id",
                "slot_frame_override",
                "slot_frame_id",
                "priority",
            }:
                continue
            normalized[str(key)] = field

        out.append(normalized)

    return out


def _derive_entry_concept_ids(raw: Mapping[str, Any], senses: List[Dict[str, Any]]) -> List[str]:
    concept_ids = _unique_keep_order(
        _to_str_list(raw.get("concept_ids", raw.get("concept_id", [])))
    )
    for sense in senses:
        concept_ids.extend(_to_str_list(sense.get("concept_ids", [])))
    return _unique_keep_order([str(v) for v in concept_ids])


def _is_entry_mapping(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    grammar = value.get("grammar")
    if not isinstance(grammar, Mapping):
        return False
    return (
        isinstance(value.get("vector"), Mapping)
        or "word" in value
        or "lemma" in value
        or "category" in value
        or "slots" in value
        or "relations" in value
        or "surface_forms" in value
        or "senses" in value
        or "concept_ids" in value
        or "slot_frame_id" in value
    )


def _canonicalize_grammar(grammar: Any) -> Dict[str, Any]:
    raw = dict(grammar) if isinstance(grammar, Mapping) else {}

    sub_pos = raw.get("sub_pos", raw.get("subpos", ""))
    pos = str(raw.get("pos", "unknown"))
    conjugation_type = str(raw.get("conjugation_type", raw.get("conj_type", "none")))
    conjugation_slot = str(raw.get("conjugation_slot", raw.get("conj_slot", "none")))
    connectability = _to_float(raw.get("connectability", 0.0), 0.0)

    out: Dict[str, Any] = {
        "pos": pos,
        "sub_pos": str(sub_pos),
        "conjugation_type": conjugation_type,
        "conjugation_slot": conjugation_slot,
        "connectability": connectability,
        "independent": _to_bool(raw.get("independent", True), True),
        "can_start": _to_bool(raw.get("can_start", False), False),
        "can_end": _to_bool(raw.get("can_end", False), False),
        "content_word": _to_bool(raw.get("content_word", True), True),
        "function_word": _to_bool(raw.get("function_word", False), False),
        "roles": _to_str_list(raw.get("roles", [])),
        "requires_prev": _to_str_list(raw.get("requires_prev", [])),
        "requires_next": _to_str_list(raw.get("requires_next", [])),
        "forbid_prev": _to_str_list(raw.get("forbid_prev", [])),
        "forbid_next": _to_str_list(raw.get("forbid_next", [])),
    }

    for key, value in raw.items():
        if key in {"subpos"}:
            continue
        if key not in out:
            out[key] = value

    return out


def _canonicalize_vector(vector: Any) -> Dict[str, float]:
    if not isinstance(vector, Mapping):
        return {}
    out: Dict[str, float] = {}
    for key, value in vector.items():
        out[str(key)] = round(_to_float(value, 0.0), 6)
    return out


def _canonicalize_entry(key: str, entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise TypeError(f"Lexicon entry for {key!r} must be a mapping")

    raw = dict(entry)
    grammar = _canonicalize_grammar(raw.get("grammar", {}))
    canonical_word = str(raw.get("word", raw.get("lemma", key)))
    senses = _canonicalize_senses(raw.get("senses", []))
    concept_ids = _derive_entry_concept_ids(raw, senses)
    surface_forms = _canonicalize_surface_forms(raw.get("surface_forms", raw.get("surfaces", [])), canonical_word)
    slot_frame_id = _to_optional_str(raw.get("slot_frame_id", raw.get("slot_frame")))

    normalized: Dict[str, Any] = {
        "word": canonical_word,
        "category": str(raw.get("category", grammar.get("pos", "unknown"))),
        "hierarchy": _to_str_list(raw.get("hierarchy", [])),
        "vector": _canonicalize_vector(raw.get("vector", {})),
        "grammar": grammar,
        "slots": list(raw.get("slots", [])) if isinstance(raw.get("slots", []), list) else [],
        "relations": list(raw.get("relations", [])) if isinstance(raw.get("relations", []), list) else [],
        "frequency": _to_float(raw.get("frequency", 0.0), 0.0),
        "style_tags": _to_str_list(raw.get("style_tags", [])),
        "meta": dict(raw.get("meta", {})) if isinstance(raw.get("meta", {}), Mapping) else {},
        "surface_forms": surface_forms,
        "senses": senses,
        "concept_ids": concept_ids,
    }

    reading = _to_optional_str(raw.get("reading"))
    if reading is not None:
        normalized["reading"] = reading
    if slot_frame_id is not None:
        normalized["slot_frame_id"] = slot_frame_id

    for field in (
        "word",
        "category",
        "hierarchy",
        "vector",
        "grammar",
        "slots",
        "relations",
        "frequency",
        "style_tags",
        "meta",
        "surface_forms",
        "surfaces",
        "senses",
        "concept_ids",
        "concept_id",
        "slot_frame_id",
        "slot_frame",
        "reading",
    ):
        raw.pop(field, None)

    normalized.update(raw)
    return normalized


def _flatten_hierarchy_node(node: Any, entries: Dict[str, Dict[str, Any]]) -> None:
    if isinstance(node, Mapping):
        if _is_entry_mapping(node):
            word = str(node.get("word", "")).strip()
            if word:
                entries[word] = _canonicalize_entry(word, node)
            return
        for value in node.values():
            _flatten_hierarchy_node(value, entries)
    elif isinstance(node, list):
        for value in node:
            _flatten_hierarchy_node(value, entries)


def flatten_hierarchical_lexicon(data: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}

    if "entries" in data and isinstance(data["entries"], Mapping):
        for key, value in data["entries"].items():
            if _is_entry_mapping(value):
                entries[str(key)] = _canonicalize_entry(str(key), value)

    if "lexical_entries" in data and isinstance(data["lexical_entries"], Mapping):
        for key, value in data["lexical_entries"].items():
            if isinstance(value, Mapping):
                canonical_key = str(value.get("word", value.get("lemma", key)))
                entries.setdefault(canonical_key, _canonicalize_entry(canonical_key, value))

    if entries:
        return entries

    if "lexicon" in data and isinstance(data["lexicon"], Mapping):
        _flatten_hierarchy_node(data["lexicon"], entries)
        if entries:
            return entries

    _flatten_hierarchy_node(data, entries)
    return entries


def _ensure_indexes(data: MutableMapping[str, Any], entries: Mapping[str, Dict[str, Any]]) -> None:
    indexes = data.setdefault("indexes", {})
    if not isinstance(indexes, dict):
        indexes = {}
        data["indexes"] = indexes

    by_pos: Dict[str, List[str]] = {}
    can_start: List[str] = []
    can_end: List[str] = []
    content_words: List[str] = []
    function_words: List[str] = []
    entry_path: Dict[str, List[str]] = {}
    surface_to_entry: Dict[str, List[str]] = {}
    concept_to_entries: Dict[str, List[str]] = {}
    entry_to_concepts: Dict[str, List[str]] = {}
    slot_frame_to_entries: Dict[str, List[str]] = {}
    sense_to_entry: Dict[str, List[str]] = {}

    existing_entry_path = indexes.get("entry_path", {})
    if not isinstance(existing_entry_path, Mapping):
        existing_entry_path = {}

    def append_unique(target: Dict[str, List[str]], key: str | None, value: str) -> None:
        if not key:
            return
        bucket = target.setdefault(str(key), [])
        if value not in bucket:
            bucket.append(value)

    for word, entry in entries.items():
        grammar = _canonicalize_grammar(entry.get("grammar", {}))
        pos = str(grammar.get("pos", "unknown"))
        by_pos.setdefault(pos, []).append(word)

        if grammar.get("can_start", False):
            can_start.append(word)
        if grammar.get("can_end", False):
            can_end.append(word)
        if grammar.get("content_word", False):
            content_words.append(word)
        if grammar.get("function_word", False):
            function_words.append(word)

        hierarchy = entry.get("hierarchy", [])
        if isinstance(hierarchy, list) and hierarchy:
            entry_path[word] = [str(v) for v in hierarchy] + [word]
        else:
            current_path = existing_entry_path.get(word)
            if isinstance(current_path, list):
                entry_path[word] = [str(v) for v in current_path]
            elif isinstance(current_path, str):
                entry_path[word] = [segment for segment in current_path.split("/") if segment]
            else:
                entry_path[word] = [word]

        append_unique(surface_to_entry, word, word)
        reading = _to_optional_str(entry.get("reading"))
        append_unique(surface_to_entry, reading, word)
        for form in _canonicalize_surface_forms(entry.get("surface_forms", []), word):
            append_unique(surface_to_entry, _to_optional_str(form.get("text")), word)

        senses = _canonicalize_senses(entry.get("senses", []))
        concept_ids = _derive_entry_concept_ids(entry, senses)
        if concept_ids:
            entry_to_concepts[word] = list(concept_ids)
        for concept_id in concept_ids:
            append_unique(concept_to_entries, concept_id, word)

        slot_frame_id = _to_optional_str(entry.get("slot_frame_id"))
        append_unique(slot_frame_to_entries, slot_frame_id, word)

        for sense in senses:
            append_unique(sense_to_entry, _to_optional_str(sense.get("id")), word)
            append_unique(slot_frame_to_entries, _to_optional_str(sense.get("slot_frame_override")), word)

    indexes["by_pos"] = by_pos
    indexes["can_start"] = can_start
    indexes["can_end"] = can_end
    indexes["content_words"] = content_words
    indexes["function_words"] = function_words
    indexes["entry_path"] = entry_path
    indexes["surface_to_entry"] = surface_to_entry
    indexes["concept_to_entries"] = concept_to_entries
    indexes["entry_to_concepts"] = entry_to_concepts
    indexes["slot_frame_to_entries"] = slot_frame_to_entries
    indexes["sense_to_entry"] = sense_to_entry

    # backward-compat aliases
    indexes["content_word"] = list(content_words)
    indexes["function_word"] = list(function_words)


def _ensure_meta(data: MutableMapping[str, Any], entries: Mapping[str, Dict[str, Any]]) -> None:
    meta = data.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta

    semantic_axes = meta.get("semantic_axes")
    if not isinstance(semantic_axes, list) or not semantic_axes:
        axes: List[str] = []
        for entry in entries.values():
            vector = entry.get("vector", {})
            if isinstance(vector, Mapping):
                axes.extend(str(k) for k in vector.keys())
        semantic_axes = _unique_keep_order(axes) or list(DEFAULT_SEMANTIC_AXES)

    meta["semantic_axes"] = [str(v) for v in semantic_axes]
    if "version" not in meta:
        meta["version"] = "v4"
    meta["entry_count"] = len(entries)


def _route_hierarchy_from_pos(pos: str) -> List[str]:
    if pos == "pronoun":
        return ["lexicon", "content_words", "pronouns", "other"]
    if pos in {"particle_case"}:
        return ["lexicon", "function_words", "particles", "case"]
    if pos in {"particle_binding"}:
        return ["lexicon", "function_words", "particles", "binding"]
    if pos in {"particle_conjunctive"}:
        return ["lexicon", "function_words", "particles", "conjunctive"]
    if pos in {"particle_sentence_final"}:
        return ["lexicon", "function_words", "particles", "sentence_final"]
    if pos == "auxiliary":
        return ["lexicon", "function_words", "auxiliaries"]
    if pos == "copula":
        return ["lexicon", "function_words", "copulas"]
    if pos == "iteration_mark":
        return ["lexicon", "function_words", "special_marks"]
    if pos in {"verb", "verb_stem"}:
        return ["lexicon", "content_words", "verbs", "stems", "oov"]
    if pos == "verb_suffix":
        return ["lexicon", "content_words", "verbs", "suffixes"]
    if pos == "adjective_i":
        return ["lexicon", "content_words", "adjectives", "i", "oov"]
    if pos in {"adjective_stem", "adjective_na_helper"}:
        if pos == "adjective_na_helper":
            return ["lexicon", "content_words", "adjectives", "na", "helper"]
        return ["lexicon", "content_words", "adjectives", "na", "stems", "oov"]
    if pos == "adjective_i_ending":
        return ["lexicon", "content_words", "adjectives", "i", "endings"]
    if pos == "adverb":
        return ["lexicon", "content_words", "adverbs", "oov"]
    if pos == "conjunction":
        return ["lexicon", "content_words", "conjunctions"]
    if pos == "interjection":
        return ["lexicon", "content_words", "interjections"]
    if pos == "adnominal":
        return ["lexicon", "content_words", "adnominals"]
    if pos == "prefix":
        return ["lexicon", "content_words", "prefixes"]
    if pos == "suffix":
        return ["lexicon", "content_words", "suffixes"]
    return ["lexicon", "content_words", "nouns", "generated", "oov"]


def build_hierarchical_container_from_entries(
    entries: Mapping[str, Dict[str, Any]],
    meta: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    canonical_entries = {str(k): _canonicalize_entry(str(k), v) for k, v in entries.items()}

    container: Dict[str, Any] = {
        "meta": dict(meta or {}),
        "lexicon": {
            "function_words": {
                "particles": {"case": {}, "binding": {}, "conjunctive": {}, "sentence_final": {}},
                "auxiliaries": {},
                "copulas": {},
                "special_marks": {},
            },
            "content_words": {
                "pronouns": {"personal": {}, "demonstrative": {}, "interrogative": {}, "other": {}},
                "nouns": {
                    "core": {"time": {}, "space": {}, "body": {}, "nature": {}, "society": {}, "object": {}, "abstract": {}, "other": {}},
                    "generated": {"semantic_mix": {}, "kanji_bigram": {}, "kanji_trigram": {}, "oov": {}},
                },
                "verbs": {
                    "stems": {"core_actions": {}, "mental_actions": {}, "state_change": {}, "communication": {}, "generated": {}, "oov": {}},
                    "suffixes": {},
                },
                "adjectives": {
                    "i": {"core": {}, "generated": {}, "endings": {}, "oov": {}},
                    "na": {"stems": {"core": {}, "generated": {}, "oov": {}}, "helper": {}},
                },
                "adverbs": {"core": {}, "generated": {}, "oov": {}},
                "conjunctions": {},
                "interjections": {},
                "adnominals": {},
                "prefixes": {},
                "suffixes": {},
            },
        },
        "indexes": {},
    }

    entry_path: Dict[str, List[str]] = {}

    for word, entry in canonical_entries.items():
        hierarchy = entry.get("hierarchy", [])
        if isinstance(hierarchy, list) and hierarchy:
            path = ["lexicon"] + [str(v) for v in hierarchy]
        else:
            pos = str(entry.get("grammar", {}).get("pos", "unknown"))
            path = _route_hierarchy_from_pos(pos)

        node: Dict[str, Any] = container
        for segment in path:
            next_node = node.setdefault(segment, {})
            if not isinstance(next_node, dict):
                raise TypeError(f"Hierarchy node is not a dict while inserting {word!r}: {segment!r}")
            node = next_node

        entry_copy = dict(entry)
        entry_copy["hierarchy"] = path[1:]
        node[word] = entry_copy
        entry_path[word] = path[1:] + [word]

    container["indexes"]["entry_path"] = entry_path
    _ensure_indexes(container, canonical_entries)
    _ensure_meta(container, canonical_entries)
    container["entries"] = canonical_entries
    return container


def normalize_lexicon_container(data: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypeError("Unsupported lexicon container format")

    entries = flatten_hierarchical_lexicon(data)
    meta = dict(data.get("meta", {})) if isinstance(data.get("meta", {}), Mapping) else {}
    container = build_hierarchical_container_from_entries(entries, meta=meta)

    indexes = data.get("indexes", {})
    if isinstance(indexes, Mapping):
        merged_indexes = dict(indexes)
        merged_indexes.update(container.get("indexes", {}))
        container["indexes"] = merged_indexes
        _ensure_indexes(container, container["entries"])

    top_level_extras = {
        str(key): value
        for key, value in data.items()
        if key not in {"meta", "indexes", "entries", "lexicon", "lexical_entries"}
    }
    container.update(top_level_extras)
    return container


def _new_validation_report() -> Dict[str, Any]:
    return {"errors": [], "warnings": []}


def _add_validation_issue(report: MutableMapping[str, Any], level: str, message: str) -> None:
    if level not in {"errors", "warnings"}:
        raise ValueError(f"Unsupported validation level: {level}")
    bucket = report.setdefault(level, [])
    if isinstance(bucket, list):
        bucket.append(str(message))


def _finalize_validation_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    errors = [str(v) for v in report.get("errors", [])] if isinstance(report.get("errors", []), list) else []
    warnings = [str(v) for v in report.get("warnings", [])] if isinstance(report.get("warnings", []), list) else []
    return {
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
    }


def _raise_for_validation_errors(report: Mapping[str, Any], prefix: str = "Lexicon validation failed") -> None:
    errors = [str(v) for v in report.get("errors", [])] if isinstance(report.get("errors", []), list) else []
    if not errors:
        return
    preview = "\n".join(f"- {message}" for message in errors[:20])
    if len(errors) > 20:
        preview += f"\n- ... and {len(errors) - 20} more"
    raise ValueError(f"{prefix}:\n{preview}")


def _validate_surface_forms(surface_forms: Any, report: MutableMapping[str, Any], entry_label: str, *, strict_schema: bool) -> None:
    if not isinstance(surface_forms, list):
        _add_validation_issue(report, "errors", f"{entry_label}: surface_forms must be a list")
        return
    if strict_schema and not surface_forms:
        _add_validation_issue(report, "errors", f"{entry_label}: surface_forms must not be empty in strict schema mode")
    seen: set[tuple[str, str]] = set()
    for index, form in enumerate(surface_forms):
        label = f"{entry_label}.surface_forms[{index}]"
        if not isinstance(form, Mapping):
            _add_validation_issue(report, "errors", f"{label}: surface form must be a mapping")
            continue
        text_value = _to_optional_str(form.get("text", form.get("surface", form.get("value"))))
        if not text_value:
            _add_validation_issue(report, "errors", f"{label}: text must be a non-empty string")
            continue
        kind_value = _to_optional_str(form.get("kind", form.get("type", "variant")))
        if not kind_value:
            _add_validation_issue(report, "errors", f"{label}: kind must be a non-empty string")
            continue
        dedupe_key = (text_value, kind_value)
        if dedupe_key in seen:
            _add_validation_issue(report, "warnings", f"{label}: duplicate surface form {dedupe_key!r}")
        seen.add(dedupe_key)


def _infer_relation_direction(relation: Mapping[str, Any], relation_rule: Mapping[str, Any] | None = None) -> str | None:
    explicit_direction = _to_optional_str(relation.get("direction"))
    if explicit_direction is not None:
        return explicit_direction
    if "bidirectional" in relation:
        return "bidirectional" if _to_bool(relation.get("bidirectional"), False) else "outbound"
    if relation_rule is not None:
        return _to_optional_str(relation_rule.get("direction"))
    return None


def _validate_relation_mapping(
    relation: Any,
    report: MutableMapping[str, Any],
    relation_label: str,
    *,
    existing_targets: Mapping[str, Any] | None = None,
    strict_schema: bool,
    strict_relations: bool,
    require_closed_relations: bool,
) -> None:
    if not isinstance(relation, Mapping):
        _add_validation_issue(report, "errors", f"{relation_label}: relation must be a mapping")
        return

    relation_type = _to_optional_str(relation.get("type"))
    relation_target = _to_optional_str(relation.get("target"))
    if not relation_type:
        _add_validation_issue(report, "errors", f"{relation_label}: type must be a non-empty string")
    if not relation_target:
        _add_validation_issue(report, "errors", f"{relation_label}: target must be a non-empty string")

    relation_rule = RELATION_TYPE_RULES.get(relation_type or "")
    if relation_type and relation_rule is None and strict_relations:
        _add_validation_issue(report, "errors", f"{relation_label}: unknown relation type {relation_type!r}")
    elif relation_type and relation_rule is None:
        _add_validation_issue(report, "warnings", f"{relation_label}: unknown relation type {relation_type!r}")

    weight_value = relation.get("weight")
    if weight_value is not None:
        try:
            weight = float(weight_value)
        except (TypeError, ValueError):
            _add_validation_issue(report, "errors", f"{relation_label}: weight must be numeric")
        else:
            if weight < 0.0 or weight > 1.0:
                _add_validation_issue(report, "warnings", f"{relation_label}: weight {weight!r} is outside the recommended range [0.0, 1.0]")
    elif strict_relations:
        _add_validation_issue(report, "warnings", f"{relation_label}: weight is strongly recommended in strict relation mode")

    direction = _infer_relation_direction(relation, relation_rule)
    if direction is None:
        if strict_relations:
            _add_validation_issue(report, "warnings", f"{relation_label}: direction is strongly recommended in strict relation mode")
    elif direction not in RELATION_DIRECTIONS:
        _add_validation_issue(report, "errors", f"{relation_label}: direction must be one of {sorted(RELATION_DIRECTIONS)!r}")
    elif relation_rule is not None and direction != relation_rule.get("direction"):
        _add_validation_issue(report, "warnings", f"{relation_label}: direction {direction!r} differs from the canonical rule for {relation_type!r} ({relation_rule.get('direction')!r})")

    confidence_value = relation.get("confidence")
    if confidence_value is not None:
        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            _add_validation_issue(report, "errors", f"{relation_label}: confidence must be numeric")
        else:
            if confidence < 0.0 or confidence > 1.0:
                _add_validation_issue(report, "warnings", f"{relation_label}: confidence {confidence!r} is outside the recommended range [0.0, 1.0]")
    elif strict_relations:
        _add_validation_issue(report, "warnings", f"{relation_label}: confidence is strongly recommended in strict relation mode")

    usage_stage = relation.get("usage_stage")
    if usage_stage is not None:
        if not isinstance(usage_stage, list):
            _add_validation_issue(report, "errors", f"{relation_label}: usage_stage must be a list when provided")
        else:
            stages = [_to_optional_str(value) for value in usage_stage]
            if any(stage is None for stage in stages):
                _add_validation_issue(report, "errors", f"{relation_label}: usage_stage must contain only non-empty strings")
            else:
                unknown_stages = sorted(stage for stage in stages if stage not in RELATION_USAGE_STAGES)
                if unknown_stages:
                    _add_validation_issue(report, "errors", f"{relation_label}: usage_stage contains unknown values {unknown_stages!r}")
                if relation_rule is not None:
                    canonical_stages = set(str(v) for v in relation_rule.get("usage_stage", []))
                    missing_overlap = canonical_stages.isdisjoint(set(stages))
                    if canonical_stages and missing_overlap:
                        _add_validation_issue(report, "warnings", f"{relation_label}: usage_stage {stages!r} does not overlap the canonical stages for {relation_type!r} ({sorted(canonical_stages)!r})")
    elif strict_relations:
        _add_validation_issue(report, "warnings", f"{relation_label}: usage_stage is strongly recommended in strict relation mode")

    relation_layer = _to_optional_str(relation.get("layer"))
    if relation_layer is not None:
        if relation_layer not in RELATION_LAYERS:
            _add_validation_issue(report, "errors", f"{relation_label}: layer must be one of {sorted(RELATION_LAYERS)!r}")
        elif relation_rule is not None and relation_layer != relation_rule.get("layer"):
            _add_validation_issue(report, "warnings", f"{relation_label}: layer {relation_layer!r} differs from the canonical layer for {relation_type!r} ({relation_rule.get('layer')!r})")

    inverse_type = _to_optional_str(relation.get("inverse_type"))
    if inverse_type is not None:
        if inverse_type not in RELATION_TYPE_RULES:
            _add_validation_issue(report, "warnings", f"{relation_label}: inverse_type {inverse_type!r} is not a registered relation type")
        elif relation_rule is not None and relation_rule.get("inverse_type") and inverse_type != relation_rule.get("inverse_type"):
            _add_validation_issue(report, "warnings", f"{relation_label}: inverse_type {inverse_type!r} differs from the canonical inverse for {relation_type!r} ({relation_rule.get('inverse_type')!r})")

    axes = relation.get("axes")
    if axes is not None and not isinstance(axes, Mapping):
        _add_validation_issue(report, "errors", f"{relation_label}: axes must be a mapping when provided")
    constraints = relation.get("constraints")
    if constraints is not None and not isinstance(constraints, Mapping):
        _add_validation_issue(report, "errors", f"{relation_label}: constraints must be a mapping when provided")

    meta = relation.get("meta")
    if meta is not None and not isinstance(meta, Mapping):
        _add_validation_issue(report, "errors", f"{relation_label}: meta must be a mapping when provided")
        meta = None
    source_value = _to_optional_str(relation.get("source"))
    if source_value is None and isinstance(meta, Mapping):
        source_value = _to_optional_str(meta.get("source"))
    if strict_relations and source_value is None:
        _add_validation_issue(report, "warnings", f"{relation_label}: source is strongly recommended in strict relation mode")

    if relation_target and existing_targets:
        if relation_target not in existing_targets:
            level = "errors" if require_closed_relations else "warnings"
            message = f"{relation_label}: target {relation_target!r} is not present in this container"
            _add_validation_issue(report, level, message)


def _validate_slot_frames_section(
    slot_frames: Any,
    report: MutableMapping[str, Any],
    *,
    strict_schema: bool,
) -> Dict[str, Dict[str, Any]]:
    if slot_frames is None:
        if strict_schema:
            _add_validation_issue(report, "errors", "strict schema mode requires top-level slot_frames")
        return {}
    if not isinstance(slot_frames, Mapping):
        _add_validation_issue(report, "errors", "slot_frames must be a mapping")
        return {}

    normalized: Dict[str, Dict[str, Any]] = {}
    for key, raw_frame in slot_frames.items():
        frame_key = str(key)
        label = f"slot_frames[{frame_key!r}]"
        if not isinstance(raw_frame, Mapping):
            _add_validation_issue(report, "errors", f"{label}: slot frame must be a mapping")
            continue
        frame = dict(raw_frame)
        frame_id = _to_optional_str(frame.get("id", frame_key))
        if not frame_id:
            _add_validation_issue(report, "errors", f"{label}: id must be a non-empty string")
            continue
        if strict_schema and _to_optional_str(frame.get("id")) is None:
            _add_validation_issue(report, "errors", f"{label}: strict schema mode requires an explicit id field")
        if _to_optional_str(frame.get("id")) not in {None, frame_key}:
            _add_validation_issue(report, "errors", f"{label}: id must match the mapping key")
        slots = frame.get("slots", [])
        if not isinstance(slots, list):
            _add_validation_issue(report, "errors", f"{label}: slots must be a list")
            continue
        if strict_schema and not slots:
            _add_validation_issue(report, "errors", f"{label}: strict schema mode requires at least one slot")
        seen_slot_names: set[str] = set()
        for index, raw_slot in enumerate(slots):
            slot_label = f"{label}.slots[{index}]"
            if not isinstance(raw_slot, Mapping):
                _add_validation_issue(report, "errors", f"{slot_label}: slot must be a mapping")
                continue
            slot_name = _to_optional_str(raw_slot.get("name"))
            if not slot_name:
                _add_validation_issue(report, "errors", f"{slot_label}: name must be a non-empty string")
                continue
            if slot_name in seen_slot_names:
                _add_validation_issue(report, "errors", f"{slot_label}: duplicate slot name {slot_name!r}")
            seen_slot_names.add(slot_name)
            required_value = raw_slot.get("required")
            if required_value is not None and not isinstance(required_value, bool):
                _add_validation_issue(report, "warnings", f"{slot_label}: required should be boolean")
        normalized[frame_key] = frame
    return normalized


def _validate_concepts_section(
    concepts: Any,
    report: MutableMapping[str, Any],
    *,
    strict_schema: bool,
    strict_relations: bool,
    require_closed_relations: bool,
    slot_frames: Mapping[str, Any] | None = None,
    declared_axes: List[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    if concepts is None:
        if strict_schema:
            _add_validation_issue(report, "errors", "strict schema mode requires top-level concepts")
        return {}
    if not isinstance(concepts, Mapping):
        _add_validation_issue(report, "errors", "concepts must be a mapping")
        return {}

    slot_frames = slot_frames or {}
    declared_axis_set = set(declared_axes or [])
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, raw_concept in concepts.items():
        concept_key = str(key)
        label = f"concepts[{concept_key!r}]"
        if not isinstance(raw_concept, Mapping):
            _add_validation_issue(report, "errors", f"{label}: concept must be a mapping")
            continue
        concept = dict(raw_concept)
        concept_id = _to_optional_str(concept.get("id", concept_key))
        if not concept_id:
            _add_validation_issue(report, "errors", f"{label}: id must be a non-empty string")
            continue
        if strict_schema and _to_optional_str(concept.get("id")) is None:
            _add_validation_issue(report, "errors", f"{label}: strict schema mode requires an explicit id field")
        if _to_optional_str(concept.get("id")) not in {None, concept_key}:
            _add_validation_issue(report, "errors", f"{label}: id must match the mapping key")
        if strict_schema and not _to_optional_str(concept.get("label")):
            _add_validation_issue(report, "errors", f"{label}: strict schema mode requires a non-empty label")
        axes = concept.get("axes", {})
        if axes is not None and not isinstance(axes, Mapping):
            _add_validation_issue(report, "errors", f"{label}: axes must be a mapping when provided")
        elif isinstance(axes, Mapping) and declared_axis_set:
            unknown_axes = sorted(str(axis) for axis in axes.keys() if str(axis) not in declared_axis_set)
            if unknown_axes:
                _add_validation_issue(report, "warnings", f"{label}: undeclared semantic axes {unknown_axes}")
        default_slot_frame_id = _to_optional_str(concept.get("default_slot_frame_id"))
        if default_slot_frame_id and slot_frames and default_slot_frame_id not in slot_frames:
            _add_validation_issue(report, "errors", f"{label}: default_slot_frame_id {default_slot_frame_id!r} is not defined in slot_frames")
        relations = concept.get("relations", [])
        if relations is not None and not isinstance(relations, list):
            _add_validation_issue(report, "errors", f"{label}: relations must be a list when provided")
        elif isinstance(relations, list):
            for index, relation in enumerate(relations):
                relation_label = f"{label}.relations[{index}]"
                _validate_relation_mapping(
                    relation,
                    report,
                    relation_label,
                    existing_targets=concepts,
                    strict_schema=strict_schema,
                    strict_relations=strict_relations,
                    require_closed_relations=require_closed_relations,
                )
        normalized[concept_key] = concept
    return normalized


def _collect_entry_concept_references(entry: Mapping[str, Any]) -> List[str]:
    concept_ids = _to_str_list(entry.get("concept_ids", []))
    senses = entry.get("senses", [])
    if isinstance(senses, list):
        for sense in senses:
            if isinstance(sense, Mapping):
                concept_ids.extend(_to_str_list(sense.get("concept_ids", [])))
    return _unique_keep_order([str(value) for value in concept_ids if _to_optional_str(value)])


def _validate_entries_section(
    entries: Any,
    report: MutableMapping[str, Any],
    *,
    strict_schema: bool,
    strict_relations: bool,
    require_closed_relations: bool,
    concepts: Mapping[str, Any] | None = None,
    slot_frames: Mapping[str, Any] | None = None,
    declared_axes: List[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(entries, Mapping):
        _add_validation_issue(report, "errors", "entries must be a mapping")
        return {}

    concepts = concepts or {}
    slot_frames = slot_frames or {}
    declared_axis_set = set(declared_axes or [])
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, raw_entry in entries.items():
        entry_key = str(key)
        label = f"entries[{entry_key!r}]"
        if not isinstance(raw_entry, Mapping):
            _add_validation_issue(report, "errors", f"{label}: entry must be a mapping")
            continue
        entry = dict(raw_entry)
        word = _to_optional_str(entry.get("word", entry.get("lemma", entry_key)))
        if not word:
            _add_validation_issue(report, "errors", f"{label}: word/lemma must be a non-empty string")
            continue
        if strict_schema and not entry.get("surface_forms"):
            _add_validation_issue(report, "errors", f"{label}: strict schema mode requires surface_forms")
        grammar = entry.get("grammar", {})
        if not isinstance(grammar, Mapping):
            _add_validation_issue(report, "errors", f"{label}: grammar must be a mapping")
        else:
            pos = _to_optional_str(grammar.get("pos"))
            if not pos:
                _add_validation_issue(report, "errors", f"{label}: grammar.pos must be a non-empty string")
        vector = entry.get("vector", {})
        if vector is not None and not isinstance(vector, Mapping):
            _add_validation_issue(report, "errors", f"{label}: vector must be a mapping when provided")
        elif isinstance(vector, Mapping) and declared_axis_set:
            unknown_axes = sorted(str(axis) for axis in vector.keys() if str(axis) not in declared_axis_set)
            if unknown_axes:
                _add_validation_issue(report, "warnings", f"{label}: undeclared semantic axes {unknown_axes}")
        surface_forms = entry.get("surface_forms", [])
        _validate_surface_forms(surface_forms, report, label, strict_schema=strict_schema)
        reading = entry.get("reading")
        if reading is not None and _to_optional_str(reading) is None:
            _add_validation_issue(report, "errors", f"{label}: reading must be a non-empty string when provided")
        concept_refs = _collect_entry_concept_references(entry)
        if strict_schema and not concept_refs:
            _add_validation_issue(report, "errors", f"{label}: strict schema mode requires at least one concept reference")
        for concept_id in concept_refs:
            if concepts and concept_id not in concepts:
                _add_validation_issue(report, "errors", f"{label}: concept reference {concept_id!r} is not defined in concepts")
        slot_frame_id = _to_optional_str(entry.get("slot_frame_id"))
        if slot_frame_id and slot_frames and slot_frame_id not in slot_frames:
            _add_validation_issue(report, "errors", f"{label}: slot_frame_id {slot_frame_id!r} is not defined in slot_frames")
        entry_relations = entry.get("relations")
        if entry_relations is not None:
            if not isinstance(entry_relations, list):
                _add_validation_issue(report, "errors", f"{label}: relations must be a list when provided")
            else:
                for index, relation in enumerate(entry_relations):
                    relation_label = f"{label}.relations[{index}]"
                    _validate_relation_mapping(
                        relation,
                        report,
                        relation_label,
                        existing_targets=concepts,
                        strict_schema=strict_schema,
                        strict_relations=strict_relations,
                        require_closed_relations=require_closed_relations,
                    )
        senses = entry.get("senses", [])
        if senses is not None and not isinstance(senses, list):
            _add_validation_issue(report, "errors", f"{label}: senses must be a list when provided")
        elif isinstance(senses, list):
            for index, sense in enumerate(senses):
                sense_label = f"{label}.senses[{index}]"
                if not isinstance(sense, Mapping):
                    _add_validation_issue(report, "errors", f"{sense_label}: sense must be a mapping")
                    continue
                sense_id = sense.get("id")
                if strict_schema and _to_optional_str(sense_id) is None:
                    _add_validation_issue(report, "errors", f"{sense_label}: strict schema mode requires a non-empty id")
                if "concept_ids" in sense:
                    if not isinstance(sense.get("concept_ids"), list):
                        _add_validation_issue(report, "errors", f"{sense_label}: concept_ids must be a list")
                    else:
                        for concept_id in _to_str_list(sense.get("concept_ids", [])):
                            if concepts and concept_id not in concepts:
                                _add_validation_issue(report, "errors", f"{sense_label}: concept reference {concept_id!r} is not defined in concepts")
                slot_override = _to_optional_str(sense.get("slot_frame_override", sense.get("slot_frame_id")))
                if slot_override and slot_frames and slot_override not in slot_frames:
                    _add_validation_issue(report, "errors", f"{sense_label}: slot_frame_override {slot_override!r} is not defined in slot_frames")
        normalized[entry_key] = entry
    return normalized


def collect_lexicon_validation_report(
    data: Mapping[str, Any],
    *,
    strict_schema: bool = False,
    strict_relations: bool = False,
    require_closed_relations: bool = False,
) -> Dict[str, Any]:
    report = _new_validation_report()
    if not isinstance(data, Mapping):
        _add_validation_issue(report, "errors", "lexicon container must be a mapping")
        return _finalize_validation_report(report)

    meta = data.get("meta", {})
    if meta is not None and not isinstance(meta, Mapping):
        _add_validation_issue(report, "errors", "meta must be a mapping")
        meta = {}
    indexes = data.get("indexes", {})
    if indexes is not None and not isinstance(indexes, Mapping):
        _add_validation_issue(report, "errors", "indexes must be a mapping")

    meta = dict(meta) if isinstance(meta, Mapping) else {}
    semantic_axes = meta.get("semantic_axes", [])
    if semantic_axes is None:
        semantic_axes = []
    if not isinstance(semantic_axes, list):
        _add_validation_issue(report, "errors", "meta.semantic_axes must be a list when provided")
        semantic_axes = []
    declared_axes = [str(axis) for axis in semantic_axes]

    if strict_schema and _to_optional_str(meta.get("schema_version")) is None:
        _add_validation_issue(report, "warnings", "strict schema mode recommends meta.schema_version")

    slot_frames = _validate_slot_frames_section(data.get("slot_frames"), report, strict_schema=strict_schema)
    concepts = _validate_concepts_section(
        data.get("concepts"),
        report,
        strict_schema=strict_schema,
        strict_relations=strict_relations,
        require_closed_relations=require_closed_relations,
        slot_frames=slot_frames,
        declared_axes=declared_axes,
    )
    entries = data.get("entries", {})
    if strict_schema and not entries:
        _add_validation_issue(report, "errors", "strict schema mode requires entries after normalization")
    _validate_entries_section(
        entries,
        report,
        strict_schema=strict_schema,
        strict_relations=strict_relations,
        require_closed_relations=require_closed_relations,
        concepts=concepts,
        slot_frames=slot_frames,
        declared_axes=declared_axes,
    )
    return _finalize_validation_report(report)


def validate_lexicon_container(
    data: Mapping[str, Any],
    *,
    strict_schema: bool = False,
    strict_relations: bool = False,
    require_closed_relations: bool = False,
) -> Dict[str, Any]:
    report = collect_lexicon_validation_report(
        data,
        strict_schema=strict_schema,
        strict_relations=strict_relations,
        require_closed_relations=require_closed_relations,
    )
    _raise_for_validation_errors(report)
    return report


def collect_raw_lexicon_validation_report(
    data: Mapping[str, Any],
    *,
    strict_schema: bool = False,
    strict_relations: bool = False,
    require_closed_relations: bool = False,
) -> Dict[str, Any]:
    report = _new_validation_report()
    if not isinstance(data, Mapping):
        _add_validation_issue(report, "errors", "raw lexicon container must be a mapping")
        return _finalize_validation_report(report)

    if strict_schema:
        required_top_level = ("meta", "concepts", "slot_frames", "lexical_entries")
        for field in required_top_level:
            if field not in data:
                _add_validation_issue(report, "errors", f"strict schema mode requires top-level field {field!r}")

    lexical_entries = data.get("lexical_entries")
    if lexical_entries is not None:
        if not isinstance(lexical_entries, Mapping):
            _add_validation_issue(report, "errors", "lexical_entries must be a mapping")
        else:
            for key, raw_entry in lexical_entries.items():
                entry_key = str(key)
                label = f"lexical_entries[{entry_key!r}]"
                if not isinstance(raw_entry, Mapping):
                    _add_validation_issue(report, "errors", f"{label}: entry must be a mapping")
                    continue
                entry_id = _to_optional_str(raw_entry.get("id"))
                lemma = _to_optional_str(raw_entry.get("lemma"))
                if strict_schema and not entry_id:
                    _add_validation_issue(report, "errors", f"{label}: strict schema mode requires id")
                if strict_schema and not lemma:
                    _add_validation_issue(report, "errors", f"{label}: strict schema mode requires lemma")
                if entry_id and entry_id != entry_key:
                    _add_validation_issue(report, "errors", f"{label}: id must match the mapping key")
                if raw_entry.get("surface_forms") is not None and not isinstance(raw_entry.get("surface_forms"), list):
                    _add_validation_issue(report, "errors", f"{label}: surface_forms must be a list when provided")
                if raw_entry.get("senses") is not None and not isinstance(raw_entry.get("senses"), list):
                    _add_validation_issue(report, "errors", f"{label}: senses must be a list when provided")
    elif "entries" not in data:
        _add_validation_issue(report, "errors", "raw lexicon container must include either lexical_entries or entries")

    normalized_report = collect_lexicon_validation_report(
        normalize_lexicon_container(data),
        strict_schema=strict_schema,
        strict_relations=strict_relations,
        require_closed_relations=require_closed_relations,
    )
    for level in ("errors", "warnings"):
        for message in normalized_report.get(level, []):
            _add_validation_issue(report, level, message)
    return _finalize_validation_report(report)


def validate_raw_lexicon_container(
    data: Mapping[str, Any],
    *,
    strict_schema: bool = False,
    strict_relations: bool = False,
    require_closed_relations: bool = False,
) -> Dict[str, Any]:
    report = collect_raw_lexicon_validation_report(
        data,
        strict_schema=strict_schema,
        strict_relations=strict_relations,
        require_closed_relations=require_closed_relations,
    )
    _raise_for_validation_errors(report, prefix="Raw lexicon validation failed")
    return report


def load_json_lexicon_container(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("Unsupported JSON lexicon format")
    return normalize_lexicon_container(data)


def collect_string_table(data: Dict[str, Any]) -> tuple[List[str], Dict[str, int]]:
    strings = set()

    meta = data.get("meta", {})
    entries = data.get("entries", {})

    def add(value: Any) -> None:
        if isinstance(value, str):
            strings.add(value)

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                add(key)
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
        elif isinstance(obj, str):
            add(obj)

    walk(meta)
    for entry_key, entry in entries.items():
        add(entry_key)
        add(entry.get("word"))
        add(entry.get("category"))

        vector = entry.get("vector", {})
        if isinstance(vector, Mapping):
            for key in vector.keys():
                add(key)

        grammar = _canonicalize_grammar(entry.get("grammar", {}))
        for key, value in grammar.items():
            add(key)
            if isinstance(value, str):
                add(value)
            elif isinstance(value, list):
                for x in value:
                    add(x)
            elif isinstance(value, dict):
                walk(value)

    table = sorted(strings)
    index = {s: i for i, s in enumerate(table)}
    return table, index


def encode_string_table(table: List[str]) -> bytes:
    out = bytearray()
    out += write_uvarint(len(table))
    for s in table:
        out += write_str(s)
    return bytes(out)


def decode_string_table(buf: io.BytesIO) -> List[str]:
    n = read_uvarint(buf)
    return [read_str(buf) for _ in range(n)]


def encode_string_id_list(values: List[str], s2i: Dict[str, int]) -> bytes:
    out = bytearray()
    out += write_uvarint(len(values))
    for value in values:
        out += write_uvarint(s2i[value])
    return bytes(out)


def decode_string_id_list(buf: io.BytesIO, table: List[str]) -> List[str]:
    n = read_uvarint(buf)
    out = []
    for _ in range(n):
        idx = read_uvarint(buf)
        out.append(table[idx])
    return out


def skip_string_id_list(buf: io.BytesIO) -> None:
    n = read_uvarint(buf)
    for _ in range(n):
        read_uvarint(buf)


def _canonicalize_minimal_entry(word: str, extras_entry: Mapping[str, Any] | None) -> Dict[str, Any]:
    raw = dict(extras_entry) if isinstance(extras_entry, Mapping) else {}
    senses = _canonicalize_senses(raw.get("senses", []))
    concept_ids = _derive_entry_concept_ids(raw, senses)
    entry: Dict[str, Any] = {
        "word": word,
        "lemma": str(raw.get("lemma", word)),
        "surface_forms": _canonicalize_surface_forms(raw.get("surface_forms", raw.get("surfaces", [])), word),
        "senses": senses,
        "concept_ids": concept_ids,
    }
    reading = _to_optional_str(raw.get("reading"))
    if reading is not None:
        entry["reading"] = reading
    slot_frame_id = _to_optional_str(raw.get("slot_frame_id", raw.get("slot_frame")))
    if slot_frame_id is not None:
        entry["slot_frame_id"] = slot_frame_id
    return entry


def _decode_binary_entry_minimal(
    buf: io.BytesIO,
    string_table: List[str],
    semantic_axes: List[str],
) -> Dict[str, Any]:
    word = string_table[read_uvarint(buf)]
    _ = string_table[read_uvarint(buf)]

    skip_bytes_with_len(buf)

    read_uvarint(buf)
    read_uvarint(buf)
    read_uvarint(buf)
    read_uvarint(buf)
    scalar_blob = buf.read(5)
    if len(scalar_blob) != 5:
        raise EOFError("Unexpected EOF while skipping grammar scalars")

    for _field in CORE_LIST_FIELDS:
        skip_string_id_list(buf)

    extras_blob = read_bytes_with_len(buf)
    extras = json.loads(extras_blob.decode("utf-8"))
    extras_entry = extras.get("entry") if isinstance(extras, Mapping) else None
    return _canonicalize_minimal_entry(word, extras_entry if isinstance(extras_entry, Mapping) else None)


def _build_flat_lexicon_container(
    *,
    meta: Mapping[str, Any],
    entries: Mapping[str, Dict[str, Any]],
    top_level_extras: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    container: Dict[str, Any] = {
        "meta": dict(meta),
        "entries": dict(entries),
    }
    extras = dict(top_level_extras) if isinstance(top_level_extras, Mapping) else {}
    if "indexes" not in extras or not isinstance(extras.get("indexes"), Mapping):
        extras["indexes"] = {}
    for key, value in extras.items():
        container[str(key)] = value
    return container


def _decode_binary_entry(
    buf: io.BytesIO,
    string_table: List[str],
    semantic_axes: List[str],
) -> Dict[str, Any]:
    word = string_table[read_uvarint(buf)]
    category = string_table[read_uvarint(buf)]

    vec_blob = read_bytes_with_len(buf)
    qvec = unpack_i16_list(vec_blob, len(semantic_axes))
    vector = {
        semantic_axes[i]: round(dequantize_i16_to_unit_float(qvec[i]), 6)
        for i in range(len(semantic_axes))
    }

    pos = string_table[read_uvarint(buf)]
    sub_pos = string_table[read_uvarint(buf)]
    conj_type = string_table[read_uvarint(buf)]
    conj_slot = string_table[read_uvarint(buf)]
    connectability = round(struct.unpack("<f", buf.read(4))[0], 6)

    flags = struct.unpack("<B", buf.read(1))[0]
    grammar = {
        "pos": pos,
        "sub_pos": sub_pos,
        "conjugation_type": conj_type,
        "conjugation_slot": conj_slot,
        "connectability": connectability,
    }
    for i, field in enumerate(CORE_BOOL_FIELDS):
        grammar[field] = bool(flags & (1 << i))
    for field in CORE_LIST_FIELDS:
        grammar[field] = decode_string_id_list(buf, string_table)

    extras_blob = read_bytes_with_len(buf)
    extras = json.loads(extras_blob.decode("utf-8"))
    if isinstance(extras.get("grammar"), Mapping):
        grammar.update(dict(extras["grammar"]))

    entry = {"word": word, "category": category, "vector": vector, "grammar": grammar}
    if isinstance(extras.get("entry"), Mapping):
        entry.update(dict(extras["entry"]))

    return _canonicalize_entry(word, entry)


def load_lsd_lexicon_container(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        file_magic = f.read(len(MAGIC))
        if file_magic != MAGIC:
            raise ValueError("Invalid file magic")
        version = struct.unpack("<H", f.read(2))[0]
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")
        raw_size = struct.unpack("<I", f.read(4))[0]
        compressed = f.read()

    raw = zlib.decompress(compressed)
    if len(raw) != raw_size:
        raise ValueError(f"Raw size mismatch: expected {raw_size}, got {len(raw)}")

    buf = io.BytesIO(raw)
    inner_magic = read_bytes_with_len(buf)
    if inner_magic != MAGIC:
        raise ValueError("Inner payload magic mismatch")
    inner_version = read_uvarint(buf)
    if inner_version != VERSION:
        raise ValueError(f"Inner payload version mismatch: {inner_version}")

    semantic_axes_n = read_uvarint(buf)
    semantic_axes_ids = [read_uvarint(buf) for _ in range(semantic_axes_n)]

    grammar_axes_n = read_uvarint(buf)
    grammar_axes_ids = [read_uvarint(buf) for _ in range(grammar_axes_n)]

    meta_blob = read_bytes_with_len(buf)
    meta = json.loads(meta_blob.decode("utf-8"))
    top_level_extras = meta.pop(TOP_LEVEL_BINARY_META_KEY, {})
    if not isinstance(top_level_extras, Mapping):
        top_level_extras = {}

    string_table = decode_string_table(buf)

    semantic_axes = [string_table[i] for i in semantic_axes_ids]
    grammar_axes = [string_table[i] for i in grammar_axes_ids]
    meta["semantic_axes"] = semantic_axes
    meta["grammar_axes"] = grammar_axes

    entry_count = read_uvarint(buf)
    entries: Dict[str, Any] = {}

    progress = ConsoleProgressBar(entry_count, title=f"Loading {path.name}", enabled=should_show_progress(path))
    for _ in range(entry_count):
        key = string_table[read_uvarint(buf)]
        entry = _decode_binary_entry(buf, string_table, semantic_axes)
        entries[key] = entry
        progress.update()
    progress.close()

    return normalize_lexicon_container({"meta": meta, "entries": entries, **dict(top_level_extras)})


class IndexedLSDLexicon(Mapping[str, Dict[str, Any]]):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fp = self.path.open("rb")
        self._mm = mmap.mmap(self._fp.fileno(), 0, access=mmap.ACCESS_READ)

        magic_len = len(INDEXED_MAGIC)
        self._magic = self._mm[0:magic_len]
        if self._magic != INDEXED_MAGIC:
            self.close()
            raise ValueError("Not an indexed LSD v2 file")

        self._version = struct.unpack_from("<H", self._mm, magic_len)[0]
        if self._version != INDEXED_VERSION:
            self.close()
            raise ValueError(f"Unsupported indexed version: {self._version}")

        self._meta_size = struct.unpack_from("<I", self._mm, magic_len + 2)[0]
        self._string_table_size = struct.unpack_from("<I", self._mm, magic_len + 6)[0]
        self._entry_count = struct.unpack_from("<I", self._mm, magic_len + 10)[0]

        self._meta_offset = magic_len + 14
        self._string_table_offset = self._meta_offset + self._meta_size
        self._index_offset = self._string_table_offset + self._string_table_size
        self._index_row_size = 16

        self._meta = json.loads(bytes(self._mm[self._meta_offset:self._meta_offset + self._meta_size]).decode("utf-8"))
        st_blob = bytes(self._mm[self._string_table_offset:self._string_table_offset + self._string_table_size])
        self._string_table = decode_string_table(io.BytesIO(st_blob))
        self._semantic_axes = list(self._meta.get("semantic_axes", []))

        self._keys: List[str] = []
        self._key_to_row: Dict[str, int] = {}
        for i in range(self._entry_count):
            row_off = self._index_offset + i * self._index_row_size
            key_sid, _rec_off, _rec_size = struct.unpack_from("<IQI", self._mm, row_off)
            key = self._string_table[key_sid]
            self._keys.append(key)
            self._key_to_row[key] = i

    def __enter__(self) -> "IndexedLSDLexicon":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def meta(self) -> Dict[str, Any]:
        return dict(self._meta)

    @property
    def axes(self) -> List[str]:
        return list(self._semantic_axes)

    def __len__(self) -> int:
        return self._entry_count

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def keys(self) -> List[str]:
        return list(self._keys)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._key_to_row

    def _row_info(self, row_index: int) -> tuple[int, int, int]:
        row_off = self._index_offset + row_index * self._index_row_size
        return struct.unpack_from("<IQI", self._mm, row_off)

    def _decode_record(self, rec_off: int, rec_size: int) -> Dict[str, Any]:
        rec = bytes(self._mm[rec_off:rec_off + rec_size])
        buf = io.BytesIO(rec)
        return _decode_binary_entry(buf, self._string_table, self._semantic_axes)

    def _decode_record_minimal(self, rec_off: int, rec_size: int) -> Dict[str, Any]:
        rec = bytes(self._mm[rec_off:rec_off + rec_size])
        buf = io.BytesIO(rec)
        return _decode_binary_entry_minimal(buf, self._string_table, self._semantic_axes)

    def iter_decoded_entries(self, *, lightweight: bool = False) -> Iterator[tuple[str, Dict[str, Any]]]:
        decoder = self._decode_record_minimal if lightweight else self._decode_record
        for row_index, key in enumerate(self._keys):
            _, rec_off, rec_size = self._row_info(row_index)
            yield key, decoder(rec_off, rec_size)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        row_index = self._key_to_row[key]
        _, rec_off, rec_size = self._row_info(row_index)
        return self._decode_record(rec_off, rec_size)

    def get(self, key: str, default: Any = None) -> Any:
        row_index = self._key_to_row.get(key)
        if row_index is None:
            return default
        _, rec_off, rec_size = self._row_info(row_index)
        return self._decode_record(rec_off, rec_size)

    def close(self) -> None:
        mm = getattr(self, "_mm", None)
        fp = getattr(self, "_fp", None)
        self._keys = []
        self._key_to_row = {}
        self._string_table = []
        self._semantic_axes = []
        self._meta = {}
        if mm is not None:
            try:
                mm.close()
            finally:
                self._mm = None
        if fp is not None:
            try:
                fp.close()
            finally:
                self._fp = None


def load_indexed_lsd_lexicon_container(
    path: str | Path,
    *,
    normalize: bool = True,
    lightweight: bool = False,
) -> Dict[str, Any]:
    with IndexedLSDLexicon(path) as lex:
        progress = ConsoleProgressBar(len(lex), title=f"Loading {Path(path).name}", enabled=should_show_progress(path))
        entries: Dict[str, Any] = {}
        for key, entry in lex.iter_decoded_entries(lightweight=lightweight):
            entries[key] = entry
            progress.update()
        progress.close()
        meta = lex.meta
        top_level_extras = meta.pop(TOP_LEVEL_BINARY_META_KEY, {})
        if not isinstance(top_level_extras, Mapping):
            top_level_extras = {}
        flat = _build_flat_lexicon_container(meta=meta, entries=entries, top_level_extras=top_level_extras)
        if not normalize:
            return flat
        return normalize_lexicon_container(flat)


def inspect_lexicon_storage(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    info: Dict[str, Any] = {
        "path": str(path),
        "suffix": suffix,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "storage": "json",
        "binary": False,
        "fully_indexed": False,
    }
    if suffix not in {".lsd", ".lsdx"}:
        return info
    with path.open("rb") as f:
        magic = f.read(max(len(INDEXED_MAGIC), len(MAGIC)))
    if magic[:len(INDEXED_MAGIC)] == INDEXED_MAGIC:
        info.update({"storage": "indexed_lsdx", "binary": True, "fully_indexed": True})
    elif magic[:len(MAGIC)] == MAGIC:
        info.update({"storage": "lsd", "binary": True, "fully_indexed": False})
    else:
        info.update({"storage": "unknown_binary", "binary": True, "fully_indexed": False})
    return info


def profile_lexicon_load(
    path: str | Path,
    *,
    sample_size: int = 128,
    skip_materialize: bool = False,
    lightweight_materialize: bool = False,
) -> Dict[str, Any]:
    path = Path(path)
    storage = inspect_lexicon_storage(path)
    results: Dict[str, Any] = {"storage": storage, "timing_ms": {}, "sample_size": int(sample_size)}

    detect_started = time.perf_counter()
    _ = inspect_lexicon_storage(path)
    results["timing_ms"]["detect_format_ms"] = round((time.perf_counter() - detect_started) * 1000.0, 3)

    if storage.get("storage") != "indexed_lsdx":
        materialize_started = time.perf_counter()
        container = load_lexicon_container(path)
        results["timing_ms"]["materialize_container_ms"] = round((time.perf_counter() - materialize_started) * 1000.0, 3)
        results["materialized"] = {
            "entry_count": len(container.get("entries", {})),
            "concept_count": len(container.get("concepts", {})),
            "slot_frame_count": len(container.get("slot_frames", {})),
        }
        return results

    open_started = time.perf_counter()
    with IndexedLSDLexicon(path) as lex:
        results["timing_ms"]["indexed_open_ms"] = round((time.perf_counter() - open_started) * 1000.0, 3)
        results["indexed_header"] = {
            "entry_count": len(lex),
            "semantic_axes": len(lex.axes),
            "key_count": len(lex.keys()),
        }

        sample_started = time.perf_counter()
        decoded_entries = 0
        for decoded_entries, (_key, _entry) in enumerate(lex.iter_decoded_entries(lightweight=lightweight_materialize), start=1):
            if decoded_entries >= max(int(sample_size), 0):
                break
        sample_elapsed_ms = (time.perf_counter() - sample_started) * 1000.0
        results["timing_ms"]["indexed_sample_decode_ms"] = round(sample_elapsed_ms, 3)
        results["sample"] = {
            "decoded_entries": decoded_entries,
            "avg_ms_per_entry": round(sample_elapsed_ms / max(decoded_entries, 1), 6),
        }

    if skip_materialize:
        results["materialized"] = None
        return results

    materialize_started = time.perf_counter()
    container = load_indexed_lsd_lexicon_container(path, normalize=not lightweight_materialize, lightweight=lightweight_materialize)
    results["timing_ms"]["materialize_container_ms"] = round((time.perf_counter() - materialize_started) * 1000.0, 3)
    results["materialized"] = {
        "entry_count": len(container.get("entries", {})),
        "concept_count": len(container.get("concepts", {})),
        "slot_frame_count": len(container.get("slot_frames", {})),
        "lightweight": bool(lightweight_materialize),
        "normalized": not bool(lightweight_materialize),
    }
    return results


def load_lexicon_container(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".lsd", ".lsdx"}:
        with path.open("rb") as f:
            magic = f.read(max(len(INDEXED_MAGIC), len(MAGIC)))
        if magic[:len(INDEXED_MAGIC)] == INDEXED_MAGIC:
            return load_indexed_lsd_lexicon_container(path)
        if magic[:len(MAGIC)] == MAGIC:
            return load_lsd_lexicon_container(path)
        raise ValueError(f"Unsupported lexicon binary format: {path}")

    return load_json_lexicon_container(path)


def load_lexicon_entries(path: str | Path) -> Dict[str, Dict[str, Any]]:
    return load_lexicon_container(path).get("entries", {})


def open_indexed_lexicon(path: str | Path) -> IndexedLSDLexicon:
    return IndexedLSDLexicon(path)


def export_entries_lexicon_container(data: Mapping[str, Any]) -> Dict[str, Any]:
    container = normalize_lexicon_container(data)
    validate_lexicon_container(container, strict_schema=False)
    exported = {
        "meta": dict(container.get("meta", {})),
        "entries": dict(container.get("entries", {})),
        "indexes": dict(container.get("indexes", {})),
    }
    if isinstance(container.get("concepts"), Mapping):
        exported["concepts"] = dict(container.get("concepts", {}))
    if isinstance(container.get("slot_frames"), Mapping):
        exported["slot_frames"] = dict(container.get("slot_frames", {}))
    for key, value in container.items():
        if key not in {"meta", "entries", "indexes", "lexicon", "lexical_entries", "concepts", "slot_frames"}:
            exported[str(key)] = value
    return exported


def export_lexical_entries_lexicon_container(data: Mapping[str, Any]) -> Dict[str, Any]:
    container = normalize_lexicon_container(data)
    validate_lexicon_container(container, strict_schema=False)

    lexical_entries: Dict[str, Dict[str, Any]] = {}
    for key, raw_entry in dict(container.get("entries", {})).items():
        entry_key = str(key)
        entry = _canonicalize_entry(entry_key, raw_entry)
        lemma = _to_optional_str(entry.get("lemma", entry.get("word", entry_key))) or entry_key
        entry_id = _to_optional_str(entry.get("id")) or f"lex:{lemma}"

        exported_entry: Dict[str, Any] = {
            "id": entry_id,
            "lemma": lemma,
            "surface_forms": _canonicalize_surface_forms(entry.get("surface_forms", []), lemma),
            "grammar": _canonicalize_grammar(entry.get("grammar", {})),
            "senses": _canonicalize_senses(entry.get("senses", [])),
            "style_tags": _to_str_list(entry.get("style_tags", [])),
            "frequency": _to_float(entry.get("frequency", 0.0), 0.0),
            "meta": dict(entry.get("meta", {})) if isinstance(entry.get("meta", {}), Mapping) else {},
        }

        reading = _to_optional_str(entry.get("reading"))
        if reading is not None:
            exported_entry["reading"] = reading

        slot_frame_id = _to_optional_str(entry.get("slot_frame_id"))
        if slot_frame_id is not None:
            exported_entry["slot_frame_id"] = slot_frame_id

        concept_ids = _unique_keep_order(_to_str_list(entry.get("concept_ids", [])))
        if concept_ids:
            exported_entry["concept_ids"] = concept_ids

        for field in ("category", "hierarchy", "vector", "slots", "relations"):
            value = entry.get(field)
            if value not in (None, [], {}):
                exported_entry[field] = value
            elif field in {"slots", "relations"} and isinstance(value, list):
                exported_entry[field] = list(value)
            elif field == "vector" and isinstance(value, Mapping):
                exported_entry[field] = dict(value)

        for field in ("word", "lemma", "id", "surface_forms", "grammar", "senses", "style_tags", "frequency", "meta", "reading", "slot_frame_id", "concept_ids"):
            entry.pop(field, None)
        for field in ("category", "hierarchy", "vector", "slots", "relations"):
            entry.pop(field, None)
        exported_entry.update(entry)
        lexical_entries[entry_id] = exported_entry

    exported: Dict[str, Any] = {
        "meta": dict(container.get("meta", {})),
        "concepts": dict(container.get("concepts", {})) if isinstance(container.get("concepts"), Mapping) else {},
        "slot_frames": dict(container.get("slot_frames", {})) if isinstance(container.get("slot_frames"), Mapping) else {},
        "lexical_entries": lexical_entries,
        "indexes": dict(container.get("indexes", {})),
    }
    for key, value in container.items():
        if key not in {"meta", "entries", "indexes", "lexicon", "lexical_entries", "concepts", "slot_frames"}:
            exported[str(key)] = value
    return exported


def export_hierarchical_lexicon_container(data: Mapping[str, Any]) -> Dict[str, Any]:
    container = normalize_lexicon_container(data)
    validate_lexicon_container(container, strict_schema=False)
    exported = dict(container)
    exported.pop("entries", None)
    return exported


def save_json_lexicon_container(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    save_obj = export_hierarchical_lexicon_container(data)
    with path.open("w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)


def _extract_binary_payload_fields(entry: Dict[str, Any]) -> tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    word = str(entry.get("word", ""))
    category = str(entry.get("category", "unknown"))
    vector = _canonicalize_vector(entry.get("vector", {}))
    grammar = _canonicalize_grammar(entry.get("grammar", {}))

    known_scalar_grammar_fields = set(CORE_SCALAR_FIELDS) | set(CORE_BOOL_FIELDS) | set(CORE_LIST_FIELDS)
    extra_grammar = {k: v for k, v in grammar.items() if k not in known_scalar_grammar_fields}
    extra_entry = {k: v for k, v in entry.items() if k not in {"word", "category", "vector", "grammar"}}
    return word, category, vector, grammar, extra_grammar, extra_entry


def save_lsd_lexicon_container(path: str | Path, data: Dict[str, Any], compress_level: int = 9) -> None:
    container = normalize_lexicon_container(data)
    validate_lexicon_container(container, strict_schema=False)
    meta = dict(container.get("meta", {}))
    entries: Dict[str, Any] = dict(container.get("entries", {}))
    top_level_extras = {
        str(key): value
        for key, value in container.items()
        if key not in {"meta", "entries", "indexes", "lexicon"}
    }
    if top_level_extras:
        meta[TOP_LEVEL_BINARY_META_KEY] = top_level_extras
    else:
        meta.pop(TOP_LEVEL_BINARY_META_KEY, None)
    semantic_axes: List[str] = list(meta.get("semantic_axes", []))
    grammar_axes: List[str] = list(meta.get("grammar_axes", []))

    if not semantic_axes:
        _ensure_meta({"meta": meta}, entries)
        semantic_axes = list(meta.get("semantic_axes", []))

    flat_container = {"meta": meta, "entries": entries}
    string_table, s2i = collect_string_table(flat_container)
    entry_keys = sorted(entries.keys())

    payload = bytearray()
    payload += write_bytes_with_len(MAGIC)
    payload += write_uvarint(VERSION)
    payload += write_uvarint(len(semantic_axes))
    for ax in semantic_axes:
        payload += write_uvarint(s2i[ax])
    payload += write_uvarint(len(grammar_axes))
    for ax in grammar_axes:
        payload += write_uvarint(s2i[ax])

    meta_blob = stable_json_dumps(meta).encode("utf-8")
    payload += write_bytes_with_len(meta_blob)
    payload += encode_string_table(string_table)
    payload += write_uvarint(len(entry_keys))

    progress = ConsoleProgressBar(max(len(entry_keys), 1), title=f"Saving {Path(path).name}", enabled=should_show_progress(path, min_bytes=0))
    for key in entry_keys:
        entry = entries[key]
        word, category, vector, grammar, extra_grammar, extra_entry = _extract_binary_payload_fields(entry)

        payload += write_uvarint(s2i[key])
        payload += write_uvarint(s2i[word])
        payload += write_uvarint(s2i[category])

        qvec = [quantize_unit_float_to_i16(vector.get(ax, 0.0)) for ax in semantic_axes]
        payload += write_bytes_with_len(pack_i16_list(qvec))

        pos = str(grammar.get("pos", ""))
        sub_pos = str(grammar.get("sub_pos", ""))
        conj_type = str(grammar.get("conjugation_type", "none"))
        conj_slot = str(grammar.get("conjugation_slot", "none"))
        connectability = float(grammar.get("connectability", 0.0))

        payload += write_uvarint(s2i[pos])
        payload += write_uvarint(s2i[sub_pos])
        payload += write_uvarint(s2i[conj_type])
        payload += write_uvarint(s2i[conj_slot])
        payload += struct.pack("<f", connectability)

        flags = 0
        for i, field in enumerate(CORE_BOOL_FIELDS):
            if bool(grammar.get(field, False)):
                flags |= (1 << i)
        payload += struct.pack("<B", flags)

        for field in CORE_LIST_FIELDS:
            values = grammar.get(field, [])
            if not isinstance(values, list):
                values = []
            payload += encode_string_id_list([str(v) for v in values], s2i)

        extras = {"entry": extra_entry, "grammar": extra_grammar}
        payload += write_bytes_with_len(stable_json_dumps(extras).encode("utf-8"))
        progress.update()

    compressed = zlib.compress(bytes(payload), level=compress_level)
    path = Path(path)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<H", VERSION))
        f.write(struct.pack("<I", len(payload)))
        f.write(compressed)
    progress.close()


def save_indexed_lsd_lexicon_container(path: str | Path, data: Dict[str, Any]) -> None:
    container = normalize_lexicon_container(data)
    validate_lexicon_container(container, strict_schema=False)
    meta = dict(container.get("meta", {}))
    entries: Dict[str, Any] = dict(container.get("entries", {}))
    top_level_extras = {
        str(key): value
        for key, value in container.items()
        if key not in {"meta", "entries", "indexes", "lexicon"}
    }
    if top_level_extras:
        meta[TOP_LEVEL_BINARY_META_KEY] = top_level_extras
    else:
        meta.pop(TOP_LEVEL_BINARY_META_KEY, None)
    semantic_axes: List[str] = list(meta.get("semantic_axes", []))
    if not semantic_axes:
        _ensure_meta({"meta": meta}, entries)
        semantic_axes = list(meta.get("semantic_axes", []))

    flat_container = {"meta": meta, "entries": entries}
    string_table, s2i = collect_string_table(flat_container)
    entry_keys = sorted(entries.keys())

    records = bytearray()
    index_rows: List[tuple[int, int, int]] = []
    progress = ConsoleProgressBar(max(len(entry_keys), 1), title=f"Saving {Path(path).name}", enabled=should_show_progress(path, min_bytes=0))

    for key in entry_keys:
        entry = entries[key]
        word, category, vector, grammar, extra_grammar, extra_entry = _extract_binary_payload_fields(entry)

        rec = bytearray()
        rec += write_uvarint(s2i[word])
        rec += write_uvarint(s2i[category])

        qvec = [quantize_unit_float_to_i16(vector.get(ax, 0.0)) for ax in semantic_axes]
        rec += write_bytes_with_len(pack_i16_list(qvec))

        pos = str(grammar.get("pos", ""))
        sub_pos = str(grammar.get("sub_pos", ""))
        conj_type = str(grammar.get("conjugation_type", "none"))
        conj_slot = str(grammar.get("conjugation_slot", "none"))
        connectability = float(grammar.get("connectability", 0.0))

        rec += write_uvarint(s2i[pos])
        rec += write_uvarint(s2i[sub_pos])
        rec += write_uvarint(s2i[conj_type])
        rec += write_uvarint(s2i[conj_slot])
        rec += struct.pack("<f", connectability)

        flags = 0
        for i, field in enumerate(CORE_BOOL_FIELDS):
            if bool(grammar.get(field, False)):
                flags |= (1 << i)
        rec += struct.pack("<B", flags)

        for field in CORE_LIST_FIELDS:
            values = grammar.get(field, [])
            if not isinstance(values, list):
                values = []
            rec += encode_string_id_list([str(v) for v in values], s2i)

        extras = {"entry": extra_entry, "grammar": extra_grammar}
        rec += write_bytes_with_len(stable_json_dumps(extras).encode("utf-8"))

        rec_off = len(records)
        rec_size = len(rec)
        records += rec
        index_rows.append((s2i[key], rec_off, rec_size))
        progress.update()

    meta_blob = stable_json_dumps(meta).encode("utf-8")
    st_blob = encode_string_table(string_table)
    header_size = len(INDEXED_MAGIC) + 2 + 4 + 4 + 4

    path = Path(path)
    with path.open("wb") as f:
        f.write(INDEXED_MAGIC)
        f.write(struct.pack("<H", INDEXED_VERSION))
        f.write(struct.pack("<I", len(meta_blob)))
        f.write(struct.pack("<I", len(st_blob)))
        f.write(struct.pack("<I", len(index_rows)))
        f.write(meta_blob)
        f.write(st_blob)

        base_records_offset = header_size + len(meta_blob) + len(st_blob) + len(index_rows) * 16
        for key_sid, rec_off, rec_size in index_rows:
            f.write(struct.pack("<IQI", key_sid, base_records_offset + rec_off, rec_size))
        f.write(records)
    progress.close()


def save_lexicon_container(path: str | Path, data: Dict[str, Any], compress_level: int = 9) -> None:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".lsd":
        save_lsd_lexicon_container(path, data, compress_level=compress_level)
    elif suffix == ".lsdx":
        save_indexed_lsd_lexicon_container(path, data)
    else:
        save_json_lexicon_container(path, data)