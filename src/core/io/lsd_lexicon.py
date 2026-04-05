from __future__ import annotations

import io
import json
import mmap
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Tuple

MAGIC = b"LSLMDICT"
VERSION = 1

INDEXED_MAGIC = b"LSLMDX2"
INDEXED_VERSION = 2


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


def _is_entry_mapping(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("vector"), Mapping)
        and "grammar" in value
    )


def _flatten_hierarchy_node(node: Any, entries: Dict[str, Dict[str, Any]]) -> None:
    if isinstance(node, Mapping):
        if _is_entry_mapping(node):
            word = str(node.get("word", "")).strip()
            if word:
                entries[word] = dict(node)
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
                entries[str(key)] = dict(value)
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

    indexes.setdefault("by_pos", {})
    indexes.setdefault("can_start", [])
    indexes.setdefault("can_end", [])
    indexes.setdefault("content_word", [])
    indexes.setdefault("function_word", [])
    indexes.setdefault("entry_path", {})

    by_pos: Dict[str, List[str]] = {}
    can_start: List[str] = []
    can_end: List[str] = []
    content_word: List[str] = []
    function_word: List[str] = []

    for word, entry in entries.items():
        grammar = entry.get("grammar", {})
        pos = str(grammar.get("pos", "unknown"))
        by_pos.setdefault(pos, []).append(word)
        if grammar.get("can_start", False):
            can_start.append(word)
        if grammar.get("can_end", False):
            can_end.append(word)
        if grammar.get("content_word", False):
            content_word.append(word)
        if grammar.get("function_word", False):
            function_word.append(word)

    indexes["by_pos"] = by_pos
    indexes["can_start"] = can_start
    indexes["can_end"] = can_end
    indexes["content_word"] = content_word
    indexes["function_word"] = function_word


def _ensure_meta(data: MutableMapping[str, Any], entries: Mapping[str, Dict[str, Any]]) -> None:
    meta = data.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta

    if "semantic_axes" not in meta:
        axes: List[str] = []
        for entry in entries.values():
            vector = entry.get("vector", {})
            if isinstance(vector, Mapping):
                axes.extend(str(k) for k in vector.keys())
        deduped: List[str] = []
        seen = set()
        for axis in axes:
            if axis not in seen:
                seen.add(axis)
                deduped.append(axis)
        meta["semantic_axes"] = deduped

    meta["entry_count"] = len(entries)


def build_hierarchical_container_from_entries(
    entries: Mapping[str, Dict[str, Any]],
    meta: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
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

    entry_path: Dict[str, str] = {}

    for word, entry in entries.items():
        hierarchy = list(entry.get("hierarchy", []))
        if hierarchy:
            path = ["lexicon"] + hierarchy
        else:
            pos = str(entry.get("grammar", {}).get("pos", "unknown"))
            if pos == "pronoun":
                path = ["lexicon", "content_words", "pronouns", "other"]
            elif pos in {"particle_case"}:
                path = ["lexicon", "function_words", "particles", "case"]
            elif pos in {"particle_binding"}:
                path = ["lexicon", "function_words", "particles", "binding"]
            elif pos in {"particle_conjunctive"}:
                path = ["lexicon", "function_words", "particles", "conjunctive"]
            elif pos in {"particle_sentence_final"}:
                path = ["lexicon", "function_words", "particles", "sentence_final"]
            elif pos == "auxiliary":
                path = ["lexicon", "function_words", "auxiliaries"]
            elif pos == "copula":
                path = ["lexicon", "function_words", "copulas"]
            elif pos == "iteration_mark":
                path = ["lexicon", "function_words", "special_marks"]
            elif pos in {"verb", "verb_stem"}:
                path = ["lexicon", "content_words", "verbs", "stems", "oov"]
            elif pos == "verb_suffix":
                path = ["lexicon", "content_words", "verbs", "suffixes"]
            elif pos == "adjective_i":
                path = ["lexicon", "content_words", "adjectives", "i", "oov"]
            elif pos in {"adjective_stem", "adjective_na_helper"}:
                if pos == "adjective_na_helper":
                    path = ["lexicon", "content_words", "adjectives", "na", "helper"]
                else:
                    path = ["lexicon", "content_words", "adjectives", "na", "stems", "oov"]
            elif pos == "adjective_i_ending":
                path = ["lexicon", "content_words", "adjectives", "i", "endings"]
            elif pos == "adverb":
                path = ["lexicon", "content_words", "adverbs", "oov"]
            elif pos == "conjunction":
                path = ["lexicon", "content_words", "conjunctions"]
            elif pos == "interjection":
                path = ["lexicon", "content_words", "interjections"]
            elif pos == "adnominal":
                path = ["lexicon", "content_words", "adnominals"]
            elif pos == "prefix":
                path = ["lexicon", "content_words", "prefixes"]
            elif pos == "suffix":
                path = ["lexicon", "content_words", "suffixes"]
            else:
                path = ["lexicon", "content_words", "nouns", "generated", "oov"]

        node = container
        for segment in path:
            node = node[segment]
        entry_copy = dict(entry)
        entry_copy["hierarchy"] = path[1:]
        node[word] = entry_copy
        entry_path[word] = "/".join(path + [word])

    container["indexes"]["entry_path"] = entry_path
    _ensure_indexes(container, entries)
    _ensure_meta(container, entries)
    return container


def load_json_lexicon_container(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError("Unsupported JSON lexicon format")

    entries = flatten_hierarchical_lexicon(data)
    container: Dict[str, Any] = dict(data)
    container["entries"] = entries
    _ensure_indexes(container, entries)
    _ensure_meta(container, entries)
    return container


def collect_string_table(data: Dict[str, Any]) -> tuple[List[str], Dict[str, int]]:
    strings = set()

    meta = data.get("meta", {})
    entries = data.get("entries", {})

    def add(s: Any) -> None:
        if isinstance(s, str):
            strings.add(s)

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                add(k)
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)
        elif isinstance(obj, str):
            add(obj)

    walk(meta)
    for entry_key, entry in entries.items():
        add(entry_key)
        add(entry.get("word"))
        add(entry.get("category"))

        vector = entry.get("vector", {})
        for k in vector.keys():
            add(k)

        grammar = entry.get("grammar", {})
        for k, v in grammar.items():
            add(k)
            if isinstance(v, str):
                add(v)
            elif isinstance(v, list):
                for x in v:
                    add(x)
            elif isinstance(v, dict):
                walk(v)

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
    for v in values:
        out += write_uvarint(s2i[v])
    return bytes(out)


def decode_string_id_list(buf: io.BytesIO, table: List[str]) -> List[str]:
    n = read_uvarint(buf)
    out = []
    for _ in range(n):
        idx = read_uvarint(buf)
        out.append(table[idx])
    return out


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

    string_table = decode_string_table(buf)

    semantic_axes = [string_table[i] for i in semantic_axes_ids]
    grammar_axes = [string_table[i] for i in grammar_axes_ids]
    meta["semantic_axes"] = semantic_axes
    meta["grammar_axes"] = grammar_axes

    entry_count = read_uvarint(buf)

    bool_fields = ["independent", "can_start", "can_end", "content_word", "function_word"]
    list_fields = ["roles", "requires_prev", "requires_next", "forbid_prev", "forbid_next"]

    entries: Dict[str, Any] = {}
    progress = ConsoleProgressBar(entry_count, title=f"Loading {path.name}", enabled=should_show_progress(path))
    for _ in range(entry_count):
        key = string_table[read_uvarint(buf)]
        word = string_table[read_uvarint(buf)]
        category = string_table[read_uvarint(buf)]

        vec_blob = read_bytes_with_len(buf)
        qvec = unpack_i16_list(vec_blob, len(semantic_axes))
        vector = {semantic_axes[i]: round(dequantize_i16_to_unit_float(qvec[i]), 6) for i in range(len(semantic_axes))}

        pos = string_table[read_uvarint(buf)]
        subpos = string_table[read_uvarint(buf)]
        conj_type = string_table[read_uvarint(buf)]
        conj_slot = string_table[read_uvarint(buf)]
        connectability = round(struct.unpack("<f", buf.read(4))[0], 6)

        flags = struct.unpack("<B", buf.read(1))[0]
        grammar = {
            "pos": pos,
            "subpos": subpos,
            "conjugation_type": conj_type,
            "conjugation_slot": conj_slot,
            "connectability": connectability,
        }
        for i, bf in enumerate(bool_fields):
            grammar[bf] = bool(flags & (1 << i))
        for lf in list_fields:
            grammar[lf] = decode_string_id_list(buf, string_table)

        extras_blob = read_bytes_with_len(buf)
        extras = json.loads(extras_blob.decode("utf-8"))
        if extras.get("grammar"):
            grammar.update(extras["grammar"])

        entry = {"word": word, "category": category, "vector": vector, "grammar": grammar}
        if extras.get("entry"):
            entry.update(extras["entry"])
        entries[key] = entry
        progress.update()

    progress.close()
    container = {"meta": meta, "entries": entries}
    return build_hierarchical_container_from_entries(entries, meta=meta) | {"entries": entries}


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

        word = self._string_table[read_uvarint(buf)]
        category = self._string_table[read_uvarint(buf)]

        vec_blob = read_bytes_with_len(buf)
        qvec = unpack_i16_list(vec_blob, len(self._semantic_axes))
        vector = {self._semantic_axes[i]: round(dequantize_i16_to_unit_float(qvec[i]), 6) for i in range(len(self._semantic_axes))}

        pos = self._string_table[read_uvarint(buf)]
        subpos = self._string_table[read_uvarint(buf)]
        conj_type = self._string_table[read_uvarint(buf)]
        conj_slot = self._string_table[read_uvarint(buf)]
        connectability = round(struct.unpack("<f", buf.read(4))[0], 6)

        bool_fields = ["independent", "can_start", "can_end", "content_word", "function_word"]
        list_fields = ["roles", "requires_prev", "requires_next", "forbid_prev", "forbid_next"]

        flags = struct.unpack("<B", buf.read(1))[0]
        grammar = {
            "pos": pos,
            "subpos": subpos,
            "conjugation_type": conj_type,
            "conjugation_slot": conj_slot,
            "connectability": connectability,
        }
        for i, bf in enumerate(bool_fields):
            grammar[bf] = bool(flags & (1 << i))
        for lf in list_fields:
            grammar[lf] = decode_string_id_list(buf, self._string_table)

        extras_blob = read_bytes_with_len(buf)
        extras = json.loads(extras_blob.decode("utf-8"))
        if extras.get("grammar"):
            grammar.update(extras["grammar"])

        entry = {"word": word, "category": category, "vector": vector, "grammar": grammar}
        if extras.get("entry"):
            entry.update(extras["entry"])
        return entry

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


def load_indexed_lsd_lexicon_container(path: str | Path) -> Dict[str, Any]:
    with IndexedLSDLexicon(path) as lex:
        progress = ConsoleProgressBar(len(lex), title=f"Loading {Path(path).name}", enabled=should_show_progress(path))
        entries: Dict[str, Any] = {}
        for k in lex.keys():
            entries[k] = lex[k]
            progress.update()
        progress.close()
        return build_hierarchical_container_from_entries(entries, meta=lex.meta) | {"entries": entries}


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


def save_json_lexicon_container(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    entries = data.get("entries", {})
    if ("lexicon" not in data or "indexes" not in data) and isinstance(entries, Mapping):
        data = build_hierarchical_container_from_entries(entries, meta=data.get("meta", {})) | {"entries": dict(entries)}
    save_obj = dict(data)
    save_obj.pop("entries", None)
    with path.open("w", encoding="utf-8") as f:
        json.dump(save_obj, f, ensure_ascii=False, indent=2)


def save_lsd_lexicon_container(path: str | Path, data: Dict[str, Any], compress_level: int = 9) -> None:
    meta = dict(data.get("meta", {}))
    entries: Dict[str, Any] = dict(data.get("entries", {}))
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

    bool_fields = ["independent", "can_start", "can_end", "content_word", "function_word"]
    list_fields = ["roles", "requires_prev", "requires_next", "forbid_prev", "forbid_next"]
    known_scalar_grammar_fields = {"pos", "subpos", "conjugation_type", "conjugation_slot", "connectability", *bool_fields, *list_fields}

    progress = ConsoleProgressBar(max(len(entry_keys), 1), title=f"Saving {Path(path).name}", enabled=should_show_progress(path, min_bytes=0))
    for key in entry_keys:
        entry = entries[key]
        word = entry.get("word", key)
        category = entry.get("category", "")
        vector = entry.get("vector", {})
        grammar = entry.get("grammar", {})

        payload += write_uvarint(s2i[key])
        payload += write_uvarint(s2i[word])
        payload += write_uvarint(s2i[category])

        qvec = [quantize_unit_float_to_i16(vector.get(ax, 0.0)) for ax in semantic_axes]
        payload += write_bytes_with_len(pack_i16_list(qvec))

        pos = grammar.get("pos", "")
        subpos = grammar.get("subpos", "")
        conj_type = grammar.get("conjugation_type", "none")
        conj_slot = grammar.get("conjugation_slot", "none")
        connectability = float(grammar.get("connectability", 0.0))

        payload += write_uvarint(s2i[pos] if pos in s2i else 0)
        payload += write_uvarint(s2i[subpos] if subpos in s2i else 0)
        payload += write_uvarint(s2i[conj_type] if conj_type in s2i else 0)
        payload += write_uvarint(s2i[conj_slot] if conj_slot in s2i else 0)
        payload += struct.pack("<f", connectability)

        flags = 0
        for i, bf in enumerate(bool_fields):
            if bool(grammar.get(bf, False)):
                flags |= (1 << i)
        payload += struct.pack("<B", flags)

        for lf in list_fields:
            values = grammar.get(lf, [])
            if not isinstance(values, list):
                values = []
            payload += encode_string_id_list(values, s2i)

        extra_grammar = {k: v for k, v in grammar.items() if k not in known_scalar_grammar_fields}
        extra_entry = {k: v for k, v in entry.items() if k not in {"word", "category", "vector", "grammar"}}
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
    meta = dict(data.get("meta", {}))
    entries: Dict[str, Any] = dict(data.get("entries", {}))
    semantic_axes: List[str] = list(meta.get("semantic_axes", []))
    if not semantic_axes:
        _ensure_meta({"meta": meta}, entries)
        semantic_axes = list(meta.get("semantic_axes", []))

    flat_container = {"meta": meta, "entries": entries}
    string_table, s2i = collect_string_table(flat_container)
    entry_keys = sorted(entries.keys())

    bool_fields = ["independent", "can_start", "can_end", "content_word", "function_word"]
    list_fields = ["roles", "requires_prev", "requires_next", "forbid_prev", "forbid_next"]
    known_scalar_grammar_fields = {"pos", "subpos", "conjugation_type", "conjugation_slot", "connectability", *bool_fields, *list_fields}

    records = bytearray()
    index_rows: List[tuple[int, int, int]] = []
    progress = ConsoleProgressBar(max(len(entry_keys), 1), title=f"Saving {Path(path).name}", enabled=should_show_progress(path, min_bytes=0))

    for key in entry_keys:
        entry = entries[key]
        word = entry.get("word", key)
        category = entry.get("category", "")
        vector = entry.get("vector", {})
        grammar = entry.get("grammar", {})

        rec = bytearray()
        rec += write_uvarint(s2i[word])
        rec += write_uvarint(s2i[category])

        qvec = [quantize_unit_float_to_i16(vector.get(ax, 0.0)) for ax in semantic_axes]
        rec += write_bytes_with_len(pack_i16_list(qvec))

        pos = grammar.get("pos", "")
        subpos = grammar.get("subpos", "")
        conj_type = grammar.get("conjugation_type", "none")
        conj_slot = grammar.get("conjugation_slot", "none")
        connectability = float(grammar.get("connectability", 0.0))

        rec += write_uvarint(s2i[pos] if pos in s2i else 0)
        rec += write_uvarint(s2i[subpos] if subpos in s2i else 0)
        rec += write_uvarint(s2i[conj_type] if conj_type in s2i else 0)
        rec += write_uvarint(s2i[conj_slot] if conj_slot in s2i else 0)
        rec += struct.pack("<f", connectability)

        flags = 0
        for i, bf in enumerate(bool_fields):
            if bool(grammar.get(bf, False)):
                flags |= (1 << i)
        rec += struct.pack("<B", flags)

        for lf in list_fields:
            values = grammar.get(lf, [])
            if not isinstance(values, list):
                values = []
            rec += encode_string_id_list(values, s2i)

        extra_grammar = {k: v for k, v in grammar.items() if k not in known_scalar_grammar_fields}
        extra_entry = {k: v for k, v in entry.items() if k not in {"word", "category", "vector", "grammar"}}
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
