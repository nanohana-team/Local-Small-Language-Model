from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from src.core.io.lsd_lexicon import load_lexicon_entries

_LEXICON_CACHE: Dict[tuple[str, int, int], Dict[str, Dict[str, Any]]] = {}

DEFAULT_FALLBACK_CATEGORY = "unknown"
DEFAULT_DEPENDENCY_MODE = "strict"

_AXIS_ALIASES = {"relation": "relational"}
_PREFERRED_AXIS_ORDER = (
    "syntax_phase",
    "semantic_phase",
    "temporal_dynamic",
    "agency_causality",
    "abstraction",
    "relational",
    "valence",
    "intensity_urgency",
    "social",
    "formality",
)

_CONTENT_POS = {
    "noun", "pronoun",
    "verb", "verb_stem",
    "adjective_i", "adjective_na", "adjective_stem", "adjective_i_ending",
    "adverb", "interjection", "conjunction", "adnominal",
}
_FUNCTION_POS = {
    "particle", "particle_case", "particle_binding", "particle_conjunctive", "particle_sentence_final",
    "auxiliary", "copula", "suffix", "prefix", "verb_suffix", "adjective_na_helper", "iteration_mark",
}

UnknownWordResolver = Callable[[str], Optional[Dict[str, Any]]]
UnknownWordPersistor = Callable[[Dict[str, Dict[str, Any]]], None]


class DivergencePrimitive:
    def __init__(
        self,
        lexicon: Mapping[str, Any],
        weights: Dict[str, float] | None = None,
        random_seed: int | None = None,
        randomness: float = 0.04,
        candidate_expansion: int = 2,
        seed_anchor_strength: float = 0.95,
        current_anchor_strength: float = 0.60,
        lexical_anchor_strength: float = 0.55,
        keep_input_bonus: float = 0.45,
        max_content_pool: int = 600,
        max_function_pool: int = 80,
        per_pos_limit: int = 96,
        shortlist_factor: int = 6,
    ) -> None:
        self.lexicon = self.normalize_lexicon(lexicon)
        self.axes = self._infer_axes(self.lexicon)
        self.weights = {axis: 1.0 for axis in self.axes}
        if weights:
            for key, value in weights.items():
                norm_key = self._normalize_axis_name(key)
                if norm_key in self.weights:
                    self.weights[norm_key] = float(value)

        self.randomness = max(0.0, float(randomness))
        self.candidate_expansion = max(1, int(candidate_expansion))
        self.seed_anchor_strength = max(0.0, float(seed_anchor_strength))
        self.current_anchor_strength = max(0.0, float(current_anchor_strength))
        self.lexical_anchor_strength = max(0.0, float(lexical_anchor_strength))
        self.keep_input_bonus = max(0.0, float(keep_input_bonus))
        self.max_content_pool = max(120, int(max_content_pool))
        self.max_function_pool = max(12, int(max_function_pool))
        self.per_pos_limit = max(20, int(per_pos_limit))
        self.shortlist_factor = max(2, int(shortlist_factor))
        self.rng = random.Random(random_seed)

        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        self.category_centroids = self._build_category_centroids()
        self.words_by_pos = self._build_words_by_pos()
        self.content_words = [w for w in self.lexicon if self.grammar_of(w).get("content_word", False)]
        self.function_words = [w for w in self.lexicon if self.grammar_of(w).get("function_word", False)]

        self.word_vectors: Dict[str, Tuple[float, ...]] = {}
        self.word_axis_sums: Dict[str, float] = {}
        self.word_char_sets: Dict[str, set[str]] = {}
        for word, entry in self.lexicon.items():
            vec = entry.get("vector", {})
            tup = tuple(float(vec.get(axis, 0.0)) for axis in self.axes)
            self.word_vectors[word] = tup
            self.word_axis_sums[word] = sum(tup)
            self.word_char_sets[word] = set(word)

    @classmethod
    def load_lexicon(cls, path: str | Path) -> Dict[str, Dict[str, Any]]:
        path = Path(path)
        stat = path.stat()
        cache_key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
        cached = _LEXICON_CACHE.get(cache_key)
        if cached is not None:
            return cached
        raw = load_lexicon_entries(path)
        normalized = cls.normalize_lexicon(raw)
        _LEXICON_CACHE.clear()
        _LEXICON_CACHE[cache_key] = normalized
        return normalized

    @classmethod
    def normalize_lexicon(cls, raw_lexicon: Mapping[str, Any] | List[Any]) -> Dict[str, Dict[str, Any]]:
        if isinstance(raw_lexicon, list):
            items = []
            for item in raw_lexicon:
                if not isinstance(item, Mapping):
                    continue
                word = str(item.get("word", "")).strip()
                if not word:
                    continue
                items.append((word, item))
        elif isinstance(raw_lexicon, Mapping):
            if "entries" in raw_lexicon and isinstance(raw_lexicon["entries"], Mapping):
                items = list(raw_lexicon["entries"].items())
            elif "lexicon" in raw_lexicon and isinstance(raw_lexicon["lexicon"], Mapping):
                items = list(raw_lexicon["lexicon"].items())
            else:
                items = list(raw_lexicon.items())
        else:
            raise TypeError("Unsupported lexicon format")

        normalized: Dict[str, Dict[str, Any]] = {}
        for raw_word, raw_entry in items:
            if not isinstance(raw_entry, Mapping):
                continue
            word = str(raw_entry.get("word", raw_word)).strip()
            if not word:
                continue

            raw_vector = raw_entry.get("vector") or raw_entry.get("axes") or raw_entry.get("params")
            if not isinstance(raw_vector, Mapping):
                continue

            vector = {
                cls._normalize_axis_name(axis): float(value)
                for axis, value in raw_vector.items()
                if cls._normalize_axis_name(axis)
            }
            if not vector:
                continue

            grammar = cls._normalize_grammar(raw_entry.get("grammar", {}), raw_entry)
            category = str(raw_entry.get("category", DEFAULT_FALLBACK_CATEGORY)).strip() or DEFAULT_FALLBACK_CATEGORY
            entry = {"word": word, "category": category, "vector": vector, "grammar": grammar}
            if "hierarchy" in raw_entry:
                entry["hierarchy"] = list(raw_entry.get("hierarchy", []))
            normalized[word] = entry
        return normalized

    @staticmethod
    def _normalize_axis_name(axis: str) -> str:
        axis = str(axis).strip()
        return _AXIS_ALIASES.get(axis, axis)

    @classmethod
    def _infer_axes(cls, lexicon: Mapping[str, Dict[str, Any]]) -> Tuple[str, ...]:
        axis_set = set()
        for entry in lexicon.values():
            vector = entry.get("vector", {})
            if isinstance(vector, Mapping):
                axis_set.update(cls._normalize_axis_name(k) for k in vector.keys())
        ordered = [axis for axis in _PREFERRED_AXIS_ORDER if axis in axis_set]
        extras = sorted(axis for axis in axis_set if axis not in set(ordered))
        return tuple(ordered + extras)

    @staticmethod
    def _to_string_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @classmethod
    def _normalize_grammar(cls, raw_grammar: Any, raw_entry: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        raw_grammar = raw_grammar if isinstance(raw_grammar, Mapping) else {}
        raw_entry = raw_entry if isinstance(raw_entry, Mapping) else {}
        pos = str(raw_grammar.get("pos", raw_entry.get("pos", DEFAULT_FALLBACK_CATEGORY))).strip() or DEFAULT_FALLBACK_CATEGORY
        subpos = str(raw_grammar.get("subpos", raw_entry.get("subpos", ""))).strip()
        independent = bool(raw_grammar.get("independent", pos in _CONTENT_POS))
        dependency_mode = str(raw_grammar.get("dependency_mode", DEFAULT_DEPENDENCY_MODE)).strip() or DEFAULT_DEPENDENCY_MODE
        return {
            "pos": pos,
            "subpos": subpos,
            "independent": independent,
            "dependency_mode": dependency_mode,
            "can_start": bool(raw_grammar.get("can_start", pos not in _FUNCTION_POS)),
            "can_end": bool(
                raw_grammar.get(
                    "can_end",
                    pos in {"noun", "pronoun", "verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula", "auxiliary", "interjection"},
                )
            ),
            "content_word": bool(raw_grammar.get("content_word", pos in _CONTENT_POS)),
            "function_word": bool(raw_grammar.get("function_word", pos in _FUNCTION_POS)),
            "requires_prev": cls._to_string_list(raw_grammar.get("requires_prev")),
            "requires_next": cls._to_string_list(raw_grammar.get("requires_next")),
            "forbid_prev": cls._to_string_list(raw_grammar.get("forbid_prev")),
            "forbid_next": cls._to_string_list(raw_grammar.get("forbid_next")),
            "roles": cls._to_string_list(raw_grammar.get("roles")),
            "forms": cls._to_string_list(raw_grammar.get("forms")),
            "connectability": float(raw_grammar.get("connectability", 0.5)),
        }

    def _build_category_centroids(self) -> Dict[str, Dict[str, float]]:
        buckets: Dict[str, List[Dict[str, float]]] = {}
        for entry in self.lexicon.values():
            category = str(entry.get("category", DEFAULT_FALLBACK_CATEGORY))
            vec = entry.get("vector", {})
            if not vec:
                continue
            buckets.setdefault(category, []).append({axis: float(vec.get(axis, 0.0)) for axis in self.axes})

        centroids: Dict[str, Dict[str, float]] = {}
        for category, vectors in buckets.items():
            centroids[category] = {axis: sum(v[axis] for v in vectors) / len(vectors) for axis in self.axes}
        return centroids

    def _build_words_by_pos(self) -> Dict[str, List[str]]:
        buckets: Dict[str, List[str]] = {}
        for word, entry in self.lexicon.items():
            pos = str(entry.get("grammar", {}).get("pos", DEFAULT_FALLBACK_CATEGORY))
            buckets.setdefault(pos, []).append(word)
        return buckets

    def has_word(self, word: str) -> bool:
        return word in self.lexicon

    def normalize_entry_for_word(self, word: str, raw_entry: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("raw_entry must be a mapping")

        raw_word = str(raw_entry.get("word", word)).strip() or str(word).strip()
        if not raw_word:
            raise ValueError("word is required")

        raw_vector = raw_entry.get("vector") or raw_entry.get("axes") or raw_entry.get("params")
        if not isinstance(raw_vector, Mapping):
            raise ValueError(f"missing vector for word={raw_word}")

        vector = {
            self._normalize_axis_name(axis): float(value)
            for axis, value in raw_vector.items()
            if self._normalize_axis_name(axis)
        }
        if not vector:
            raise ValueError(f"empty vector for word={raw_word}")

        normalized_vector: Dict[str, float] = {}
        for axis in self.axes:
            normalized_vector[axis] = float(vector.get(axis, 0.0))

        grammar = self._normalize_grammar(raw_entry.get("grammar", {}), raw_entry)
        category = str(raw_entry.get("category", self.guess_category(raw_word))).strip() or self.guess_category(raw_word)

        entry: Dict[str, Any] = {
            "word": raw_word,
            "category": category,
            "vector": normalized_vector,
            "grammar": grammar,
        }
        if "hierarchy" in raw_entry:
            entry["hierarchy"] = list(raw_entry.get("hierarchy", []))
        return entry

    def register_word(self, word: str, raw_entry: Mapping[str, Any]) -> Dict[str, Any]:
        entry = self.normalize_entry_for_word(word, raw_entry)
        self.lexicon[entry["word"]] = entry
        self._rebuild_indices()
        return entry

    def register_words(self, entries: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        added: Dict[str, Dict[str, Any]] = {}
        for word, raw_entry in entries.items():
            try:
                entry = self.normalize_entry_for_word(word, raw_entry)
            except Exception:
                continue
            self.lexicon[entry["word"]] = entry
            added[entry["word"]] = entry

        if added:
            self._rebuild_indices()
        return added

    def resolve_unknown_words(
        self,
        words: Iterable[str],
        resolver: UnknownWordResolver | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        if resolver is None:
            return {}

        added: Dict[str, Dict[str, Any]] = {}
        unknowns = [str(w) for w in words if str(w) and not self.has_word(str(w))]
        if not unknowns:
            return added

        for word in list(dict.fromkeys(unknowns)):
            try:
                resolved = resolver(word)
            except Exception as e:
                print(f"[UNKNOWN][RESOLVE][ERROR] word={word} error={e}", flush=True)
                continue

            if not resolved:
                print(f"[UNKNOWN][RESOLVE][MISS] word={word}", flush=True)
                continue

            try:
                entry = self.register_word(word, resolved)
            except Exception as e:
                print(f"[UNKNOWN][REGISTER][ERROR] word={word} error={e}", flush=True)
                continue

            added[word] = entry
            print(
                f"[UNKNOWN][REGISTERED] word={word} "
                f"category={entry.get('category')} pos={entry.get('grammar', {}).get('pos')}",
                flush=True,
            )

        return added

    def entry_of(self, word: str) -> Dict[str, Any]:
        if word in self.lexicon:
            return self.lexicon[word]
        return {
            "category": self.guess_category(word),
            "vector": self.estimate_oov_vector(word),
            "grammar": self.guess_grammar(word),
        }

    def grammar_of(self, word: str) -> Dict[str, Any]:
        return dict(self.entry_of(word).get("grammar", {}))

    def pos_of(self, word: str) -> str:
        return str(self.grammar_of(word).get("pos", DEFAULT_FALLBACK_CATEGORY))

    def category_of(self, word: str) -> str:
        if word in self.lexicon:
            return str(self.lexicon[word].get("category", DEFAULT_FALLBACK_CATEGORY))
        return self.guess_category(word)

    def guess_category(self, word: str) -> str:
        pos = self.guess_grammar(word).get("pos", DEFAULT_FALLBACK_CATEGORY)
        mapping = {
            "noun": "noun",
            "pronoun": "pronoun",
            "verb": "verb_stem",
            "verb_stem": "verb_stem",
            "verb_suffix": "verb_suffix",
            "adjective_i": "adjective_stem",
            "adjective_na": "adjective_stem",
            "adjective_stem": "adjective_stem",
            "adjective_i_ending": "adjective_i_ending",
            "adjective_na_helper": "adjective_na_helper",
            "adverb": "adverb",
            "particle": "particle",
            "particle_case": "particle",
            "particle_binding": "particle",
            "particle_conjunctive": "particle",
            "particle_sentence_final": "particle",
            "auxiliary": "auxiliary",
            "copula": "copula",
            "interjection": "interjection",
            "conjunction": "conjunction",
            "adnominal": "adnominal",
            "suffix": "suffix",
            "prefix": "prefix",
            "iteration_mark": "repetition_mark",
        }
        return mapping.get(str(pos), DEFAULT_FALLBACK_CATEGORY)

    def guess_grammar(self, word: str) -> Dict[str, Any]:
        particles_case = {"が", "を", "に", "へ", "と", "より", "から", "まで", "で", "の"}
        particles_binding = {"は", "も", "こそ", "しか", "でも", "さえ", "だけ", "など"}
        particles_sentence = {"ね", "よ", "な", "か", "や", "ぞ", "わ", "さ", "かな", "かも"}
        particles_conjunctive = {"て", "で", "つつ", "ながら"}
        copula = {"だ", "です", "である", "だった", "でしょう", "ではない"}
        auxiliaries = {"たい", "ない", "た", "ます", "う", "よう", "らしい", "そうだ", "べきだ", "れる", "られる", "ている", "ていく", "ておく", "てみる", "てしまう", "ぬ"}
        conjunctions = {"そして", "しかし", "だから", "でも", "また", "ただし", "つまり", "なお", "さらに", "一方で"}
        interjections = {"あ", "お", "ねえ", "はい", "うん", "え", "おい", "もしもし", "わあ"}
        adnominals = {"この", "その", "あの", "どの", "ある", "あらゆる"}

        if word == "々":
            return self._normalize_grammar({
                "pos": "iteration_mark",
                "subpos": "kanji_repetition",
                "independent": False,
                "can_start": False,
                "can_end": False,
                "requires_prev": ["noun", "pronoun"],
                "requires_next": ["particle_case", "particle_binding", "copula", "none"],
                "forbid_prev": ["particle_case", "particle_binding", "auxiliary", "iteration_mark"],
                "forbid_next": ["iteration_mark"],
            })
        if word in particles_case:
            return self._normalize_grammar({
                "pos": "particle_case",
                "independent": False,
                "can_start": False,
                "can_end": False,
                "requires_prev": ["noun", "pronoun", "iteration_mark"],
                "requires_next": ["verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula", "noun"],
            })
        if word in particles_binding:
            return self._normalize_grammar({
                "pos": "particle_binding",
                "independent": False,
                "can_start": False,
                "can_end": False,
                "requires_prev": ["noun", "pronoun", "phrase", "iteration_mark"],
                "requires_next": ["verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula", "noun"],
            })
        if word in particles_conjunctive:
            return self._normalize_grammar({
                "pos": "particle_conjunctive",
                "independent": False,
                "can_start": False,
                "can_end": False,
                "requires_prev": ["verb_stem", "verb_suffix", "auxiliary", "adjective_stem"],
            })
        if word in particles_sentence:
            return self._normalize_grammar({
                "pos": "particle_sentence_final",
                "independent": False,
                "can_start": False,
                "can_end": True,
                "requires_prev": ["verb", "verb_stem", "verb_suffix", "adjective_i", "adjective_na", "adjective_stem", "copula", "auxiliary"],
            })
        if word in copula:
            return self._normalize_grammar({
                "pos": "copula",
                "independent": False,
                "can_start": False,
                "can_end": True,
                "requires_prev": ["noun", "adjective_na", "adjective_stem"],
            })
        if word in auxiliaries:
            return self._normalize_grammar({
                "pos": "auxiliary",
                "independent": False,
                "can_start": False,
                "can_end": True,
                "requires_prev": ["verb", "verb_stem", "verb_suffix", "adjective_i", "copula"],
                "requires_next": ["auxiliary", "particle", "particle_sentence_final", "none"],
            })
        if word in conjunctions:
            return self._normalize_grammar({
                "pos": "conjunction",
                "independent": True,
                "can_start": True,
                "can_end": False,
                "requires_next": ["noun", "pronoun", "verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem"],
            })
        if word in interjections:
            return self._normalize_grammar({"pos": "interjection", "independent": True, "can_start": True, "can_end": True})
        if word in adnominals:
            return self._normalize_grammar({"pos": "adnominal", "independent": True, "can_start": True, "can_end": False, "requires_next": ["noun"]})
        if word.endswith(("する", "した", "して", "される", "させる", "なる", "いく", "くる", "いる", "ある", "みる", "言う", "思う")):
            return self._normalize_grammar({"pos": "verb", "independent": True, "can_start": True, "can_end": True, "forms": ["predicate"], "roles": ["predicate"]})
        if word.endswith("い"):
            return self._normalize_grammar({"pos": "adjective_i", "independent": True, "can_start": True, "can_end": True, "roles": ["predicate", "modifier"]})
        if word == "な":
            return self._normalize_grammar({"pos": "adjective_na_helper", "independent": False, "can_start": False, "can_end": False, "requires_prev": ["adjective_stem"], "requires_next": ["noun"]})
        if word.endswith(("に", "く", "と")):
            return self._normalize_grammar({"pos": "adverb", "independent": True, "can_start": True, "can_end": False, "roles": ["adverbial"]})
        return self._normalize_grammar({"pos": "noun", "independent": True, "can_start": True, "can_end": True, "roles": ["subject", "object"]})

    def vector_of(self, word: str) -> Dict[str, float]:
        if word in self.lexicon:
            vec = self.lexicon[word].get("vector", {})
            return {axis: float(vec.get(axis, 0.0)) for axis in self.axes}
        return self.estimate_oov_vector(word)

    def estimate_oov_vector(self, word: str) -> Dict[str, float]:
        category = self.guess_category(word)
        if category in self.category_centroids:
            base = dict(self.category_centroids[category])
        elif DEFAULT_FALLBACK_CATEGORY in self.category_centroids:
            base = dict(self.category_centroids[DEFAULT_FALLBACK_CATEGORY])
        else:
            base = {axis: 0.0 for axis in self.axes}

        h = sum(ord(c) for c in word)
        for i, axis in enumerate(self.axes):
            jitter = (((h // (i + 1)) % 23) - 11) / 500.0
            base[axis] = max(-1.0, min(1.0, float(base.get(axis, 0.0)) + jitter))
        return base

    def blend_vector(self, words: List[str]) -> Dict[str, float]:
        if not words:
            return {axis: 0.0 for axis in self.axes}
        out = {axis: 0.0 for axis in self.axes}
        for w in words:
            vec = self.word_vectors.get(w)
            if vec is None:
                d = self.vector_of(w)
                for axis in self.axes:
                    out[axis] += d[axis]
            else:
                for i, axis in enumerate(self.axes):
                    out[axis] += vec[i]
        inv = 1.0 / max(len(words), 1)
        for axis in self.axes:
            out[axis] *= inv
        return out

    def apply_bias(self, vector: Dict[str, float], direction_bias: Dict[str, float]) -> Dict[str, float]:
        out = dict(vector)
        for axis, delta in direction_bias.items():
            norm_axis = self._normalize_axis_name(axis)
            if norm_axis in out:
                out[norm_axis] = max(-1.0, min(1.0, out[norm_axis] + float(delta)))
        return out

    def weighted_distance(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        total = 0.0
        for axis in self.axes:
            diff = a[axis] - b[axis]
            total += self.weights[axis] * diff * diff
        return math.sqrt(total)

    def _weighted_distance_tuple(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> float:
        total = 0.0
        for i, axis in enumerate(self.axes):
            diff = a[i] - b[i]
            total += self.weights[axis] * diff * diff
        return math.sqrt(total)

    def dependency_score(self, word: str, context_words: List[str]) -> float:
        grammar = self.grammar_of(word)
        mode = str(grammar.get("dependency_mode", DEFAULT_DEPENDENCY_MODE))
        if mode == "free":
            return 0.0

        pos = self.pos_of(word)
        prev_word = context_words[-1] if context_words else ""
        prev_pos = self.pos_of(prev_word) if prev_word else "none"

        requires_prev = set(grammar.get("requires_prev", []))
        forbid_prev = set(grammar.get("forbid_prev", []))
        penalty = 0.0

        if not context_words and not grammar.get("can_start", True):
            penalty += 1.5
        if requires_prev and prev_pos not in requires_prev and prev_word not in requires_prev and "phrase" not in requires_prev:
            penalty += 1.1
        if prev_pos in forbid_prev or prev_word in forbid_prev:
            penalty += 1.2
        if pos in _FUNCTION_POS and not prev_word:
            penalty += 1.4
        return penalty

    def _randomized_score(self, base_score: float) -> float:
        if self.randomness <= 0.0:
            return base_score
        scale = max(0.001, abs(base_score) + 0.25)
        noise = self.rng.uniform(-self.randomness, self.randomness) * scale
        return base_score + noise

    def _char_overlap_ratio(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        sa = self.word_char_sets.get(a, set(a))
        sb = self.word_char_sets.get(b, set(b))
        inter = len(sa & sb)
        union = len(sa | sb)
        if union <= 0:
            return 0.0
        return inter / union

    def _max_char_overlap(self, word: str, anchors: List[str]) -> float:
        if not anchors:
            return 0.0
        return max(self._char_overlap_ratio(word, anchor) for anchor in anchors)

    def _context_anchor_bonus(self, word: str, seed_words: List[str], active_words: List[str], source_word: str, current_words: List[str]) -> float:
        bonus = 0.0
        pos = self.pos_of(word)
        src_pos = self.pos_of(source_word)

        if word in seed_words:
            bonus += self.keep_input_bonus
        if word in current_words:
            bonus += self.keep_input_bonus * 0.75

        seed_overlap = self._max_char_overlap(word, seed_words)
        active_overlap = self._max_char_overlap(word, active_words)
        bonus += self.lexical_anchor_strength * seed_overlap
        bonus += (self.lexical_anchor_strength * 0.5) * active_overlap
        if pos == src_pos and pos != DEFAULT_FALLBACK_CATEGORY:
            bonus += 0.08
        return bonus

    def _preferred_pos_targets(self, source_word: str, context_words: List[str]) -> set[str]:
        src_pos = self.pos_of(source_word)
        prev_pos = self.pos_of(context_words[-1]) if context_words else "none"

        if src_pos in {"noun", "pronoun"}:
            return {"particle_case", "particle_binding", "noun", "pronoun", "verb", "adjective_i", "adjective_stem"}
        if src_pos in {"particle_case", "particle_binding"} or prev_pos in {"particle_case", "particle_binding"}:
            return {"verb", "verb_stem", "adjective_i", "adjective_stem", "noun", "pronoun", "copula"}
        if src_pos in {"verb", "verb_stem"}:
            return {"auxiliary", "particle_sentence_final", "particle_conjunctive", "noun", "adverb"}
        if src_pos in {"adjective_i", "adjective_stem"}:
            return {"noun", "copula", "particle_case", "adverb", "particle_sentence_final"}
        if src_pos == "particle_sentence_final":
            return {"interjection", "copula", "auxiliary", "verb", "adjective_i", "noun"}
        return {"noun", "pronoun", "verb", "verb_stem", "adjective_i", "adjective_stem", "adverb"}

    def _candidate_universe(self, allow_function_words: bool, source_word: str, context_words: List[str], seed_words: List[str], current_words: List[str]) -> List[str]:
        preferred_pos = self._preferred_pos_targets(source_word, context_words)
        bucket: List[str] = []
        seen = set()

        def add_word(word: str) -> None:
            if word not in seen:
                seen.add(word)
                bucket.append(word)

        for word in seed_words + current_words:
            if word in self.lexicon:
                add_word(word)

        for pos in preferred_pos:
            words = self.words_by_pos.get(pos, [])
            limit = min(len(words), self.per_pos_limit)
            for word in words[:limit]:
                add_word(word)

        if allow_function_words:
            for word in self.function_words[: self.max_function_pool]:
                add_word(word)

        for word in self.content_words[: self.max_content_pool]:
            add_word(word)

        return bucket

    def _vector_to_tuple(self, vector: Dict[str, float]) -> Tuple[float, ...]:
        return tuple(float(vector.get(axis, 0.0)) for axis in self.axes)

    def nearest_words(
        self,
        target_vector: Dict[str, float],
        top_k: int,
        exclude: Iterable[str],
        context_words: List[str] | None = None,
        allow_function_words: bool = True,
        seed_words: List[str] | None = None,
        active_words: List[str] | None = None,
        source_word: str = "",
        current_words: List[str] | None = None,
    ) -> List[Tuple[str, float]]:
        excluded = set(exclude)
        context_words = context_words or []
        seed_words = seed_words or []
        active_words = active_words or []
        current_words = current_words or []

        target_tuple = self._vector_to_tuple(target_vector)
        seed_vector = self.blend_vector(seed_words) if seed_words else {axis: 0.0 for axis in self.axes}
        current_vector = self.blend_vector(current_words) if current_words else {axis: 0.0 for axis in self.axes}
        seed_tuple = self._vector_to_tuple(seed_vector)
        current_tuple = self._vector_to_tuple(current_vector)

        candidate_words = self._candidate_universe(allow_function_words, source_word, context_words, seed_words, current_words)

        target_sum = sum(target_tuple)
        cheap: List[Tuple[str, float]] = []
        for word in candidate_words:
            if word in excluded:
                continue
            cheap_score = abs(self.word_axis_sums.get(word, 0.0) - target_sum)
            if word in seed_words:
                cheap_score -= 0.30
            if word in current_words:
                cheap_score -= 0.18
            cheap.append((word, cheap_score))

        cheap.sort(key=lambda x: x[1])
        shortlist_size = max(top_k * self.shortlist_factor, top_k + 16)
        shortlist = [word for word, _ in cheap[:shortlist_size]]

        scored: List[Tuple[str, float, float]] = []
        for word in shortlist:
            grammar = self.grammar_of(word)
            if not allow_function_words and grammar.get("function_word", False):
                continue

            candidate_tuple = self.word_vectors[word]
            distance = self._weighted_distance_tuple(target_tuple, candidate_tuple)
            dependency_penalty = self.dependency_score(word, context_words)
            seed_distance = self._weighted_distance_tuple(seed_tuple, candidate_tuple) if seed_words else 0.0
            current_distance = self._weighted_distance_tuple(current_tuple, candidate_tuple) if current_words else 0.0
            anchor_bonus = self._context_anchor_bonus(word, seed_words, active_words, source_word, current_words)

            base_score = (
                distance
                + dependency_penalty
                + (self.seed_anchor_strength * seed_distance)
                + (self.current_anchor_strength * current_distance)
                - anchor_bonus
            )
            noisy_score = self._randomized_score(base_score)
            scored.append((word, noisy_score, base_score))

        scored.sort(key=lambda x: x[1])
        candidate_pool = scored[: max(top_k, top_k * self.candidate_expansion)]
        if len(candidate_pool) <= top_k:
            return [(word, round(base, 6)) for word, _, base in candidate_pool]

        head = candidate_pool[:top_k]
        tail = candidate_pool[top_k:]
        mix_count = min(len(tail), max(1, top_k // 5))
        chosen_tail = self.rng.sample(tail, mix_count) if mix_count > 0 else []
        chosen = head[: max(1, top_k - mix_count)] + chosen_tail
        chosen.sort(key=lambda x: x[1])
        return [(word, round(base, 6)) for word, _, base in chosen[:top_k]]

    def diverge(
        self,
        input_words: List[str],
        depth: int = 2,
        top_k: int = 5,
        direction_bias: Dict[str, float] | None = None,
        seed_blend_ratio: float = 0.12,
        allow_function_words: bool = True,
    ) -> Dict[str, Any]:
        direction_bias = {self._normalize_axis_name(k): float(v) for k, v in (direction_bias or {}).items()}
        known_words = [w for w in input_words if self.has_word(w)]
        unknown_words = [w for w in input_words if not self.has_word(w)]

        current_words = list(dict.fromkeys(input_words))
        active_words = list(current_words)
        pool = list(active_words)
        seen = set(current_words)
        frontier = list(current_words)
        layers: List[Dict[str, Any]] = []
        seed_vector = self.blend_vector(current_words)

        for _ in range(depth):
            expanded_entries: List[Dict[str, Any]] = []
            next_frontier: List[str] = []

            frontier_iter = list(frontier)
            self.rng.shuffle(frontier_iter)
            for src in frontier_iter:
                src_vector = self.vector_of(src)
                blended = {
                    axis: ((1.0 - seed_blend_ratio) * src_vector[axis] + seed_blend_ratio * seed_vector[axis])
                    for axis in self.axes
                }
                target = self.apply_bias(blended, direction_bias)
                nearest = self.nearest_words(
                    target,
                    top_k=top_k,
                    exclude=seen,
                    context_words=active_words,
                    allow_function_words=allow_function_words,
                    seed_words=current_words,
                    active_words=active_words,
                    source_word=src,
                    current_words=current_words,
                )

                for word, score in nearest:
                    if word not in seen:
                        seen.add(word)
                    pool.append(word)
                    next_frontier.append(word)
                    expanded_entries.append({
                        "word": word,
                        "from": src,
                        "score": round(score, 6),
                        "category": self.category_of(word),
                        "pos": self.pos_of(word),
                    })

            next_frontier = list(dict.fromkeys(next_frontier))
            self.rng.shuffle(next_frontier)
            layers.append({"source_words": frontier_iter, "expanded": expanded_entries})
            frontier = next_frontier
            active_words.extend(next_frontier)
            active_words = list(dict.fromkeys(active_words))
            if not frontier:
                break

        return {
            "axes": list(self.axes),
            "input_words": input_words,
            "known_words": known_words,
            "unknown_words": unknown_words,
            "direction_bias": direction_bias,
            "weights": self.weights,
            "randomness": self.randomness,
            "candidate_expansion": self.candidate_expansion,
            "seed_anchor_strength": self.seed_anchor_strength,
            "current_anchor_strength": self.current_anchor_strength,
            "lexical_anchor_strength": self.lexical_anchor_strength,
            "layers": layers,
            "pool": list(dict.fromkeys(pool)),
        }


def parse_axis_pairs(items: List[str], axes: List[str] | Tuple[str, ...] | None = None) -> Dict[str, float]:
    axis_set = {DivergencePrimitive._normalize_axis_name(axis) for axis in (axes or [])}
    result: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        key = DivergencePrimitive._normalize_axis_name(k.strip())
        if axis_set and key not in axis_set:
            continue
        result[key] = float(v)
    return result


class DivergenceModel:
    def __init__(
        self,
        lexicon: Mapping[str, Any],
        weights: Dict[str, float] | None = None,
        random_seed: int | None = None,
        randomness: float = 0.04,
        candidate_expansion: int = 4,
        default_branch: int = 32,
        final_branch: int = 16,
        model_path: str | Path | None = None,
        unknown_word_resolver: UnknownWordResolver | None = None,
        persist_unknown_words: UnknownWordPersistor | None = None,
        auto_resolve_unknown_words: bool = True,
    ) -> None:
        self.default_branch = max(1, int(default_branch))
        self.final_branch = max(1, int(final_branch))
        self.model_path = Path(model_path) if model_path else None
        self.unknown_word_resolver = unknown_word_resolver
        self.persist_unknown_words = persist_unknown_words
        self.auto_resolve_unknown_words = bool(auto_resolve_unknown_words)

        self.state: Dict[str, Any] = {
            "weights": dict(weights or {}),
            "randomness": float(randomness),
            "candidate_expansion": int(candidate_expansion),
            "default_branch": self.default_branch,
            "final_branch": self.final_branch,
            "learning_meta": {
                "episodes_seen": 0,
                "last_avg_score": 0.5,
            },
            "version": 4,
        }

        if self.model_path and self.model_path.exists():
            self.load(self.model_path)

        self.primitive = DivergencePrimitive(
            lexicon=lexicon,
            weights=self.state.get("weights"),
            random_seed=random_seed,
            randomness=float(self.state.get("randomness", randomness)),
            candidate_expansion=int(self.state.get("candidate_expansion", candidate_expansion)),
        )

    @property
    def lexicon(self) -> Mapping[str, Any]:
        return self.primitive.lexicon

    @property
    def axes(self) -> Tuple[str, ...]:
        return self.primitive.axes

    def load(self, path: str | Path) -> None:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            self.state.update(obj)
            self.default_branch = int(self.state.get("default_branch", self.default_branch))
            self.final_branch = int(self.state.get("final_branch", self.final_branch))

    def save(self, path: str | Path | None = None) -> None:
        p = Path(path) if path else self.model_path
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)

    @staticmethod
    def flatten_unique(tokens: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for token in tokens:
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def ensure_words_registered(self, words: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        if not self.auto_resolve_unknown_words or self.unknown_word_resolver is None:
            return {}

        added = self.primitive.resolve_unknown_words(words, resolver=self.unknown_word_resolver)
        if added and self.persist_unknown_words is not None:
            try:
                self.persist_unknown_words(added)
                print(f"[UNKNOWN][PERSIST] added={len(added)}", flush=True)
            except Exception as e:
                print(f"[UNKNOWN][PERSIST][ERROR] error={e}", flush=True)
        return added

    def expand(
        self,
        token: str,
        context: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        allow_function_words: bool = True,
    ) -> List[str]:
        self.ensure_words_registered([token] + list(context or []))

        result = self.primitive.diverge(
            input_words=[token],
            depth=1,
            top_k=int(top_k or self.default_branch),
            allow_function_words=allow_function_words,
        )
        pool = [w for w in result.get("pool", []) if w != token]

        if context:
            context_set = set(context)
            prioritized = [w for w in pool if w not in context_set]
            if prioritized:
                pool = prioritized + [w for w in pool if w in context_set]

        return self.flatten_unique(pool[: int(top_k or self.default_branch)])

    def multi_expand(
        self,
        tokens: List[str],
        depth: int,
        top_k: Optional[int] = None,
        allow_function_words: bool = True,
    ) -> Dict[str, Any]:
        self.ensure_words_registered(tokens)

        branch = int(top_k or self.default_branch)
        current_tokens = self.flatten_unique(tokens)
        steps: List[Dict[str, Any]] = []

        for d in range(depth):
            expanded = []
            next_tokens: List[str] = []
            for token in current_tokens:
                children = self.expand(
                    token=token,
                    context=current_tokens,
                    top_k=branch,
                    allow_function_words=allow_function_words,
                )
                expanded.append({"token": token, "children": children})
                next_tokens.extend(children)

            next_tokens = self.flatten_unique(next_tokens)
            steps.append(
                {
                    "depth": d + 1,
                    "input_tokens": list(current_tokens),
                    "expanded": expanded,
                    "output_tokens": list(next_tokens),
                }
            )
            current_tokens = next_tokens
            if not current_tokens:
                break

        return {
            "input_tokens": list(tokens),
            "steps": steps,
            "output_tokens": list(current_tokens),
        }

    def final_expand(
        self,
        tokens: List[str],
        original_input: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[str]:
        self.ensure_words_registered(list(tokens) + list(original_input or []))

        branch = int(top_k or self.final_branch)
        tokens = self.flatten_unique(tokens)
        if not tokens:
            return (original_input or [])[:branch]

        expanded: List[str] = []
        for token in tokens:
            children = self.expand(
                token=token,
                context=tokens,
                top_k=max(1, branch // max(1, len(tokens))),
                allow_function_words=True,
            )
            expanded.extend(children)

        merged = self.flatten_unique(tokens + expanded)
        return self._surface_arrange(merged[:branch], original_input or [])

    def _surface_arrange(self, tokens: List[str], original_input: List[str]) -> List[str]:
        if not tokens:
            return ["…"]

        pos_map = {t: self.primitive.pos_of(t) for t in tokens}
        nouns = [t for t in tokens if pos_map[t] in {"noun", "pronoun"}]
        verbs = [t for t in tokens if pos_map[t] in {"verb", "verb_stem", "copula"}]
        adjs = [t for t in tokens if pos_map[t] in {"adjective_i", "adjective_na", "adjective_stem"}]
        adverbs = [t for t in tokens if pos_map[t] == "adverb"]
        particles = [t for t in tokens if pos_map[t] in {"particle", "particle_case", "particle_binding"}]
        finals = [t for t in tokens if pos_map[t] == "particle_sentence_final"]

        out: List[str] = []

        if nouns:
            out.append(nouns[0])

        if particles:
            preferred = None
            for p in ["は", "が", "を", "に", "で", "も", "と"]:
                if p in particles:
                    preferred = p
                    break
            if preferred:
                out.append(preferred)

        if adverbs:
            out.append(adverbs[0])

        if len(nouns) >= 2 and "を" in particles:
            if not out or out[-1] != "を":
                out.append("を")
            out.append(nouns[1])

        if adjs:
            out.append(adjs[0])

        if verbs:
            out.append(verbs[0])
        else:
            for token in original_input:
                if self.primitive.pos_of(token) in {"verb", "verb_stem", "copula"}:
                    out.append(token)
                    break

        if finals:
            out.append(finals[0])

        out = self.flatten_unique(out)
        if not out:
            out = self.flatten_unique(tokens)[:4]

        if not out or out[-1] not in {"。", "！", "？", "…"}:
            out.append("。")
        return out

    def _safe_score(self, ep: Dict[str, Any]) -> float:
        evaluation = ep.get("evaluation") or {}
        try:
            score = float(evaluation.get("score_total", 0.0))
        except Exception:
            score = 0.0
        return max(0.0, min(1.0, score))

    def _episode_axis_signal(self, ep: Dict[str, Any]) -> Dict[str, float]:
        signal = {axis: 0.0 for axis in self.axes}

        input_tokens = list(ep.get("input_tokens", []))
        initial_core = list(ep.get("initial_core", []))
        mid_converged = list(ep.get("mid_converged", []))
        final_expanded = [t for t in ep.get("final_expanded", []) if t != "。"]
        target_tokens = [t for t in ep.get("target_tokens", []) if str(t).strip() and str(t).strip() not in {"。", "、", "？", "！"}]

        if not final_expanded:
            final_expanded = list(mid_converged)

        input_vec = self.primitive.blend_vector(input_tokens) if input_tokens else {axis: 0.0 for axis in self.axes}
        core_vec = self.primitive.blend_vector(initial_core) if initial_core else input_vec
        mid_vec = self.primitive.blend_vector(mid_converged) if mid_converged else core_vec
        out_vec = self.primitive.blend_vector(final_expanded) if final_expanded else mid_vec

        if target_tokens:
            target_vec = self.primitive.blend_vector(target_tokens)
            for axis in self.axes:
                input_err = abs(input_vec[axis] - target_vec[axis])
                core_err = abs(core_vec[axis] - target_vec[axis])
                mid_err = abs(mid_vec[axis] - target_vec[axis])
                out_err = abs(out_vec[axis] - target_vec[axis])

                improvement = (input_err - out_err) + 0.5 * (core_err - mid_err)
                signal[axis] = improvement
        else:
            for axis in self.axes:
                drift_from_input = abs(out_vec[axis] - input_vec[axis])
                drift_from_core = abs(out_vec[axis] - core_vec[axis])
                drift_mid = abs(out_vec[axis] - mid_vec[axis])

                signal[axis] = (
                    0.30 * drift_from_input
                    + 0.30 * drift_from_core
                    + 0.20 * drift_mid
                    - 0.12 * abs(mid_vec[axis] - input_vec[axis])
                )

        return signal

    def update_from_episodes(self, episodes: List[Dict[str, Any]]) -> None:
        if not episodes:
            print("[DIVERGENCE][UPDATE] skip: no episodes", flush=True)
            return

        words_to_check: List[str] = []
        for ep in episodes:
            words_to_check.extend(ep.get("input_tokens", []))
            words_to_check.extend(ep.get("initial_core", []))
            words_to_check.extend(ep.get("mid_converged", []))
            words_to_check.extend(ep.get("final_expanded", []))
            words_to_check.extend(ep.get("target_tokens", []))
        self.ensure_words_registered(words_to_check)

        weights = dict(self.state.get("weights", {}))
        learning_meta = dict(self.state.get("learning_meta", {}))

        axis_delta_sum: Dict[str, float] = {axis: 0.0 for axis in self.axes}
        total_score_sum = 0.0
        episode_count = 0

        print(f"[DIVERGENCE][UPDATE] start episodes={len(episodes)}", flush=True)

        for idx, ep in enumerate(episodes, start=1):
            score_total = self._safe_score(ep)
            centered = (score_total * 2.0) - 1.0
            reward = centered if centered >= 0.0 else centered * 0.50

            axis_signal = self._episode_axis_signal(ep)
            signal_abs_sum = sum(abs(v) for v in axis_signal.values()) or 1.0

            total_score_sum += score_total
            episode_count += 1

            print(
                f"[DIVERGENCE][EP {idx}] episode_id={ep.get('episode_id')} "
                f"score_total={score_total:.6f} centered={centered:+.6f} reward={reward:+.6f}",
                flush=True,
            )

            top_signals = sorted(axis_signal.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            top_signal_axes = {axis for axis, _ in top_signals}

            for axis, raw_signal in top_signals:
                normalized_signal = raw_signal / signal_abs_sum
                axis_delta_sum[axis] += reward * normalized_signal
                print(
                    f"  [DIVERGENCE][SIGNAL] axis={axis} raw={raw_signal:+.6f} "
                    f"normalized={normalized_signal:+.6f} accum={axis_delta_sum[axis]:+.6f}",
                    flush=True,
                )

            for axis in self.axes:
                if axis in top_signal_axes:
                    continue
                normalized_signal = axis_signal[axis] / signal_abs_sum
                axis_delta_sum[axis] += reward * normalized_signal

        if episode_count <= 0:
            print("[DIVERGENCE][UPDATE] stop: episode_count=0", flush=True)
            return

        avg_score = total_score_sum / episode_count
        print(f"[DIVERGENCE][UPDATE] avg_score={avg_score:.6f}", flush=True)

        axis_change_logs: List[tuple[str, float, float, float]] = []

        for axis in self.axes:
            current = float(weights.get(axis, 1.0))
            raw_delta = axis_delta_sum[axis]
            delta = max(-0.05, min(0.05, raw_delta * 0.75))
            new_value = max(0.40, min(2.50, current + delta))
            rounded_new = round(new_value, 6)
            weights[axis] = rounded_new
            axis_change_logs.append((axis, current, rounded_new, rounded_new - current))

        axis_change_logs.sort(key=lambda x: abs(x[3]), reverse=True)
        print("[DIVERGENCE][UPDATE] axis changes:", flush=True)
        for axis, before, after, delta in axis_change_logs:
            print(
                f"  [DIVERGENCE][AXIS] {axis}: {before:.6f} -> {after:.6f} (delta={delta:+.6f})",
                flush=True,
            )

        self.state["weights"] = weights
        self.state["learning_meta"] = {
            "episodes_seen": int(learning_meta.get("episodes_seen", 0)) + episode_count,
            "last_avg_score": round(avg_score, 6),
        }

        self.primitive.weights = {
            axis: float(weights.get(axis, 1.0)) for axis in self.axes
        }

        print(
            f"[DIVERGENCE][UPDATE] done episodes_seen={self.state['learning_meta']['episodes_seen']} "
            f"last_avg_score={self.state['learning_meta']['last_avg_score']}",
            flush=True,
        )