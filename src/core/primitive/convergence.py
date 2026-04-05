from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from src.core.primitive.divergence import (
    DivergenceModel,
    DivergencePrimitive,
    UnknownWordPersistor,
    UnknownWordResolver,
)


class ConvergencePrimitive:
    def __init__(
        self,
        lexicon: Dict[str, Dict[str, Any]] | None = None,
        weights: Dict[str, float] | None = None,
        random_seed: int | None = None,
        randomness: float = 0.08,
        candidate_expansion: int = 2,
        divergence_instance: DivergencePrimitive | None = None,
    ) -> None:
        if divergence_instance is not None:
            self.div = divergence_instance
        else:
            if lexicon is None:
                raise ValueError("lexicon or divergence_instance is required")
            self.div = DivergencePrimitive(lexicon=lexicon, weights=weights)

        self.lexicon = self.div.lexicon
        self.axes = self.div.axes
        self.weights = self.div.weights
        self.randomness = max(0.0, float(randomness))
        self.candidate_expansion = max(1, int(candidate_expansion))
        self.rng = random.Random(random_seed)

    def has_word(self, word: str) -> bool:
        return self.div.has_word(word)

    def register_word(self, word: str, raw_entry: Mapping[str, Any]) -> Dict[str, Any]:
        entry = self.div.register_word(word, raw_entry)
        self.lexicon = self.div.lexicon
        self.axes = self.div.axes
        self.weights = self.div.weights
        return entry

    def register_words(self, entries: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
        added = self.div.register_words(entries)
        self.lexicon = self.div.lexicon
        self.axes = self.div.axes
        self.weights = self.div.weights
        return added

    def resolve_unknown_words(
        self,
        words: Iterable[str],
        resolver: UnknownWordResolver | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        added = self.div.resolve_unknown_words(words, resolver=resolver)
        self.lexicon = self.div.lexicon
        self.axes = self.div.axes
        self.weights = self.div.weights
        return added

    def vector_of(self, word: str) -> Dict[str, float]:
        return self.div.vector_of(word)

    def grammar_of(self, word: str) -> Dict[str, Any]:
        return self.div.grammar_of(word)

    def pos_of(self, word: str) -> str:
        return self.div.pos_of(word)

    def category_of(self, word: str) -> str:
        return self.div.category_of(word)

    def blend_vector(self, words: List[str]) -> Dict[str, float]:
        return self.div.blend_vector(words)

    def weighted_distance(self, a: Dict[str, float], b: Dict[str, float]) -> float:
        return self.div.weighted_distance(a, b)

    def avg_distance_to_inputs(self, word: str, input_words: List[str]) -> float:
        if not input_words:
            return 0.0
        vec = self.vector_of(word)
        distances = [self.weighted_distance(vec, self.vector_of(w)) for w in input_words]
        return sum(distances) / len(distances)

    def min_distance_to_inputs(self, word: str, input_words: List[str]) -> float:
        if not input_words:
            return 0.0
        vec = self.vector_of(word)
        return min(self.weighted_distance(vec, self.vector_of(w)) for w in input_words)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return str(text or "").strip().replace(" ", "").replace("　", "")

    @staticmethod
    def _char_jaccard(a: str, b: str) -> float:
        sa = set(ConvergencePrimitive._normalize_text(a))
        sb = set(ConvergencePrimitive._normalize_text(b))
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    def _context_profile(self, anchor_words: List[str]) -> Dict[str, Any]:
        pos_list = [self.pos_of(w) for w in anchor_words]
        content_count = sum(1 for w in anchor_words if self.grammar_of(w).get("content_word", False))
        function_count = sum(1 for w in anchor_words if self.grammar_of(w).get("function_word", False))
        return {
            "pos_list": pos_list,
            "content_count": content_count,
            "function_count": function_count,
            "has_predicate": any(
                p in {"verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula"}
                for p in pos_list
            ),
            "has_nominal": any(p in {"noun", "pronoun"} for p in pos_list),
            "last_word": anchor_words[-1] if anchor_words else "",
            "last_pos": pos_list[-1] if pos_list else "none",
        }

    def grammar_penalty(self, word: str, anchor_words: List[str]) -> float:
        grammar = self.grammar_of(word)
        pos = str(grammar.get("pos", "unknown"))
        profile = self._context_profile(anchor_words)
        penalty = 0.0

        if not anchor_words and not grammar.get("can_start", True):
            penalty += 1.5

        requires_prev = set(grammar.get("requires_prev", []))
        forbid_prev = set(grammar.get("forbid_prev", []))
        prev_word = profile["last_word"]
        prev_pos = profile["last_pos"]

        if requires_prev and prev_pos not in requires_prev and prev_word not in requires_prev and "phrase" not in requires_prev:
            penalty += 1.1
        if prev_pos in forbid_prev or prev_word in forbid_prev:
            penalty += 1.2

        if grammar.get("function_word", False) and profile["function_count"] >= profile["content_count"] + 1:
            penalty += 1.0

        if pos in {
            "particle",
            "particle_case",
            "particle_binding",
            "particle_conjunctive",
            "particle_sentence_final",
            "auxiliary",
            "suffix",
            "verb_suffix",
            "adjective_na_helper",
            "iteration_mark",
        } and not anchor_words:
            penalty += 1.6

        if pos in {
            "particle",
            "particle_case",
            "particle_binding",
            "particle_conjunctive",
            "particle_sentence_final",
        } and prev_pos in {
            "particle",
            "particle_case",
            "particle_binding",
            "particle_conjunctive",
            "particle_sentence_final",
            "auxiliary",
            "verb_suffix",
            "adjective_na_helper",
        }:
            penalty += 1.4

        if pos in {"auxiliary", "verb_suffix"} and prev_pos not in {
            "verb",
            "verb_stem",
            "adjective_i",
            "copula",
            "auxiliary",
            "verb_suffix",
        }:
            penalty += 1.2

        if pos in {"suffix", "iteration_mark"} and prev_pos not in {"noun", "pronoun"}:
            penalty += 1.2

        if pos == "adjective_na_helper" and prev_pos not in {"adjective_stem"}:
            penalty += 1.2

        return penalty

    def role_bonus(self, word: str, anchor_words: List[str]) -> float:
        grammar = self.grammar_of(word)
        pos = str(grammar.get("pos", "unknown"))
        profile = self._context_profile(anchor_words)
        bonus = 0.0

        if not profile["has_nominal"] and pos in {"noun", "pronoun"}:
            bonus += 0.45
        if not profile["has_predicate"] and pos in {"verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula"}:
            bonus += 0.55
        if profile["last_pos"] in {"noun", "pronoun"} and pos in {"particle", "particle_case", "particle_binding"}:
            bonus += 0.35
        if profile["last_pos"] in {"particle", "particle_case", "particle_binding", "particle_conjunctive"} and pos in {
            "verb",
            "verb_stem",
            "adjective_i",
            "adjective_na",
            "adjective_stem",
            "noun",
            "pronoun",
            "copula",
        }:
            bonus += 0.25
        if grammar.get("content_word", False):
            bonus += 0.08

        return bonus

    def _randomized_score(self, base_score: float) -> float:
        if self.randomness <= 0.0:
            return base_score
        scale = max(0.001, abs(base_score) + 0.2)
        noise = self.rng.uniform(-self.randomness, self.randomness) * scale
        return base_score + noise

    def score_candidate(
        self,
        word: str,
        anchor_words: List[str],
        candidate_counts: Dict[str, int] | None = None,
        token_keep_bias: Dict[str, float] | None = None,
        token_drop_bias: Dict[str, float] | None = None,
        desired_tokens: List[str] | None = None,
        desired_text: str = "",
    ) -> Dict[str, Any]:
        avg_input_distance = self.avg_distance_to_inputs(word, anchor_words)
        min_input_distance = self.min_distance_to_inputs(word, anchor_words)

        recurrence = 1
        if candidate_counts:
            recurrence = candidate_counts.get(word, 1)

        grammar_penalty = self.grammar_penalty(word, anchor_words)
        role_bonus = self.role_bonus(word, anchor_words)
        recurrence_bonus = min(recurrence, 4) * 0.05

        keep_bias = float((token_keep_bias or {}).get(word, 0.0))
        drop_bias = float((token_drop_bias or {}).get(word, 0.0))

        teacher_bonus = 0.0
        desired_set = set(desired_tokens or [])
        if word in desired_set:
            teacher_bonus += 1.40
        if desired_text and word and word in desired_text:
            teacher_bonus += 0.55

        base_final_score = (
            avg_input_distance
            + grammar_penalty
            - role_bonus
            - recurrence_bonus
            - keep_bias
            + drop_bias
            - teacher_bonus
        )
        randomized_final_score = self._randomized_score(base_final_score)
        return {
            "word": word,
            "category": self.category_of(word),
            "pos": self.pos_of(word),
            "avg_input_distance": round(avg_input_distance, 6),
            "min_input_distance": round(min_input_distance, 6),
            "grammar_penalty": round(grammar_penalty, 6),
            "role_bonus": round(role_bonus, 6),
            "recurrence": recurrence,
            "keep_bias": round(keep_bias, 6),
            "drop_bias": round(drop_bias, 6),
            "teacher_bonus": round(teacher_bonus, 6),
            "base_final_score": round(base_final_score, 6),
            "final_score": round(randomized_final_score, 6),
        }

    def build_sentence_candidates(
        self,
        anchor_words: List[str],
        scored_candidates: List[Dict[str, Any]],
        top_n: int,
    ) -> List[str]:
        current = list(anchor_words)
        selected_words: List[str] = []
        for item in scored_candidates:
            if len(selected_words) >= top_n:
                break
            if self.grammar_penalty(item["word"], current) >= 1.3:
                continue
            selected_words.append(item["word"])
            current = current + [item["word"]]
        return selected_words[:top_n]

    def converge(
        self,
        input_words: List[str],
        candidate_words: List[str],
        top_n: int = 8,
        bias: Dict[str, float] | None = None,
        candidate_counts: Dict[str, int] | None = None,
        anchor_words: List[str] | None = None,
        token_keep_bias: Dict[str, float] | None = None,
        token_drop_bias: Dict[str, float] | None = None,
        desired_tokens: List[str] | None = None,
        desired_text: str = "",
    ) -> Dict[str, Any]:
        _ = bias

        anchor_words = list(dict.fromkeys(anchor_words or input_words))
        anchor_set = set(anchor_words)

        filtered_candidates = [w for w in candidate_words if w not in anchor_set]
        unique_candidates = list(dict.fromkeys(filtered_candidates))

        if not unique_candidates:
            return {
                "axes": list(self.axes),
                "input_words": input_words,
                "anchor_words": anchor_words,
                "known_candidate_words": [],
                "unknown_candidate_words": [],
                "selected": [],
                "selected_words": [],
                "dropped": [],
            }

        scored = [
            self.score_candidate(
                word,
                anchor_words,
                candidate_counts=candidate_counts,
                token_keep_bias=token_keep_bias,
                token_drop_bias=token_drop_bias,
                desired_tokens=desired_tokens,
                desired_text=desired_text,
            )
            for word in unique_candidates
        ]
        scored.sort(key=lambda item: item["final_score"])

        sampled_pool = scored[: max(top_n, top_n * self.candidate_expansion)]
        if len(sampled_pool) > top_n:
            sampled_pool = self.rng.sample(sampled_pool, top_n)
            sampled_pool.sort(key=lambda item: item["final_score"])

        selected_words = self.build_sentence_candidates(anchor_words, sampled_pool, top_n=top_n)
        selected_set = set(selected_words)

        selected = [item for item in sampled_pool if item["word"] in selected_set][:top_n]
        dropped = [item for item in scored if item["word"] not in selected_set]

        return {
            "axes": list(self.axes),
            "input_words": input_words,
            "anchor_words": anchor_words,
            "known_candidate_words": [w for w in unique_candidates if self.has_word(w)],
            "unknown_candidate_words": [w for w in unique_candidates if not self.has_word(w)],
            "selected": selected,
            "selected_words": selected_words,
            "dropped": dropped,
            "randomness": self.randomness,
            "candidate_expansion": self.candidate_expansion,
        }


def extract_candidates_from_divergence(data: Dict[str, Any]) -> Tuple[List[str], Dict[str, int]]:
    pool = list(data.get("pool", []))
    counts: Dict[str, int] = {}
    for layer in data.get("layers", []):
        for item in layer.get("expanded", []):
            word = item.get("word")
            if word:
                counts[word] = counts.get(word, 0) + 1
    for word in pool:
        counts.setdefault(word, 1)
    return pool, counts


class ConvergenceModel:
    def __init__(
        self,
        lexicon: Mapping[str, Any] | None = None,
        divergence_model: DivergenceModel | None = None,
        weights: Dict[str, float] | None = None,
        random_seed: int | None = None,
        randomness: float = 0.08,
        candidate_expansion: int = 2,
        model_path: str | Path | None = None,
        unknown_word_resolver: UnknownWordResolver | None = None,
        persist_unknown_words: UnknownWordPersistor | None = None,
        auto_resolve_unknown_words: bool = True,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.unknown_word_resolver = unknown_word_resolver
        self.persist_unknown_words = persist_unknown_words
        self.auto_resolve_unknown_words = bool(auto_resolve_unknown_words)

        self.state: Dict[str, Any] = {
            "weights": dict(weights or {}),
            "randomness": float(randomness),
            "candidate_expansion": int(candidate_expansion),
            "token_keep_bias": {},
            "token_drop_bias": {},
            "learning_meta": {
                "episodes_seen": 0,
                "last_avg_score": 0.5,
            },
            "version": 5,
        }

        if self.model_path and self.model_path.exists():
            self.load(self.model_path)

        if divergence_model is not None:
            self.divergence_model = divergence_model
            divergence_instance = divergence_model.primitive
            lexicon = divergence_model.lexicon
        else:
            self.divergence_model = None
            divergence_instance = None

        if lexicon is None and divergence_instance is None:
            raise ValueError("lexicon or divergence_model is required")

        self.primitive = ConvergencePrimitive(
            lexicon=dict(lexicon) if lexicon is not None else None,
            weights=self.state.get("weights"),
            random_seed=random_seed,
            randomness=float(self.state.get("randomness", randomness)),
            candidate_expansion=int(self.state.get("candidate_expansion", candidate_expansion)),
            divergence_instance=divergence_instance,
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
        if self.divergence_model is not None:
            return self.divergence_model.ensure_words_registered(words)

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

    @staticmethod
    def _clean_target_tokens(tokens: List[str]) -> List[str]:
        return [str(t).strip() for t in tokens if str(t).strip() and str(t).strip() not in {"。", "、", "？", "！"}]

    @staticmethod
    def _normalize_text(text: str) -> str:
        return str(text or "").strip().replace(" ", "").replace("　", "")

    @staticmethod
    def _char_jaccard(a: str, b: str) -> float:
        sa = set(ConvergenceModel._normalize_text(a))
        sb = set(ConvergenceModel._normalize_text(b))
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    @staticmethod
    def _token_f1(predicted: List[str], target: List[str]) -> float:
        pred_set = set(str(x).strip() for x in predicted if str(x).strip())
        tgt_set = set(str(x).strip() for x in target if str(x).strip())
        if not pred_set and not tgt_set:
            return 1.0
        if not pred_set or not tgt_set:
            return 0.0
        tp = len(pred_set & tgt_set)
        if tp <= 0:
            return 0.0
        precision = tp / max(1, len(pred_set))
        recall = tp / max(1, len(tgt_set))
        if precision + recall <= 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def initial_converge(
        self,
        input_tokens: List[str],
        target_min: int = 3,
        target_max: int = 5,
    ) -> List[str]:
        self.ensure_words_registered(input_tokens)

        tokens = self.flatten_unique(input_tokens)
        if not tokens:
            return []

        scored = []
        for token in tokens:
            pos = self.primitive.pos_of(token)
            priority = 0.0
            if pos in {"noun", "pronoun", "verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula"}:
                priority -= 0.65
            elif pos in {"adverb", "interjection"}:
                priority -= 0.30
            elif pos in {"particle", "particle_case", "particle_binding", "particle_sentence_final", "auxiliary"}:
                priority += 0.40
            if token in {"？", "!", "！", "?", "。", "、"}:
                priority += 0.50
            scored.append((priority, token))

        scored.sort(key=lambda x: x[0])
        selected = [token for _, token in scored[:target_max]]

        if len(selected) < target_min:
            for token in tokens:
                if token not in selected:
                    selected.append(token)
                if len(selected) >= target_min:
                    break

        return self.flatten_unique(selected[:target_max])

    def converge(
        self,
        candidate_tokens: List[str],
        original_input: List[str],
        initial_core: List[str],
        top_n: int = 8,
        target_tokens: List[str] | None = None,
        target_text: str = "",
    ) -> List[str]:
        self.ensure_words_registered(candidate_tokens + original_input + initial_core + list(target_tokens or []))

        candidate_tokens = self.flatten_unique(candidate_tokens)
        if not candidate_tokens:
            return self.flatten_unique(initial_core[:top_n])

        candidate_counts: Dict[str, int] = {}
        for token in candidate_tokens:
            candidate_counts[token] = candidate_counts.get(token, 0) + 1

        desired_tokens = self._clean_target_tokens(target_tokens or [])

        result = self.primitive.converge(
            input_words=original_input,
            candidate_words=candidate_tokens,
            top_n=top_n,
            candidate_counts=candidate_counts,
            anchor_words=self.flatten_unique(initial_core or original_input),
            token_keep_bias=self.state.get("token_keep_bias", {}),
            token_drop_bias=self.state.get("token_drop_bias", {}),
            desired_tokens=desired_tokens,
            desired_text=target_text,
        )
        selected = list(result.get("selected_words", []))

        if desired_tokens:
            for token in desired_tokens:
                if token in candidate_tokens and token not in selected:
                    selected.insert(0, token)

        if not selected:
            selected = self.flatten_unique(initial_core[:top_n] or original_input[:top_n])

        has_predicate = any(
            self.primitive.pos_of(t) in {"verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula"}
            for t in selected
        )
        if not has_predicate:
            for token in candidate_tokens + original_input + desired_tokens:
                if self.primitive.pos_of(token) in {"verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula"} and token not in selected:
                    selected.append(token)
                    break

        has_nominal = any(self.primitive.pos_of(t) in {"noun", "pronoun"} for t in selected)
        if not has_nominal:
            for token in candidate_tokens + original_input + desired_tokens:
                if self.primitive.pos_of(token) in {"noun", "pronoun"} and token not in selected:
                    selected.insert(0, token)
                    break

        return self.flatten_unique(selected[:top_n])

    def _safe_score(self, ep: Dict[str, Any]) -> float:
        evaluation = ep.get("evaluation") or {}
        try:
            score = float(evaluation.get("score_total", 0.0))
        except Exception:
            score = 0.0
        return max(0.0, min(1.0, score))

    def _teacher_metrics(self, ep: Dict[str, Any]) -> Dict[str, float]:
        target_tokens = self._clean_target_tokens(list(ep.get("target_tokens", [])))
        target_text = str(ep.get("target_text", "")).strip()

        mid_converged = self.flatten_unique([str(t) for t in ep.get("mid_converged", []) if str(t)])
        final_expanded = self.flatten_unique([str(t) for t in ep.get("final_expanded", []) if str(t) and str(t) != "。"])
        response_text = str(ep.get("response_text", "")).strip()

        token_f1 = 0.0
        if target_tokens:
            token_f1 = max(
                self._token_f1(mid_converged, target_tokens),
                self._token_f1(final_expanded, target_tokens),
            )

        text_sim = self._char_jaccard(response_text, target_text) if target_text else 0.0

        return {
            "token_f1": round(token_f1, 6),
            "text_sim": round(text_sim, 6),
        }

    def update_from_episodes(self, episodes: List[Dict[str, Any]]) -> None:
        if not episodes:
            print("[CONVERGENCE][UPDATE] skip: no episodes", flush=True)
            return

        words_to_check: List[str] = []
        for ep in episodes:
            words_to_check.extend(ep.get("input_tokens", []))
            words_to_check.extend(ep.get("initial_core", []))
            words_to_check.extend(ep.get("mid_converged", []))
            words_to_check.extend(ep.get("final_expanded", []))
            words_to_check.extend(ep.get("target_tokens", []))
            for step in ep.get("divergence_steps", []):
                for expanded in step.get("expanded", []):
                    for child in expanded.get("children", []):
                        if child:
                            words_to_check.append(str(child))
        self.ensure_words_registered(words_to_check)

        token_keep_bias = dict(self.state.get("token_keep_bias", {}))
        token_drop_bias = dict(self.state.get("token_drop_bias", {}))
        learning_meta = dict(self.state.get("learning_meta", {}))

        total_score_sum = 0.0
        episode_count = 0

        print(f"[CONVERGENCE][UPDATE] start episodes={len(episodes)}", flush=True)

        for idx, ep in enumerate(episodes, start=1):
            score_total = self._safe_score(ep)
            teacher = self._teacher_metrics(ep)
            token_f1 = float(teacher["token_f1"])
            text_sim = float(teacher["text_sim"])

            combined = (0.65 * token_f1) + (0.20 * text_sim) + (0.15 * score_total)
            centered = (combined * 2.0) - 1.0
            reward = centered if centered >= 0.0 else centered * 0.45

            total_score_sum += combined
            episode_count += 1

            input_tokens = set(self.flatten_unique(list(ep.get("input_tokens", []))))
            target_tokens = set(self._clean_target_tokens(list(ep.get("target_tokens", []))))
            initial_core = set(self.flatten_unique(list(ep.get("initial_core", []))))
            mid_converged = set(self.flatten_unique(list(ep.get("mid_converged", []))))
            final_expanded = set(self.flatten_unique([t for t in ep.get("final_expanded", []) if str(t) != "。"]))

            kept_tokens = set()
            kept_tokens |= initial_core
            kept_tokens |= mid_converged
            kept_tokens |= final_expanded

            candidate_like = set()
            for step in ep.get("divergence_steps", []):
                for expanded in step.get("expanded", []):
                    for child in expanded.get("children", []):
                        if child:
                            candidate_like.add(str(child))

            desired_kept = {t for t in kept_tokens if t in target_tokens}
            noisy_kept = {t for t in kept_tokens if t not in target_tokens and t not in input_tokens}
            desired_dropped = {t for t in target_tokens if t not in kept_tokens}
            dropped_noise = {t for t in candidate_like if t not in kept_tokens and t not in target_tokens}

            print(
                f"[CONVERGENCE][EP {idx}] episode_id={ep.get('episode_id')} "
                f"score_total={score_total:.6f} token_f1={token_f1:.6f} text_sim={text_sim:.6f} "
                f"combined={combined:.6f} reward={reward:+.6f} "
                f"desired_kept={len(desired_kept)} noisy_kept={len(noisy_kept)} desired_dropped={len(desired_dropped)}",
                flush=True,
            )

            keep_logs: List[tuple[str, float, float, float]] = []

            for token in desired_kept:
                current = float(token_keep_bias.get(token, 0.0))
                delta = 0.26 * max(0.0, reward) + 0.16
                new_value = round(max(-4.0, min(4.0, current + delta)), 6)
                token_keep_bias[token] = new_value
                keep_logs.append((token, current, new_value, new_value - current))

            for token in desired_dropped:
                current_keep = float(token_keep_bias.get(token, 0.0))
                new_keep = round(max(-4.0, min(4.0, current_keep + 0.22)), 6)
                token_keep_bias[token] = new_keep
                keep_logs.append((token, current_keep, new_keep, new_keep - current_keep))

                current_drop = float(token_drop_bias.get(token, 0.0))
                new_drop = round(max(-4.0, min(4.0, current_drop - 0.18)), 6)
                token_drop_bias[token] = new_drop

            for token in noisy_kept:
                current = float(token_drop_bias.get(token, 0.0))
                delta = (0.22 * max(0.0, -reward)) + 0.12
                new_value = round(max(-4.0, min(4.0, current + delta)), 6)
                token_drop_bias[token] = new_value

            for token in dropped_noise:
                current = float(token_drop_bias.get(token, 0.0))
                new_value = round(max(-4.0, min(4.0, current + 0.04)), 6)
                token_drop_bias[token] = new_value

            keep_logs.sort(key=lambda x: abs(x[3]), reverse=True)
            for token, before, after, delta in keep_logs[:12]:
                print(
                    f"  [CONVERGENCE][KEEP] {token}: {before:.6f} -> {after:.6f} (delta={delta:+.6f})",
                    flush=True,
                )

        if episode_count <= 0:
            print("[CONVERGENCE][UPDATE] stop: episode_count=0", flush=True)
            return

        avg_score = total_score_sum / episode_count

        self.state["token_keep_bias"] = token_keep_bias
        self.state["token_drop_bias"] = token_drop_bias
        self.state["learning_meta"] = {
            "episodes_seen": int(learning_meta.get("episodes_seen", 0)) + episode_count,
            "last_avg_score": round(avg_score, 6),
        }

        top_keep = sorted(token_keep_bias.items(), key=lambda x: abs(float(x[1])), reverse=True)[:20]
        top_drop = sorted(token_drop_bias.items(), key=lambda x: abs(float(x[1])), reverse=True)[:20]

        print(f"[CONVERGENCE][UPDATE] avg_score={avg_score:.6f}", flush=True)
        print("[CONVERGENCE][UPDATE] top keep_bias:", flush=True)
        for token, value in top_keep:
            print(f"  [CONVERGENCE][KEEP_TOP] {token}: {float(value):+.6f}", flush=True)

        print("[CONVERGENCE][UPDATE] top drop_bias:", flush=True)
        for token, value in top_drop:
            print(f"  [CONVERGENCE][DROP_TOP] {token}: {float(value):+.6f}", flush=True)

        print(
            f"[CONVERGENCE][UPDATE] done episodes_seen={self.state['learning_meta']['episodes_seen']} "
            f"last_avg_score={self.state['learning_meta']['last_avg_score']}",
            flush=True,
        )