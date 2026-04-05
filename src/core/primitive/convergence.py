from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.core.primitive.divergence import DivergenceModel, DivergencePrimitive


class ConvergencePrimitive:
    def __init__(
        self,
        lexicon: Dict[str, Dict[str, Any]] | None = None,
        weights: Dict[str, float] | None = None,
        random_seed: int | None = None,
        randomness: float = 0.16,
        candidate_expansion: int = 3,
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

    def _context_profile(self, anchor_words: List[str]) -> Dict[str, Any]:
        pos_list = [self.pos_of(w) for w in anchor_words]
        content_count = sum(1 for w in anchor_words if self.grammar_of(w).get("content_word", False))
        function_count = sum(1 for w in anchor_words if self.grammar_of(w).get("function_word", False))
        return {
            "pos_list": pos_list,
            "content_count": content_count,
            "function_count": function_count,
            "has_predicate": any(p in {"verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula"} for p in pos_list),
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

        if pos in {"particle", "particle_case", "particle_binding", "particle_conjunctive", "particle_sentence_final", "auxiliary", "suffix", "verb_suffix", "adjective_na_helper", "iteration_mark"} and not anchor_words:
            penalty += 1.6

        if pos in {"particle", "particle_case", "particle_binding", "particle_conjunctive", "particle_sentence_final"} and prev_pos in {"particle", "particle_case", "particle_binding", "particle_conjunctive", "particle_sentence_final", "auxiliary", "verb_suffix", "adjective_na_helper"}:
            penalty += 1.4

        if pos in {"auxiliary", "verb_suffix"} and prev_pos not in {"verb", "verb_stem", "adjective_i", "copula", "auxiliary", "verb_suffix"}:
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
        if profile["last_pos"] in {"particle", "particle_case", "particle_binding", "particle_conjunctive"} and pos in {"verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "noun", "pronoun", "copula"}:
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

        base_final_score = avg_input_distance + grammar_penalty - role_bonus - recurrence_bonus - keep_bias + drop_bias
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
            "base_final_score": round(base_final_score, 6),
            "final_score": round(randomized_final_score, 6),
        }

    def build_sentence_candidates(self, anchor_words: List[str], scored_candidates: List[Dict[str, Any]], top_n: int) -> List[str]:
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
        randomness: float = 0.16,
        candidate_expansion: int = 3,
        model_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.state: Dict[str, Any] = {
            "weights": dict(weights or {}),
            "randomness": float(randomness),
            "candidate_expansion": int(candidate_expansion),
            "token_keep_bias": {},
            "token_drop_bias": {},
            "learning_meta": {
                "episodes_seen": 0,
                "last_avg_score": 50.0,
            },
            "version": 2,
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

    def initial_converge(
        self,
        input_tokens: List[str],
        target_min: int = 3,
        target_max: int = 5,
    ) -> List[str]:
        tokens = self.flatten_unique(input_tokens)
        if not tokens:
            return []

        token_keep_bias = self.state.get("token_keep_bias", {})
        token_drop_bias = self.state.get("token_drop_bias", {})

        scored = []
        for token in tokens:
            item = self.primitive.score_candidate(
                token,
                anchor_words=tokens,
                candidate_counts=None,
                token_keep_bias=token_keep_bias,
                token_drop_bias=token_drop_bias,
            )
            score = float(item["final_score"])

            pos = self.primitive.pos_of(token)
            if pos in {"noun", "pronoun", "verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula"}:
                score -= 0.20
            if pos in {"particle", "particle_case", "particle_binding", "particle_sentence_final", "auxiliary"}:
                score += 0.20

            scored.append((score, token))

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
    ) -> List[str]:
        candidate_tokens = self.flatten_unique(candidate_tokens)
        if not candidate_tokens:
            return self.flatten_unique(initial_core[:top_n])

        candidate_counts: Dict[str, int] = {}
        for token in candidate_tokens:
            candidate_counts[token] = candidate_counts.get(token, 0) + 1

        result = self.primitive.converge(
            input_words=original_input,
            candidate_words=candidate_tokens,
            top_n=top_n,
            candidate_counts=candidate_counts,
            anchor_words=self.flatten_unique(initial_core or original_input),
            token_keep_bias=self.state.get("token_keep_bias", {}),
            token_drop_bias=self.state.get("token_drop_bias", {}),
        )
        selected = list(result.get("selected_words", []))

        if not selected:
            selected = self.flatten_unique(initial_core[:top_n] or original_input[:top_n])

        has_predicate = any(
            self.primitive.pos_of(t) in {"verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula"}
            for t in selected
        )
        if not has_predicate:
            for token in candidate_tokens + original_input:
                if self.primitive.pos_of(token) in {"verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula"} and token not in selected:
                    selected.append(token)
                    break

        has_nominal = any(self.primitive.pos_of(t) in {"noun", "pronoun"} for t in selected)
        if not has_nominal:
            for token in candidate_tokens + original_input:
                if self.primitive.pos_of(token) in {"noun", "pronoun"} and token not in selected:
                    selected.insert(0, token)
                    break

        return self.flatten_unique(selected[:top_n])

    def _safe_score(self, ep: Dict[str, Any]) -> float:
        evaluation = ep.get("evaluation") or {}
        try:
            return float(evaluation.get("score_total", 50.0))
        except Exception:
            return 50.0

    def update_from_episodes(self, episodes: List[Dict[str, Any]]) -> None:
        if not episodes:
            print("[CONVERGENCE][UPDATE] skip: no episodes", flush=True)
            return

        token_keep_bias = dict(self.state.get("token_keep_bias", {}))
        token_drop_bias = dict(self.state.get("token_drop_bias", {}))
        learning_meta = dict(self.state.get("learning_meta", {}))

        total_score_sum = 0.0
        episode_count = 0

        print(f"[CONVERGENCE][UPDATE] start episodes={len(episodes)}", flush=True)

        for idx, ep in enumerate(episodes, start=1):
            total_score = self._safe_score(ep)
            total_score_sum += total_score
            episode_count += 1

            centered = (total_score - 50.0) / 50.0
            if centered >= 0:
                reward = centered
            else:
                reward = centered * 0.40

            input_tokens = set(ep.get("input_tokens", []))
            initial_core = set(ep.get("initial_core", []))
            mid_converged = set(ep.get("mid_converged", []))
            final_expanded = set(ep.get("final_expanded", []))
            response_tokens = {t for t in final_expanded if t != "。"}

            kept_tokens = set()
            kept_tokens |= initial_core
            kept_tokens |= mid_converged
            kept_tokens |= response_tokens

            candidate_like = set()
            for step in ep.get("divergence_steps", []):
                for expanded in step.get("expanded", []):
                    for child in expanded.get("children", []):
                        if child:
                            candidate_like.add(str(child))

            dropped_tokens = candidate_like - kept_tokens - input_tokens

            print(
                f"[CONVERGENCE][EP {idx}] episode_id={ep.get('episode_id')} "
                f"score_total={total_score:.6f} centered={centered:+.6f} reward={reward:+.6f} "
                f"kept={len(kept_tokens)} dropped={len(dropped_tokens)}",
                flush=True,
            )

            keep_logs: List[tuple[str, float, float, float]] = []
            for token in kept_tokens:
                pos = self.primitive.pos_of(token)
                pos_scale = 1.0
                if pos in {"verb", "verb_stem", "adjective_i", "adjective_na", "adjective_stem", "copula", "noun", "pronoun"}:
                    pos_scale = 1.15
                elif pos in {"particle", "particle_case", "particle_binding", "auxiliary", "particle_sentence_final"}:
                    pos_scale = 0.75

                current = float(token_keep_bias.get(token, 0.0))
                new_value = round(
                    max(-3.0, min(3.0, current + reward * 0.18 * pos_scale)),
                    6,
                )
                token_keep_bias[token] = new_value
                keep_logs.append((token, current, new_value, new_value - current))

            keep_logs.sort(key=lambda x: abs(x[3]), reverse=True)
            for token, before, after, delta in keep_logs[:12]:
                print(
                    f"  [CONVERGENCE][KEEP] {token}: {before:.6f} -> {after:.6f} (delta={delta:+.6f})",
                    flush=True,
                )

            drop_logs: List[tuple[str, float, float, float]] = []
            for token in dropped_tokens:
                pos = self.primitive.pos_of(token)
                pos_scale = 1.0
                if pos in {"particle", "particle_case", "particle_binding", "auxiliary", "particle_sentence_final"}:
                    pos_scale = 0.65

                current = float(token_drop_bias.get(token, 0.0))
                drop_delta = reward * 0.12 * pos_scale
                new_value = round(
                    max(-3.0, min(3.0, current + drop_delta)),
                    6,
                )
                token_drop_bias[token] = new_value
                drop_logs.append((token, current, new_value, new_value - current))

            drop_logs.sort(key=lambda x: abs(x[3]), reverse=True)
            for token, before, after, delta in drop_logs[:12]:
                print(
                    f"  [CONVERGENCE][DROP] {token}: {before:.6f} -> {after:.6f} (delta={delta:+.6f})",
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