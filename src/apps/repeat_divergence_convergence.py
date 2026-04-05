from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from src.core.primitive.divergence import DivergencePrimitive, parse_axis_pairs
from src.core.primitive.convergence import ConvergencePrimitive, extract_candidates_from_divergence


def default_lexicon_path() -> str:
    for name in (
        "libs/dict_30000_hierarchical.lsdx",
        "libs/dict_30000_hierarchical.lsd",
        "libs/dict_30000_hierarchical.json",
        "libs/dict.lsdx",
        "libs/dict.lsd",
        "libs/dict.json",
    ):
        p = Path(name)
        if p.exists():
            return str(p)
    return "libs/dict_30000_hierarchical.json"


def load_lexicon(path: Path) -> Dict[str, Dict[str, Any]]:
    return DivergencePrimitive.load_lexicon(path)


def unique_preserve(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def safe_entry(lexicon: Dict[str, Dict[str, Any]], word: str) -> Dict[str, Any]:
    return lexicon.get(word, {})


def get_grammar(lexicon: Dict[str, Dict[str, Any]], word: str) -> Dict[str, Any]:
    entry = safe_entry(lexicon, word)
    grammar = entry.get("grammar", {})
    return grammar if isinstance(grammar, dict) else {}


def get_pos(lexicon: Dict[str, Dict[str, Any]], word: str) -> str:
    grammar = get_grammar(lexicon, word)
    pos = grammar.get("pos")
    if isinstance(pos, str) and pos:
        return pos
    category = safe_entry(lexicon, word).get("category", "")
    if isinstance(category, str) and category:
        return category
    return "unknown"


def classify_role(lexicon: Dict[str, Dict[str, Any]], word: str) -> str:
    pos = get_pos(lexicon, word)
    roles = get_grammar(lexicon, word).get("roles", [])
    if not isinstance(roles, list):
        roles = []

    if pos == "iteration_mark":
        return "iteration"
    if pos in {"verb_stem", "verb"} or "predicate_core" in roles or "predicate" in roles:
        return "predicate_core"
    if pos in {"verb_suffix", "auxiliary", "copula"} or "predicate_suffix" in roles:
        return "predicate_suffix"
    if pos in {"adjective_stem", "adjective_i", "adjective_i_ending", "adverb"}:
        return "modifier"
    if pos in {"particle_case", "particle_binding", "particle_conjunctive", "particle_sentence_final", "adjective_na_helper"}:
        return "connector"
    if pos in {"noun", "pronoun"}:
        if "subject" in roles:
            return "subject_like"
        if "object" in roles:
            return "object_like"
        return "noun_like"
    return "other"


def compact_iteration_view(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "iteration": item["iteration"],
        "input_words": item["input_words"],
        "surface_input": item.get("surface_input", item["input_words"]),
        "converged_words": item["converged_words"],
        "surface_next": item.get("surface_next", item.get("next_words", [])),
    }


def build_candidate_pool(current_words: List[str], candidate_words: List[str], convergence_result: Dict[str, Any], keep_from_current: int = 4) -> List[str]:
    selected = [item["word"] for item in convergence_result.get("selected", [])]
    dropped = [item["word"] for item in convergence_result.get("dropped", [])]

    pool: List[str] = []
    pool.extend(current_words[:keep_from_current])
    pool.extend(selected[:8])
    pool.extend(candidate_words[:24])
    pool.extend(dropped[: max(2, len(selected) // 3)])
    return unique_preserve(pool)


def sample_by_role(
    lexicon: Dict[str, Dict[str, Any]],
    pool: List[str],
    rng: random.Random,
    max_subjects: int = 3,
    max_objects: int = 3,
    max_predicate_cores: int = 3,
    max_predicate_suffixes: int = 4,
    max_modifiers: int = 4,
    max_connectors: int = 4,
    max_others: int = 3,
) -> Dict[str, List[str]]:
    buckets = {
        "subject_candidates": [],
        "object_candidates": [],
        "predicate_core_candidates": [],
        "predicate_suffix_candidates": [],
        "modifier_candidates": [],
        "connector_candidates": [],
        "iteration_candidates": [],
        "other_candidates": [],
    }

    for w in pool:
        role = classify_role(lexicon, w)
        if role == "subject_like":
            buckets["subject_candidates"].append(w)
        elif role == "object_like":
            buckets["object_candidates"].append(w)
        elif role == "predicate_core":
            buckets["predicate_core_candidates"].append(w)
        elif role == "predicate_suffix":
            buckets["predicate_suffix_candidates"].append(w)
        elif role == "modifier":
            buckets["modifier_candidates"].append(w)
        elif role == "connector":
            buckets["connector_candidates"].append(w)
        elif role == "iteration":
            buckets["iteration_candidates"].append(w)
        else:
            buckets["other_candidates"].append(w)

    def sampled(items: List[str], k: int) -> List[str]:
        items = unique_preserve(items)
        if len(items) <= k:
            return items
        head = items[: max(1, k // 2)]
        tail = items[max(1, k // 2):]
        remaining = max(0, k - len(head))
        extra = rng.sample(tail, remaining) if len(tail) > remaining else tail
        return unique_preserve((head + extra)[:k])

    return {
        "subject_candidates": sampled(buckets["subject_candidates"], max_subjects),
        "object_candidates": sampled(buckets["object_candidates"], max_objects),
        "predicate_core_candidates": sampled(buckets["predicate_core_candidates"], max_predicate_cores),
        "predicate_suffix_candidates": sampled(buckets["predicate_suffix_candidates"], max_predicate_suffixes),
        "modifier_candidates": sampled(buckets["modifier_candidates"], max_modifiers),
        "connector_candidates": sampled(buckets["connector_candidates"], max_connectors),
        "iteration_candidates": sampled(buckets["iteration_candidates"], 1),
        "other_candidates": sampled(buckets["other_candidates"], max_others),
    }


def is_valid_followup(lexicon: Dict[str, Dict[str, Any]], prev_word: str, next_word: str) -> bool:
    if not prev_word:
        grammar = get_grammar(lexicon, next_word)
        return bool(grammar.get("can_start", True))

    prev_grammar = get_grammar(lexicon, prev_word)
    next_grammar = get_grammar(lexicon, next_word)

    prev_pos = get_pos(lexicon, prev_word)
    next_pos = get_pos(lexicon, next_word)

    forbid_prev = set(next_grammar.get("forbid_prev", []))
    if prev_word in forbid_prev or prev_pos in forbid_prev:
        return False

    requires_prev = set(next_grammar.get("requires_prev", []))
    if requires_prev and prev_word not in requires_prev and prev_pos not in requires_prev:
        return False

    forbid_next = set(prev_grammar.get("forbid_next", []))
    if next_word in forbid_next or next_pos in forbid_next:
        return False

    requires_next = set(prev_grammar.get("requires_next", []))
    if requires_next and next_word not in requires_next and next_pos not in requires_next and "none" not in requires_next:
        return False

    return True


def try_append_token(lexicon: Dict[str, Dict[str, Any]], seq: List[str], token: str) -> bool:
    if token in seq and get_pos(lexicon, token) not in {"particle_case", "particle_binding", "particle_conjunctive", "auxiliary"}:
        return False
    prev = seq[-1] if seq else ""
    if not is_valid_followup(lexicon, prev, token):
        return False
    if token == "々" and not seq:
        return False
    if token == "々":
        prev_pos = get_pos(lexicon, prev)
        if prev_pos not in {"noun", "pronoun"}:
            return False
    seq.append(token)
    return True


def maybe_attach_iteration_mark(lexicon: Dict[str, Dict[str, Any]], seq: List[str], rng: random.Random, role_buckets: Dict[str, List[str]]) -> None:
    if not seq:
        return
    if "々" not in role_buckets.get("iteration_candidates", []):
        return
    if rng.random() > 0.18:
        return
    prev = seq[-1]
    prev_pos = get_pos(lexicon, prev)
    if prev_pos in {"noun", "pronoun"} and is_valid_followup(lexicon, prev, "々"):
        seq.append("々")


def seed_priority_tokens(lexicon: Dict[str, Dict[str, Any]], seed_words: List[str]) -> Dict[str, List[str]]:
    subjects: List[str] = []
    objects: List[str] = []
    preds: List[str] = []
    suffixes: List[str] = []
    modifiers: List[str] = []
    connectors: List[str] = []
    others: List[str] = []

    for token in seed_words:
        role = classify_role(lexicon, token)
        if role == "subject_like":
            subjects.append(token)
        elif role == "object_like":
            objects.append(token)
        elif role == "predicate_core":
            preds.append(token)
        elif role == "predicate_suffix":
            suffixes.append(token)
        elif role == "modifier":
            modifiers.append(token)
        elif role == "connector":
            connectors.append(token)
        else:
            others.append(token)

    return {
        "subjects": unique_preserve(subjects),
        "objects": unique_preserve(objects),
        "preds": unique_preserve(preds),
        "suffixes": unique_preserve(suffixes),
        "modifiers": unique_preserve(modifiers),
        "connectors": unique_preserve(connectors),
        "others": unique_preserve(others),
    }


def compose_next_words(
    lexicon: Dict[str, Dict[str, Any]],
    current_words: List[str],
    seed_words: List[str],
    role_buckets: Dict[str, List[str]],
    rng: random.Random,
    target_size: int = 6,
    min_seed_tokens: int = 1,
) -> List[str]:
    next_words: List[str] = []
    seed_parts = seed_priority_tokens(lexicon, seed_words)

    predicate_cores = list(role_buckets["predicate_core_candidates"])
    predicate_suffixes = list(role_buckets["predicate_suffix_candidates"])
    subjects = list(role_buckets["subject_candidates"])
    objects = list(role_buckets["object_candidates"])
    modifiers = list(role_buckets["modifier_candidates"])
    connectors = list(role_buckets["connector_candidates"])
    others = list(role_buckets["other_candidates"])

    rng.shuffle(predicate_cores)
    rng.shuffle(predicate_suffixes)
    rng.shuffle(subjects)
    rng.shuffle(objects)
    rng.shuffle(modifiers)
    rng.shuffle(connectors)
    rng.shuffle(others)

    used_seed_count = 0

    def append_seed_first(candidates: List[str], limit: int | None = None) -> None:
        nonlocal used_seed_count
        added = 0
        for token in candidates:
            if len(next_words) >= target_size:
                break
            if limit is not None and added >= limit:
                break
            if try_append_token(lexicon, next_words, token):
                if token in seed_words:
                    used_seed_count += 1
                added += 1

    append_seed_first(seed_parts["subjects"], limit=1)
    append_seed_first(seed_parts["objects"], limit=1)
    maybe_attach_iteration_mark(lexicon, next_words, rng, role_buckets)
    append_seed_first(seed_parts["connectors"], limit=1)
    append_seed_first(seed_parts["modifiers"], limit=1)

    if used_seed_count < min_seed_tokens:
        carry_seed = [w for w in current_words if w in seed_words]
        append_seed_first(carry_seed, limit=min_seed_tokens - used_seed_count)

    predicate_added = False
    for token in seed_parts["preds"] + predicate_cores:
        if len(next_words) >= target_size:
            break
        if try_append_token(lexicon, next_words, token):
            if token in seed_words:
                used_seed_count += 1
            predicate_added = True
            break

    if predicate_added:
        for token in seed_parts["suffixes"] + predicate_suffixes:
            if len(next_words) >= target_size:
                break
            if try_append_token(lexicon, next_words, token):
                break

    for token in subjects + objects + modifiers + others:
        if len(next_words) >= target_size:
            break
        try_append_token(lexicon, next_words, token)

    if used_seed_count < min_seed_tokens:
        for token in seed_words:
            if len(next_words) >= target_size:
                break
            if token not in next_words and try_append_token(lexicon, next_words, token):
                break

    if not any(classify_role(lexicon, w) == "predicate_core" for w in next_words):
        for token in seed_parts["preds"] + predicate_cores:
            if token not in next_words and try_append_token(lexicon, next_words, token):
                break

    return unique_preserve(next_words[:target_size])


def detect_stagnation(history: List[List[str]], current: List[str], window: int = 2) -> bool:
    if len(history) < window:
        return False
    cur = tuple(current)
    return all(tuple(prev) == cur for prev in history[-window:])


def escape_stagnation(
    lexicon: Dict[str, Dict[str, Any]],
    candidate_pool: List[str],
    current_words: List[str],
    seed_words: List[str],
    rng: random.Random,
    target_size: int,
) -> List[str]:
    nouns = [w for w in candidate_pool if get_pos(lexicon, w) in {"noun", "pronoun"}]
    preds = [w for w in candidate_pool if classify_role(lexicon, w) == "predicate_core"]
    conns = [w for w in candidate_pool if classify_role(lexicon, w) == "connector"]

    rng.shuffle(nouns)
    rng.shuffle(preds)
    rng.shuffle(conns)

    escaped: List[str] = []
    for group in (seed_words[:1], conns[:1], nouns[:2], preds[:1], current_words[:1]):
        for token in group:
            if len(escaped) >= target_size:
                break
            try_append_token(lexicon, escaped, token)
    return unique_preserve(escaped[:target_size])


def surface_merge(tokens: List[str]) -> List[str]:
    merged: List[str] = []
    for token in tokens:
        if token == "々" and merged:
            merged[-1] = merged[-1] + "々"
        else:
            merged.append(token)
    return merged


def run_divergence_convergence(
    words,
    lexicon_path,
    axis_weights=None,
    iterations=2,
    depth=1,
    top_k=3,
    top_n=6,
    seed_blend_ratio=0.25,
    bias=None,
    summary_only=True,
    random_seed: int = 42,
    target_size: int = 6,
    verbose: bool = False,
):
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    lexicon = load_lexicon(Path(lexicon_path))
    log(f"[LEXICON] loaded entries={len(lexicon)}", flush=True)

    if axis_weights:
        tmp_div = DivergencePrimitive(lexicon=lexicon, weights=dict())
        div_weights = dict(zip(tmp_div.axes, axis_weights))
    else:
        div_weights = {}

    divergence_model = DivergencePrimitive(
        lexicon=lexicon,
        weights=div_weights,
        seed_anchor_strength=0.85,
        current_anchor_strength=0.55,
        lexical_anchor_strength=0.40,
        keep_input_bonus=0.24,
        randomness=0.06,
        candidate_expansion=2,
        max_content_pool=1000,
        max_function_pool=100,
        per_pos_limit=150,
        shortlist_factor=6,
    )

    shared_bias = parse_axis_pairs(bias or [], divergence_model.axes)
    convergence_model = ConvergencePrimitive(
        divergence_instance=divergence_model,
        random_seed=random_seed,
        randomness=0.10,
        candidate_expansion=2,
    )

    rng = random.Random(random_seed)
    initial_words = unique_preserve(list(words))
    current_words = list(initial_words)
    iterations_log = []
    history_next_words: List[List[str]] = []

    for idx in range(1, iterations + 1):
        divergence_result = divergence_model.diverge(
            input_words=current_words,
            depth=depth,
            top_k=top_k,
            direction_bias=shared_bias,
            seed_blend_ratio=seed_blend_ratio,
        )

        candidate_words, candidate_counts = extract_candidates_from_divergence(divergence_result)

        convergence_result = convergence_model.converge(
            input_words=current_words,
            candidate_words=candidate_words,
            top_n=top_n,
            bias=shared_bias,
            candidate_counts=candidate_counts,
            anchor_words=current_words,
        )

        candidate_pool = build_candidate_pool(
            current_words=current_words,
            candidate_words=candidate_words,
            convergence_result=convergence_result,
            keep_from_current=4,
        )

        role_buckets = sample_by_role(lexicon=lexicon, pool=candidate_pool, rng=rng)

        next_words = compose_next_words(
            lexicon=lexicon,
            current_words=current_words,
            seed_words=initial_words,
            role_buckets=role_buckets,
            rng=rng,
            target_size=target_size,
            min_seed_tokens=1,
        )

        stagnation = detect_stagnation(history_next_words, next_words, window=2)
        if stagnation:
            next_words = escape_stagnation(
                lexicon=lexicon,
                candidate_pool=candidate_pool,
                current_words=current_words,
                seed_words=initial_words,
                rng=rng,
                target_size=target_size,
            )

        converged_words = [item["word"] for item in convergence_result.get("selected", [])]
        iteration_record = {
            "iteration": idx,
            "input_words": list(current_words),
            "surface_input": surface_merge(current_words),
            "converged_words": converged_words,
            "surface_next": surface_merge(next_words),
        }
        if not summary_only:
            iteration_record["candidate_pool"] = candidate_pool
            iteration_record["role_buckets"] = role_buckets
            iteration_record["next_words"] = list(next_words)
            iteration_record["stagnation_escape"] = stagnation

        iterations_log.append(iteration_record)

        if not next_words:
            break

        current_words = list(next_words)
        history_next_words.append(list(next_words))

    return {
        "axes": list(divergence_model.axes),
        "initial_words": initial_words,
        "surface_initial": surface_merge(initial_words),
        "final_words": current_words,
        "surface_final": surface_merge(current_words),
        "iterations": iterations_log,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", type=Path, default=default_lexicon_path())
    parser.add_argument("--words", nargs="*")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--seed-blend-ratio", type=float, default=0.25)
    parser.add_argument("--bias", nargs="*", default=[])
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--target-size", type=int, default=6)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    no_args_mode = not args.words
    if no_args_mode:
        args.words = ["私"]
        args.summary_only = False

    result = run_divergence_convergence(
        words=args.words,
        lexicon_path=args.lexicon,
        axis_weights=None,
        iterations=args.iterations,
        depth=args.depth,
        top_k=args.top_k,
        top_n=args.top_n,
        seed_blend_ratio=args.seed_blend_ratio,
        bias=args.bias,
        summary_only=args.summary_only,
        random_seed=args.random_seed,
        target_size=args.target_size,
        verbose=args.verbose,
    )

    if args.summary_only:
        result["iterations"] = [compact_iteration_view(item) for item in result["iterations"]]

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
