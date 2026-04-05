from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.learning_dict import Learner, compute_reward, llm_eval, save_learning_log
from src.core.io.lsd_lexicon import load_lexicon_container, save_lexicon_container
from src.core.primitive.divergence import DivergencePrimitive
from src.llm.response_llm_api import generate_response


OOV_QUALITY_THRESHOLD = 0.40


def unique_preserve(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def clamp(x: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def load_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return p.read_text(encoding="utf-8")


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


def normalize_category(cat: str) -> str:
    cat = str(cat).strip()
    mapping = {
        "名詞": "noun",
        "名詞句": "noun",
        "代名詞": "pronoun",
        "動詞": "verb",
        "動詞語幹": "verb_stem",
        "形容詞": "adjective_i",
        "形容動詞": "adjective_na",
        "形容詞語幹": "adjective_stem",
        "副詞": "adverb",
        "助詞": "particle",
        "助動詞": "auxiliary",
        "接続詞": "conjunction",
        "感動詞": "interjection",
        "連体詞": "adnominal",
        "接頭辞": "prefix",
        "接尾辞": "suffix",
        "verb_auxiliary": "auxiliary",
        "verb-helper": "auxiliary",
        "adjective": "adjective_i",
        "adjective-stem": "adjective_stem",
        "verb-stem": "verb_stem",
        "auxiliary-adjective": "auxiliary",
    }
    return mapping.get(cat, cat if cat else "unknown")


def normalize_pos(pos: str, category: str) -> str:
    pos = str(pos).strip()
    mapping = {
        "noun": "noun",
        "pronoun": "pronoun",
        "verb": "verb",
        "verb_stem": "verb_stem",
        "verb-stem": "verb_stem",
        "adjective_i": "adjective_i",
        "adjective": "adjective_i",
        "adjective_na": "adjective_na",
        "adjective_stem": "adjective_stem",
        "adjective-stem": "adjective_stem",
        "adverb": "adverb",
        "particle": "particle_binding",
        "particle_case": "particle_case",
        "particle_binding": "particle_binding",
        "particle_conjunctive": "particle_conjunctive",
        "particle_sentence_final": "particle_sentence_final",
        "auxiliary": "auxiliary",
        "auxiliary-adjective": "auxiliary",
        "conjunction": "conjunction",
        "interjection": "interjection",
        "adnominal": "adnominal",
        "prefix": "prefix",
        "suffix": "suffix",
        "copula": "copula",
    }
    if pos in mapping:
        return mapping[pos]

    fallback = normalize_category(category)
    category_to_pos = {
        "noun": "noun",
        "pronoun": "pronoun",
        "verb": "verb",
        "verb_stem": "verb_stem",
        "adjective_i": "adjective_i",
        "adjective_na": "adjective_na",
        "adjective_stem": "adjective_stem",
        "adverb": "adverb",
        "auxiliary": "auxiliary",
        "conjunction": "conjunction",
        "interjection": "interjection",
        "adnominal": "adnominal",
        "prefix": "prefix",
        "suffix": "suffix",
    }
    return category_to_pos.get(fallback, "noun")


def is_atomic(word: str) -> bool:
    w = str(word).strip()
    if not w or " " in w:
        return False
    if len(w) == 1:
        return True

    blocked_exact = {"して", "いた", "ます", "でした", "である"}
    if w in blocked_exact:
        return False

    blocked_suffixes = [
        "で", "に", "が", "を", "へ", "の", "と", "も", "は",
        "から", "まで", "より", "て", "た", "だ",
        "ます", "でした", "いる", "して", "そう", "よう", "られる", "れる",
    ]
    for suffix in blocked_suffixes:
        if len(w) > len(suffix) and w.endswith(suffix):
            return False
    if len(w) >= 5:
        return False
    return True


def filter_oov_words(words: List[str]) -> List[str]:
    return unique_preserve([w for w in words if is_atomic(w)])


def refine_seed_with_llm(tokens: List[str]) -> List[str]:
    text = " ".join(tokens)
    prompt = f"""あなたは日本語の最小単位分解専用AIです。
以下の文を最小単位に分解してください。

条件:
- 半角スペース区切りのみ
- 説明禁止
- 句読点禁止
- 改行禁止
- 複合語は禁止
- 助詞・助動詞も分離する
例:
- 「公園で遊ぶ」→「公園 で 遊ぶ」
- 「楽しかった」→「楽し かった」
- 「走っている」→「走っ て いる」

入力:
{text}
"""
    try:
        print("[SEED RE-SPLIT]")
        response = generate_response(prompt)
        cleaned = response.strip().replace("\n", " ")
        refined = [t.strip() for t in cleaned.split(" ") if t.strip()]
        if refined:
            return refined
    except Exception as e:
        print(f"[SEED RE-SPLIT ERROR] {e}")
    return tokens


def load_atomic_prompt_template(path: str = "settings/atomic_sentence_prompt.txt") -> str:
    return load_text(path)


def normalize_generated_tokens(text: str) -> List[str]:
    cleaned = text.strip().replace("\r", " ").replace("\n", " ")
    tokens = [t.strip() for t in cleaned.split(" ") if t.strip()]
    return unique_preserve(tokens)


def generate_atomic_seed_words() -> List[str]:
    template = load_atomic_prompt_template()
    print("[ATOMIC GEN] requesting tokenized sentence from LLM...")
    response = generate_response(template)
    tokens = normalize_generated_tokens(response)
    if not tokens:
        raise RuntimeError("LLM returned empty token sequence")
    refined = refine_seed_with_llm(tokens)
    refined = normalize_generated_tokens(" ".join(refined))
    print("[SEED]")
    print(f" raw     = {tokens}")
    print(f" refined = {refined}")
    return refined


def get_normalized_lexicon_words(path: str) -> set[str]:
    try:
        lex = DivergencePrimitive.load_lexicon(path)
        return set(lex.keys())
    except Exception:
        return set()


def collect_unknown_words(words: List[str], lexicon_path: str) -> List[str]:
    known = get_normalized_lexicon_words(lexicon_path)
    return [w for w in unique_preserve(words) if w not in known]


def collect_record_oov_words(seed_words: List[str], record: Dict[str, Any], lexicon_path: str) -> List[str]:
    pool: List[str] = []
    pool.extend(seed_words)
    pool.extend(record.get("final_words", []))
    return collect_unknown_words(pool, lexicon_path)


def load_oov_prompt_template(path: str = "settings/oov_lexicon_prompt.txt") -> str:
    return load_text(path)


def build_oov_prompt(template: str, oov_words: List[str], axes: List[str]) -> str:
    return template.replace("{words}", "\n".join(oov_words)).replace("{axes}", ", ".join(axes))


def extract_json_object_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM response")
    return stripped[start:end + 1]


def normalize_entry(word: str, entry: Dict[str, Any], axes: List[str]) -> Dict[str, Any]:
    category = normalize_category(entry.get("category", "unknown"))
    raw_vector = entry.get("vector", {})
    if not isinstance(raw_vector, dict):
        raw_vector = {}

    vector: Dict[str, float] = {}
    for axis in axes:
        value = raw_vector.get(axis, 0.0)
        try:
            value = float(value)
        except Exception:
            value = 0.0
        vector[axis] = round(clamp(value, -1.0, 1.0), 6)

    pos = normalize_pos(entry.get("pos", category), category)

    grammar = {
        "pos": pos,
        "subpos": "",
        "independent": pos not in {
            "particle_case",
            "particle_binding",
            "particle_conjunctive",
            "particle_sentence_final",
            "auxiliary",
            "copula",
            "suffix",
            "prefix",
            "verb_suffix",
            "adjective_na_helper",
            "iteration_mark",
        },
        "can_start": pos not in {
            "particle_case",
            "particle_binding",
            "particle_conjunctive",
            "particle_sentence_final",
            "auxiliary",
            "copula",
            "suffix",
            "prefix",
            "verb_suffix",
            "adjective_na_helper",
            "iteration_mark",
        },
        "can_end": pos in {"noun", "pronoun", "verb", "verb_stem", "adjective_i", "adjective_stem", "copula", "auxiliary"},
        "content_word": pos in {"noun", "pronoun", "verb", "verb_stem", "adjective_i", "adjective_stem", "adverb", "conjunction", "interjection", "adnominal"},
        "function_word": pos in {
            "particle_case",
            "particle_binding",
            "particle_conjunctive",
            "particle_sentence_final",
            "auxiliary",
            "copula",
            "suffix",
            "prefix",
            "verb_suffix",
            "adjective_na_helper",
            "iteration_mark",
        },
        "requires_prev": [],
        "requires_next": [],
        "forbid_prev": [],
        "forbid_next": [],
        "roles": entry.get(
            "roles",
            ["subject", "object"] if pos in {"noun", "pronoun"}
            else ["predicate"] if pos in {"verb", "verb_stem"}
            else ["modifier"] if pos in {"adjective_i", "adjective_stem", "adverb"}
            else [],
        ),
        "forms": entry.get("forms", []),
        "connectability": float(entry.get("connectability", 0.5)),
    }

    return {"category": category, "vector": vector, "grammar": grammar}


def generate_lexicon_entries_for_oov(oov_words: List[str], axes: List[str]) -> Dict[str, Dict[str, Any]]:
    if not oov_words:
        return {}

    template = load_oov_prompt_template()
    prompt = build_oov_prompt(template, oov_words, axes)

    print(f"[LEXICON GEN] requesting entries for {len(oov_words)} words...")
    response = generate_response(prompt)
    json_text = extract_json_object_text(response)
    parsed = json.loads(json_text)

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")

    result: Dict[str, Dict[str, Any]] = {}
    for w in oov_words:
        if w in parsed and isinstance(parsed[w], dict):
            result[w] = normalize_entry(w, parsed[w], axes)
    return result


def merge_oov_entries_into_lexicon(path: str, new_entries: Dict[str, Dict[str, Any]]) -> int:
    if not new_entries:
        return 0

    target = Path(path)
    if target.exists():
        data = load_lexicon_container(target)
    else:
        data = {"meta": {}, "entries": {}, "lexicon": {}, "indexes": {}}

    entries = data.setdefault("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        data["entries"] = entries

    meta = data.setdefault("meta", {})
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta

    semantic_axes = list(meta.get("semantic_axes", []))
    if not semantic_axes and new_entries:
        first = next(iter(new_entries.values()))
        semantic_axes = list((first.get("vector") or {}).keys())
        if semantic_axes:
            meta["semantic_axes"] = semantic_axes

    added = 0
    for k, v in new_entries.items():
        if k in entries:
            continue

        pos = str(v.get("grammar", {}).get("pos", v.get("category", "noun")))
        hierarchy = ["content_words", "nouns", "generated", "oov"]

        if pos in {"verb", "verb_stem"}:
            hierarchy = ["content_words", "verbs", "stems", "oov"]
        elif pos == "adverb":
            hierarchy = ["content_words", "adverbs", "oov"]
        elif pos == "adjective_i":
            hierarchy = ["content_words", "adjectives", "i", "oov"]
        elif pos == "adjective_stem":
            hierarchy = ["content_words", "adjectives", "na", "stems", "oov"]
        elif pos == "particle_case":
            hierarchy = ["function_words", "particles", "case"]
        elif pos == "particle_binding":
            hierarchy = ["function_words", "particles", "binding"]
        elif pos == "particle_conjunctive":
            hierarchy = ["function_words", "particles", "conjunctive"]
        elif pos == "particle_sentence_final":
            hierarchy = ["function_words", "particles", "sentence_final"]
        elif pos == "auxiliary":
            hierarchy = ["function_words", "auxiliaries"]
        elif pos == "copula":
            hierarchy = ["function_words", "copulas"]
        elif pos == "suffix":
            hierarchy = ["function_words", "suffixes"]
        elif pos == "prefix":
            hierarchy = ["function_words", "prefixes"]

        entry = {
            "word": k,
            "category": v.get("category", "unknown"),
            "vector": dict(v.get("vector", {})),
            "grammar": dict(v.get("grammar", {})),
            "hierarchy": hierarchy,
        }
        entries[k] = entry
        added += 1

    save_lexicon_container(target, data)
    return added


def process_episode_with_llm(
    learner: Learner,
    episode_index: int,
    lexicon_path: str,
    seed_words: List[str],
) -> Dict[str, Any]:
    print(f"\n=== EPISODE {episode_index} ===")
    record = learner.run_episode(seed_words)
    record["seed_words"] = seed_words

    print("[EPISODE RESULT]")
    print(json.dumps(record, ensure_ascii=False, indent=2))

    eval_result = llm_eval(record)
    quality = eval_result["score"]
    reward = compute_reward(
        quality_score=quality,
        diversity_score=record["diversity"],
        convergence_score=record["convergence"],
        iteration_count=record["iterations"],
        relevance_score=eval_result["relevance"],
        coherence_score=eval_result["coherence"],
        focus_score=eval_result["focus"],
        noise_score=eval_result["noise"],
    )

    learner.policy.update(record["axis_weights"], reward)
    record["quality"] = quality
    record["reward"] = reward
    record["eval"] = eval_result

    save_learning_log(record, learner.policy.weights)

    print("[LEARN RESULT]")
    print(json.dumps({
        "quality": quality,
        "reward": reward,
        "eval": eval_result,
    }, ensure_ascii=False, indent=2))

    if quality < OOV_QUALITY_THRESHOLD:
        print(f"[OOV SKIP] quality={quality:.3f} < threshold={OOV_QUALITY_THRESHOLD:.3f}")
        return record

    raw_oov_words = collect_record_oov_words(seed_words, record, lexicon_path)
    print(f"[OOV RAW] {raw_oov_words}")

    oov_words = filter_oov_words(raw_oov_words)
    print(f"[OOV FILTERED] {oov_words}")

    if not oov_words:
        print("[OOV SKIP] no atomic OOV words")
        return record

    new_entries = generate_lexicon_entries_for_oov(oov_words, learner.axes)
    added = merge_oov_entries_into_lexicon(lexicon_path, new_entries)
    print(f"[OOV ADDED] {added}")

    return record


def run_single_loop(
    learner: Learner,
    episodes: int,
    loop_index: int,
    lexicon_path: str,
    regenerate_seed_every_episode: bool = True,
):
    records = []
    print(f"\n######## LOOP {loop_index} ########")

    shared_seed_words: List[str] = []
    if not regenerate_seed_every_episode:
        shared_seed_words = generate_atomic_seed_words()

    for ep in range(1, episodes + 1):
        if regenerate_seed_every_episode:
            seed_words = generate_atomic_seed_words()
        else:
            seed_words = list(shared_seed_words)

        record = process_episode_with_llm(
            learner=learner,
            episode_index=ep,
            lexicon_path=lexicon_path,
            seed_words=seed_words,
        )
        records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--lexicon", type=str, default=default_lexicon_path())
    parser.add_argument("--shared-seed", action="store_true")
    args = parser.parse_args()

    if args.loops <= 0:
        raise ValueError("--loops must be >= 1")
    if args.episodes <= 0:
        raise ValueError("--episodes must be >= 1")

    print(f"[LEXICON] loading from {args.lexicon}")
    learner = Learner(lexicon_path=args.lexicon)

    for i in range(1, args.loops + 1):
        run_single_loop(
            learner=learner,
            episodes=args.episodes,
            loop_index=i,
            lexicon_path=args.lexicon,
            regenerate_seed_every_episode=not args.shared_seed,
        )


if __name__ == "__main__":
    main()