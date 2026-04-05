from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


NOISE_TOKENS = {
    "",
    "ー",
    "化",
    "再",
    "られる",
    "せる",
    "れる",
}

PUNCTUATION_TOKENS = {"。", "、", "！", "？", "…", "「", "」", "（", "）", "(", ")"}
PARTICLE_TOKENS = {
    "は", "が", "を", "に", "へ", "と", "で", "も", "の", "から", "まで", "より",
    "ね", "よ", "か", "な", "ぞ", "さ", "わ",
}
PRONOUN_TOKENS = {"私", "わたし", "僕", "ぼく", "俺", "おれ", "あなた", "君", "きみ"}
COPULA_TOKENS = {"です", "だ"}
QUESTION_TOKENS = {"？", "?"}

DEFAULT_VERBAL_MODEL_PATH = "runtime/models/verbal_model.json"


def load_verbal_model(model_path: str = DEFAULT_VERBAL_MODEL_PATH) -> Dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        return {
            "weights": {
                "prefer_input_repair": 0.15,
                "prefer_final_tokens": 1.00,
                "prefer_desu": 0.70,
                "prefer_short": 0.55,
                "prefer_question_mark": 0.80,
            },
            "stats": {
                "updates": 0,
            },
        }

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "weights": {
                "prefer_input_repair": 0.15,
                "prefer_final_tokens": 1.00,
                "prefer_desu": 0.70,
                "prefer_short": 0.55,
                "prefer_question_mark": 0.80,
            },
            "stats": {
                "updates": 0,
            },
        }

    if not isinstance(data, dict):
        data = {}

    data.setdefault("weights", {})
    data.setdefault("stats", {})
    data["weights"].setdefault("prefer_input_repair", 0.15)
    data["weights"].setdefault("prefer_final_tokens", 1.00)
    data["weights"].setdefault("prefer_desu", 0.70)
    data["weights"].setdefault("prefer_short", 0.55)
    data["weights"].setdefault("prefer_question_mark", 0.80)
    data["stats"].setdefault("updates", 0)
    return data


def save_verbal_model(model: Dict[str, Any], model_path: str = DEFAULT_VERBAL_MODEL_PATH) -> None:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")


def verbalize(
    final_tokens: Sequence[str],
    original_input: Sequence[str] | None = None,
    model_path: str = DEFAULT_VERBAL_MODEL_PATH,
) -> str:
    model = load_verbal_model(model_path)
    candidates = generate_candidates(
        final_tokens=final_tokens,
        original_input=original_input or [],
        limit=6,
        model_path=model_path,
    )

    if not candidates:
        return "……"

    scored = score_candidates_locally(
        candidates=candidates,
        final_tokens=final_tokens,
        original_input=original_input or [],
        model=model,
    )
    scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return str(scored[0]["text"])


def generate_candidates(
    final_tokens: Sequence[str],
    original_input: Sequence[str],
    limit: int = 6,
    model_path: str = DEFAULT_VERBAL_MODEL_PATH,
) -> List[str]:
    model = load_verbal_model(model_path)
    weights = model.get("weights", {})

    final_clean = _clean_tokens(final_tokens)
    input_clean = _clean_tokens(original_input)

    candidates: List[str] = []

    # 1. final_tokens 優先
    final_only = _structure_from_final_priority(
        final_tokens=final_clean,
        original_input=[],
        prefer_desu=float(weights.get("prefer_desu", 0.70)),
        prefer_question_mark=float(weights.get("prefer_question_mark", 0.80)),
    )
    text = _cleanup_text(_join_tokens(final_only))
    if text:
        candidates.append(text)

    # 2. final優先 + input補助
    final_with_assist = _structure_from_final_priority(
        final_tokens=final_clean,
        original_input=input_clean,
        prefer_desu=float(weights.get("prefer_desu", 0.70)),
        prefer_question_mark=float(weights.get("prefer_question_mark", 0.80)),
    )
    text = _cleanup_text(_join_tokens(final_with_assist))
    if text:
        candidates.append(text)

    # 3. 状態文寄り
    state_text = _build_state_sentence(final_clean, input_clean)
    if state_text:
        candidates.append(state_text)

    # 4. 疑問文寄り
    question_text = _build_question_sentence(final_clean, input_clean)
    if question_text:
        candidates.append(question_text)

    # 5. 短文寄り
    compact_text = _build_compact_sentence(final_clean, input_clean)
    if compact_text:
        candidates.append(compact_text)

    # 6. 入力補修型（低優先）
    if float(weights.get("prefer_input_repair", 0.15)) > 0.0:
        repaired = _structure_from_input_assist(
            final_tokens=final_clean,
            original_input=input_clean,
        )
        text = _cleanup_text(_join_tokens(repaired))
        if text:
            candidates.append(text)

    seen = set()
    unique: List[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)

    return unique[:limit]


def score_candidates_locally(
    candidates: Sequence[str],
    final_tokens: Sequence[str],
    original_input: Sequence[str],
    model: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    model = model or load_verbal_model()
    weights = model.get("weights", {})

    final_clean = _clean_tokens(final_tokens)
    input_clean = _clean_tokens(original_input)

    results: List[Dict[str, Any]] = []

    for text in candidates:
        score = 0.0

        # final_tokens の反映率を強める
        final_hit = sum(1 for t in final_clean if t in text)
        input_hit = sum(1 for t in input_clean if t in text)

        score += float(weights.get("prefer_final_tokens", 1.00)) * final_hit
        score += float(weights.get("prefer_input_repair", 0.15)) * input_hit

        # 長すぎず短すぎず
        text_len = len(text)
        if 4 <= text_len <= 28:
            score += float(weights.get("prefer_short", 0.55))
        elif text_len <= 40:
            score += float(weights.get("prefer_short", 0.55)) * 0.5

        # です/だ を軽く優遇
        if "です" in text:
            score += float(weights.get("prefer_desu", 0.70))
        elif text.endswith("だ。"):
            score += float(weights.get("prefer_desu", 0.70)) * 0.5

        # 疑問符
        if any(t in final_clean for t in QUESTION_TOKENS):
            if text.endswith("？"):
                score += float(weights.get("prefer_question_mark", 0.80))

        # 句点
        if text.endswith(("。", "！", "？", "…")):
            score += 0.25

        # 連続重複の軽いペナルティ
        if "ですです" in text or "はは" in text or "もも" in text:
            score -= 0.5

        results.append({
            "text": text,
            "score": round(score, 6),
        })

    return results


def update_model_from_feedback(
    chosen_text: str,
    candidates: Sequence[str],
    final_tokens: Sequence[str],
    original_input: Sequence[str],
    model_path: str = DEFAULT_VERBAL_MODEL_PATH,
) -> Dict[str, Any]:
    model = load_verbal_model(model_path)
    weights = model.setdefault("weights", {})
    stats = model.setdefault("stats", {})

    local_scores = score_candidates_locally(
        candidates=candidates,
        final_tokens=final_tokens,
        original_input=original_input,
        model=model,
    )

    chosen = next((x for x in local_scores if x["text"] == chosen_text), None)
    best_local = max(local_scores, key=lambda x: float(x["score"]), default=None)

    if chosen is not None and best_local is not None:
        chosen_score = float(chosen["score"])
        best_score = float(best_local["score"])

        # 選ばれたものが local best とズレていたら重みを少し補正
        if chosen_score + 0.01 < best_score:
            weights["prefer_input_repair"] = max(
                0.0,
                float(weights.get("prefer_input_repair", 0.15)) - 0.01,
            )
            weights["prefer_final_tokens"] = min(
                2.0,
                float(weights.get("prefer_final_tokens", 1.00)) + 0.02,
            )
        else:
            weights["prefer_final_tokens"] = min(
                2.0,
                float(weights.get("prefer_final_tokens", 1.00)) + 0.005,
            )

        if "です" in chosen_text:
            weights["prefer_desu"] = min(
                2.0,
                float(weights.get("prefer_desu", 0.70)) + 0.01,
            )

        if len(chosen_text) <= 28:
            weights["prefer_short"] = min(
                2.0,
                float(weights.get("prefer_short", 0.55)) + 0.005,
            )

        if chosen_text.endswith("？"):
            weights["prefer_question_mark"] = min(
                2.0,
                float(weights.get("prefer_question_mark", 0.80)) + 0.01,
            )

    stats["updates"] = int(stats.get("updates", 0)) + 1
    save_verbal_model(model, model_path=model_path)
    return model


def _clean_tokens(tokens: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()

    for raw in tokens:
        token = str(raw).strip()
        if not token:
            continue
        if token in NOISE_TOKENS:
            continue
        if token not in seen:
            seen.add(token)
            result.append(token)

    return result


def _structure_from_final_priority(
    final_tokens: Sequence[str],
    original_input: Sequence[str],
    prefer_desu: float,
    prefer_question_mark: float,
) -> List[str]:
    core_tokens = _dedupe_keep_order(list(final_tokens))
    input_tokens = _dedupe_keep_order(list(original_input))

    if not core_tokens:
        core_tokens = input_tokens[:]

    has_question = any(t in QUESTION_TOKENS for t in core_tokens)
    tokens_wo_q = [
        t for t in core_tokens
        if t not in QUESTION_TOKENS and t not in {"。", "、", "！", "…"}
    ]

    subject = _pick_subject(tokens_wo_q)
    if subject is None:
        subject = _pick_subject(input_tokens)

    topic_time = _pick_first(tokens_wo_q, {"今日", "昨日", "明日", "今", "さっき", "今週", "先週", "来週"})
    if topic_time is None:
        topic_time = _pick_first(input_tokens, {"今日", "昨日", "明日", "今", "さっき", "今週", "先週", "来週"})

    state_word = _pick_state_word(tokens_wo_q)
    predicate = _pick_predicate(tokens_wo_q, preferred=state_word)

    result: List[str] = []

    if subject is not None:
        result.append(subject)
        if "は" in input_tokens or subject in PRONOUN_TOKENS:
            result.append("は")

    if topic_time is not None and topic_time not in result:
        result.append(topic_time)
        if "も" in input_tokens:
            result.append("も")

    if state_word is not None and state_word not in result:
        result.append(state_word)

    for token in tokens_wo_q:
        if token in result:
            continue
        if token in PARTICLE_TOKENS or token in PUNCTUATION_TOKENS:
            continue
        if token == predicate:
            continue
        if len(result) >= 5:
            break
        result.append(token)

    if predicate is not None and predicate not in result:
        result.append(predicate)

    while result and result[-1] in PARTICLE_TOKENS:
        result.pop()

    if has_question and prefer_question_mark > 0.0:
        result.append("？")
    elif result and result[-1] not in COPULA_TOKENS and _looks_nominal_sentence(result):
        if "です" in tokens_wo_q or prefer_desu >= 0.5:
            result.append("です")
        elif "だ" in tokens_wo_q:
            result.append("だ")

    return result[:8]


def _structure_from_input_assist(
    final_tokens: Sequence[str],
    original_input: Sequence[str],
) -> List[str]:
    tokens = _dedupe_keep_order(list(final_tokens) + list(original_input))
    if not tokens:
        return []

    result: List[str] = []
    for t in original_input:
        if t in {"。", "、"}:
            continue
        if t not in result:
            result.append(t)

    for t in final_tokens:
        if t not in result and t not in PARTICLE_TOKENS and t not in PUNCTUATION_TOKENS:
            result.append(t)
        if len(result) >= 8:
            break

    return result[:8]


def _build_state_sentence(final_tokens: Sequence[str], original_input: Sequence[str]) -> str:
    tokens = _dedupe_keep_order(list(final_tokens))
    input_tokens = _dedupe_keep_order(list(original_input))

    subject = _pick_subject(tokens) or _pick_subject(input_tokens)
    state = _pick_state_word(tokens) or _pick_state_word(input_tokens)
    topic_time = _pick_first(tokens, {"今日", "昨日", "明日"}) or _pick_first(input_tokens, {"今日", "昨日", "明日"})

    parts: List[str] = []
    if subject is not None:
        parts.extend([subject, "は"])
    if topic_time is not None:
        parts.append(topic_time)
        if "も" in input_tokens:
            parts.append("も")
    if state is not None:
        parts.append(state)
        parts.append("です")
    elif tokens:
        parts.append(tokens[0])
        parts.append("です")

    return _cleanup_text(_join_tokens(parts))


def _build_question_sentence(final_tokens: Sequence[str], original_input: Sequence[str]) -> str:
    tokens = _dedupe_keep_order(list(final_tokens))
    if not tokens:
        return ""

    if not any(t in QUESTION_TOKENS for t in tokens):
        return ""

    content = [t for t in tokens if t not in QUESTION_TOKENS and t not in PUNCTUATION_TOKENS]
    if not content:
        return "どういうこと？"

    text = _join_tokens(content[:4])
    text = _cleanup_text(text)
    if text.endswith("。"):
        text = text[:-1] + "？"
    elif not text.endswith("？"):
        text += "？"
    return text


def _build_compact_sentence(final_tokens: Sequence[str], original_input: Sequence[str]) -> str:
    tokens = _clean_tokens(final_tokens)
    if not tokens:
        tokens = _clean_tokens(original_input)

    compact = []
    for t in tokens:
        if t in PUNCTUATION_TOKENS:
            continue
        compact.append(t)
        if len(compact) >= 4:
            break

    if not compact:
        return ""

    if compact[-1] not in COPULA_TOKENS and _looks_nominal_sentence(compact):
        compact.append("です")

    return _cleanup_text(_join_tokens(compact))


def _pick_subject(tokens: Sequence[str]) -> str | None:
    for token in tokens:
        if token in PRONOUN_TOKENS:
            return token
    return None


def _pick_state_word(tokens: Sequence[str]) -> str | None:
    preferred = [
        "元気", "疲れ", "疲れた", "不安", "安心", "大丈夫", "平気",
        "悲しい", "嬉しい", "寂しい", "つらい", "眠い",
    ]
    for token in preferred:
        if token in tokens:
            return token
    return None


def _pick_predicate(tokens: Sequence[str], preferred: str | None = None) -> str | None:
    if "です" in tokens:
        return "です"
    if "だ" in tokens:
        return "だ"

    if preferred is not None and preferred in tokens:
        return "です"

    verb_like: List[str] = []
    for token in tokens:
        if token in PARTICLE_TOKENS or token in PUNCTUATION_TOKENS:
            continue
        if token.endswith(("る", "う", "く", "す", "つ", "ぬ", "ぶ", "む", "ぐ")):
            verb_like.append(token)
        elif token.endswith(("たい", "ない", "た", "だ")):
            verb_like.append(token)
        elif token.endswith(("しい", "い")) and len(token) >= 2:
            verb_like.append(token)

    if verb_like:
        return verb_like[0]

    return None


def _pick_first(tokens: Sequence[str], candidates: set[str]) -> str | None:
    for token in tokens:
        if token in candidates:
            return token
    return None


def _looks_nominal_sentence(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    last = tokens[-1]
    if last in COPULA_TOKENS:
        return False
    if last in PARTICLE_TOKENS:
        return False
    if last in PUNCTUATION_TOKENS:
        return False
    return True


def _join_tokens(tokens: Sequence[str]) -> str:
    if not tokens:
        return ""

    text = ""
    prev = ""

    for token in tokens:
        if token in {"。", "、", "！", "？", "…", "」", "）", ")"}:
            text += token
        elif token in {"「", "（", "("}:
            text += token
        elif prev in {"「", "（", "("}:
            text += token
        elif token in PARTICLE_TOKENS:
            text += token
        else:
            text += token
        prev = token

    return text


def _cleanup_text(text: str) -> str:
    replacements = [
        ("。。", "。"),
        ("、、", "、"),
        ("，，", "，"),
        ("！！", "！"),
        ("？？", "？"),
        ("はは", "は"),
        ("もも", "も"),
        ("ですです", "です"),
        ("だだ", "だ"),
    ]
    for src, dst in replacements:
        while src in text:
            text = text.replace(src, dst)

    text = text.strip()
    if not text:
        return ""

    if text.endswith("？"):
        return text

    if text[-1] not in "。！？…":
        text += "。"

    return text


def _dedupe_keep_order(tokens: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            result.append(token)
    return result