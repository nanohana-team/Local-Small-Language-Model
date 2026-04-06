from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "libs" / "dict.json"

SEMANTIC_AXES = [
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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, round(float(value), 3)))


DEFAULT_VECTOR = {
    "valence": 0.5,
    "arousal": 0.3,
    "abstractness": 0.3,
    "sociality": 0.3,
    "temporality": 0.2,
    "agency": 0.2,
    "causality": 0.2,
    "certainty": 0.5,
    "deixis": 0.1,
    "discourse_force": 0.3,
}


CATEGORY_TO_HIERARCHY = {
    "special_mark": ["function_words", "special_marks"],
    "particle": ["function_words", "particles"],
    "inflection": ["function_words", "inflections"],
    "copula": ["function_words", "copulas"],
    "auxiliary": ["function_words", "auxiliaries"],
    "pronoun": ["content_words", "pronouns"],
    "noun": ["content_words", "nouns"],
    "question_word": ["content_words", "question_words"],
    "adverb": ["content_words", "adverbs"],
    "interjection": ["content_words", "interjections"],
    "adjective_na": ["content_words", "adjectives", "na"],
    "adjective_i": ["content_words", "adjectives", "i"],
    "verb_stem": ["content_words", "verbs", "stems"],
}


def make_vector(**updates: float) -> Dict[str, float]:
    data = dict(DEFAULT_VECTOR)
    data.update({k: clamp01(v) for k, v in updates.items()})
    return data



def make_entry(
    *,
    word: str,
    category: str,
    pos: str,
    sub_pos: str = "",
    entry_type: str,
    hierarchy: Sequence[str] | None = None,
    can_start: bool = False,
    can_end: bool = False,
    independent: bool = True,
    content_word: bool = True,
    function_word: bool = False,
    requires_prev: Sequence[str] | None = None,
    requires_next: Sequence[str] | None = None,
    forbid_prev: Sequence[str] | None = None,
    forbid_next: Sequence[str] | None = None,
    vector: Dict[str, float] | None = None,
    slots: List[Dict[str, Any]] | None = None,
    relations: List[Dict[str, Any]] | None = None,
    frequency: float = 0.5,
    style_tags: Sequence[str] | None = None,
    lemma: str = "",
    stem_id: str = "",
    conj_class: str = "",
    effect: Dict[str, Any] | None = None,
    aliases: Sequence[str] | None = None,
    surface_forms: List[Dict[str, Any]] | None = None,
    meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "word": word,
        "category": category,
        "hierarchy": list(hierarchy or CATEGORY_TO_HIERARCHY.get(category, ["content_words", "generated"])),
        "vector": vector or make_vector(),
        "grammar": {
            "pos": pos,
            "sub_pos": sub_pos,
            "can_start": bool(can_start),
            "can_end": bool(can_end),
            "independent": bool(independent),
            "content_word": bool(content_word),
            "function_word": bool(function_word),
            "requires_prev": list(requires_prev or []),
            "requires_next": list(requires_next or []),
            "forbid_prev": list(forbid_prev or []),
            "forbid_next": list(forbid_next or []),
        },
        "slots": list(slots or []),
        "relations": list(relations or []),
        "frequency": clamp01(frequency),
        "style_tags": list(style_tags or ["daily"]),
        "meta": dict(meta or {}),
        "type": entry_type,
        "lemma": lemma or word,
        "stem_id": stem_id,
        "conj_class": conj_class,
        "effect": dict(effect or {}),
        "aliases": list(aliases or []),
        "surface_forms": list(surface_forms or []),
    }



def slot(name: str, required: bool = False, allowed_pos: Sequence[str] | None = None, semantic_hint: Sequence[str] | None = None) -> Dict[str, Any]:
    return {
        "name": name,
        "required": required,
        "allowed_pos": list(allowed_pos or []),
        "semantic_hint": list(semantic_hint or []),
        "note": "",
    }



def rel(relation: str, target: str, weight: float = 0.25, bidirectional: bool = False) -> Dict[str, Any]:
    return {
        "relation": relation,
        "target": target,
        "weight": clamp01(weight),
        "bidirectional": bidirectional,
        "note": "",
    }


entries: Dict[str, Dict[str, Any]] = {}



def add(entry: Dict[str, Any]) -> None:
    entries[entry["word"]] = entry


# punctuation / marks
for mark, sub_pos, can_start, can_end, certainty, force in [
    ("。", "period", False, True, 0.92, 0.55),
    ("、", "comma", False, False, 0.50, 0.35),
    ("？", "question", False, True, 0.50, 0.82),
    ("！", "exclaim", False, True, 0.62, 0.92),
    ("…", "ellipsis", False, True, 0.42, 0.58),
]:
    add(
        make_entry(
            word=mark,
            category="special_mark",
            pos="special_mark",
            sub_pos=sub_pos,
            entry_type="special_mark",
            can_start=can_start,
            can_end=can_end,
            independent=False,
            content_word=False,
            function_word=True,
            vector=make_vector(certainty=certainty, discourse_force=force, abstractness=0.1),
            frequency=0.95,
            style_tags=["daily"],
        )
    )


# particles
particle_specs = [
    ("は", "topic", {"mark_role": "topic"}),
    ("が", "actor", {"mark_role": "actor"}),
    ("を", "target", {"mark_role": "target"}),
    ("に", "goal", {"mark_role": "location_or_time"}),
    ("で", "location", {"mark_role": "location"}),
    ("と", "with", {"mark_role": "companion_or_quote"}),
    ("へ", "direction", {"mark_role": "direction"}),
    ("から", "source", {"mark_role": "source"}),
    ("まで", "limit", {"mark_role": "limit"}),
    ("も", "also", {"mark_role": "also"}),
    ("ね", "ending", {"sentence_tone": "shared"}),
    ("よ", "ending", {"sentence_tone": "assertive"}),
    ("か", "ending", {"sentence_tone": "question"}),
    ("の", "link", {"mark_role": "modifier"}),
    ("って", "topic_quote", {"mark_role": "topic_or_quote"}),
]
for word, sub_pos, effect in particle_specs:
    add(
        make_entry(
            word=word,
            category="particle",
            pos="particle",
            sub_pos=sub_pos,
            entry_type="particle",
            can_start=False,
            can_end=sub_pos == "ending",
            independent=False,
            content_word=False,
            function_word=True,
            requires_prev=["noun", "pronoun", "adjective_na", "adjective_i", "verb_stem", "question_word"],
            requires_next=[] if sub_pos == "ending" else ["noun", "pronoun", "adverb", "verb_stem", "adjective_na", "adjective_i", "special_mark"],
            vector=make_vector(discourse_force=0.45, sociality=0.35, certainty=0.52),
            frequency=0.88,
            lemma=word,
            effect=effect,
        )
    )


# inflections / auxiliaries / copulas
inflection_specs = [
    ("る", "plain", {"tense": "nonpast", "style": "plain"}),
    ("う", "plain", {"tense": "nonpast", "style": "plain"}),
    ("く", "plain", {"tense": "nonpast", "style": "plain"}),
    ("す", "plain", {"tense": "nonpast", "style": "plain"}),
    ("む", "plain", {"tense": "nonpast", "style": "plain"}),
    ("ぬ", "plain", {"tense": "nonpast", "style": "plain"}),
    ("ぶ", "plain", {"tense": "nonpast", "style": "plain"}),
    ("つ", "plain", {"tense": "nonpast", "style": "plain"}),
    ("ぐ", "plain", {"tense": "nonpast", "style": "plain"}),
    ("た", "past", {"tense": "past"}),
    ("て", "te", {"aspect": "te"}),
    ("ない", "negative", {"polarity": "negative"}),
    ("ます", "polite", {"style": "polite"}),
    ("たい", "desire", {"modality": "desire"}),
]
for word, sub_pos, effect in inflection_specs:
    add(
        make_entry(
            word=word,
            category="inflection",
            pos="auxiliary",
            sub_pos=sub_pos,
            entry_type="inflection",
            can_start=False,
            can_end=True,
            independent=False,
            content_word=False,
            function_word=True,
            requires_prev=["verb_stem", "adjective_i"],
            vector=make_vector(temporality=0.6 if sub_pos == "past" else 0.3, discourse_force=0.4),
            frequency=0.8,
            effect=effect,
        )
    )

for word, sub_pos, effect in [
    ("です", "polite", {"copula": True, "style": "polite"}),
    ("だ", "plain", {"copula": True, "style": "plain"}),
]:
    add(
        make_entry(
            word=word,
            category="copula",
            pos="copula",
            sub_pos=sub_pos,
            entry_type="copula",
            can_start=False,
            can_end=True,
            independent=False,
            content_word=False,
            function_word=True,
            requires_prev=["adjective_na", "noun", "pronoun"],
            vector=make_vector(certainty=0.8, discourse_force=0.44),
            frequency=0.9,
            effect=effect,
        )
    )


# pronouns / persons
for word, sub_pos, sociality, agency in [
    ("私", "personal", 0.72, 0.78),
    ("僕", "personal", 0.68, 0.76),
    ("俺", "personal", 0.62, 0.82),
    ("あなた", "personal", 0.84, 0.72),
    ("君", "personal", 0.74, 0.68),
    ("彼", "personal", 0.54, 0.62),
    ("彼女", "personal", 0.58, 0.64),
    ("みんな", "group", 0.92, 0.66),
    ("ユナ", "name", 0.92, 0.74),
    ("なのはさん", "name", 0.94, 0.78),
]:
    add(
        make_entry(
            word=word,
            category="pronoun",
            pos="pronoun",
            sub_pos=sub_pos,
            entry_type="surface",
            can_start=True,
            vector=make_vector(deixis=0.92, sociality=sociality, agency=agency, certainty=0.58),
            slots=[slot("actor", allowed_pos=["pronoun", "noun"], semantic_hint=["person"]), slot("target", allowed_pos=["pronoun", "noun"], semantic_hint=["person"])],
            frequency=0.72,
        )
    )


# question words
for word, sub_pos in [("何", "thing"), ("だれ", "person"), ("どこ", "place"), ("いつ", "time"), ("なぜ", "reason"), ("どう", "manner")]:
    add(
        make_entry(
            word=word,
            category="question_word",
            pos="question_word",
            sub_pos=sub_pos,
            entry_type="surface",
            can_start=True,
            vector=make_vector(deixis=0.85, discourse_force=0.92, certainty=0.22),
            slots=[slot("topic", semantic_hint=[sub_pos])],
            frequency=0.6,
        )
    )


# nouns
noun_specs = [
    ("今日", "time", make_vector(temporality=0.92, deixis=0.74), [slot("time", semantic_hint=["time"])]),
    ("明日", "time", make_vector(temporality=0.94, deixis=0.70), [slot("time", semantic_hint=["time"])]),
    ("昨日", "time", make_vector(temporality=0.94, deixis=0.68), [slot("time", semantic_hint=["time"])]),
    ("今朝", "time", make_vector(temporality=0.88, deixis=0.66), [slot("time", semantic_hint=["time"])]),
    ("今夜", "time", make_vector(temporality=0.88, deixis=0.66), [slot("time", semantic_hint=["time"])]),
    ("家", "place", make_vector(deixis=0.60, sociality=0.54), [slot("location", semantic_hint=["location"])]),
    ("部屋", "place", make_vector(deixis=0.58, sociality=0.32), [slot("location", semantic_hint=["location"])]),
    ("学校", "place", make_vector(deixis=0.52, sociality=0.62), [slot("location", semantic_hint=["location"])]),
    ("会社", "place", make_vector(deixis=0.48, sociality=0.66), [slot("location", semantic_hint=["location"])]),
    ("駅", "place", make_vector(deixis=0.56, sociality=0.42), [slot("location", semantic_hint=["location"])]),
    ("公園", "place", make_vector(deixis=0.64, sociality=0.46), [slot("location", semantic_hint=["location"])]),
    ("東京", "place", make_vector(deixis=0.58, abstractness=0.22), [slot("location", semantic_hint=["location"])]),
    ("日本", "place", make_vector(deixis=0.50, abstractness=0.28), [slot("location", semantic_hint=["location"])]),
    ("仕事", "topic", make_vector(agency=0.72, causality=0.52), [slot("topic", semantic_hint=["work"])]),
    ("話", "topic", make_vector(sociality=0.68, abstractness=0.38), [slot("topic", semantic_hint=["speech"])]),
    ("気分", "state", make_vector(abstractness=0.64, sociality=0.34), [slot("topic", semantic_hint=["state"])]),
    ("体調", "state", make_vector(abstractness=0.56, causality=0.42), [slot("topic", semantic_hint=["health"])]),
    ("ご飯", "thing", make_vector(valence=0.72, abstractness=0.08), [slot("target", semantic_hint=["food"])]),
    ("水", "thing", make_vector(valence=0.62, abstractness=0.08), [slot("target", semantic_hint=["drink"])]),
    ("音楽", "thing", make_vector(valence=0.78, abstractness=0.30), [slot("topic", semantic_hint=["music"])]),
    ("映画", "thing", make_vector(valence=0.72, abstractness=0.34), [slot("topic", semantic_hint=["movie"])]),
    ("本", "thing", make_vector(valence=0.68, abstractness=0.30), [slot("topic", semantic_hint=["book"])]),
    ("ゲーム", "thing", make_vector(valence=0.76, arousal=0.64), [slot("topic", semantic_hint=["game"])]),
    ("友達", "person", make_vector(sociality=0.92, valence=0.74), [slot("target", semantic_hint=["person"])]),
    ("家族", "person", make_vector(sociality=0.96, valence=0.78), [slot("target", semantic_hint=["person"])]),
    ("猫", "animal", make_vector(valence=0.82, arousal=0.54), [slot("target", semantic_hint=["animal"])]),
    ("犬", "animal", make_vector(valence=0.82, arousal=0.58), [slot("target", semantic_hint=["animal"])]),
    ("空", "nature", make_vector(valence=0.66, abstractness=0.24), [slot("topic", semantic_hint=["nature"])]),
    ("雨", "nature", make_vector(valence=0.34, arousal=0.44), [slot("topic", semantic_hint=["weather"])]),
    ("風", "nature", make_vector(valence=0.52, arousal=0.48), [slot("topic", semantic_hint=["weather"])]),
    ("花", "nature", make_vector(valence=0.84, arousal=0.32), [slot("topic", semantic_hint=["nature"])]),
    ("桜", "nature", make_vector(valence=0.90, arousal=0.38), [slot("topic", semantic_hint=["nature"])]),
    ("問題", "abstract", make_vector(valence=0.22, abstractness=0.72, causality=0.68), [slot("topic", semantic_hint=["issue"])]),
    ("理由", "abstract", make_vector(abstractness=0.78, causality=0.92), [slot("cause", semantic_hint=["reason"])]),
    ("意味", "abstract", make_vector(abstractness=0.88, certainty=0.42), [slot("topic", semantic_hint=["meaning"])]),
    ("方法", "abstract", make_vector(abstractness=0.72, agency=0.62), [slot("topic", semantic_hint=["method"])]),
    ("時間", "abstract", make_vector(temporality=0.96, abstractness=0.64), [slot("time", semantic_hint=["time"])]),
    ("場所", "abstract", make_vector(deixis=0.64, abstractness=0.54), [slot("location", semantic_hint=["location"])]),
]

# clean accidental walrus artifact by rebuilding vectors
noun_specs_fixed = []
for word, sub_pos, vector, slots_ in noun_specs:
    if not isinstance(vector, dict):
        vector = make_vector()
    noun_specs_fixed.append((word, sub_pos, vector, slots_))

for word, sub_pos, vector, slots_ in noun_specs_fixed:
    add(
        make_entry(
            word=word,
            category="noun",
            pos="noun",
            sub_pos=sub_pos,
            entry_type="surface",
            can_start=True,
            vector=vector,
            slots=slots_,
            frequency=0.62,
        )
    )


# adjective-na
for word, valence in [
    ("元気", 0.82),
    ("大丈夫", 0.74),
    ("静か", 0.64),
    ("暇", 0.58),
    ("便利", 0.76),
    ("平気", 0.68),
    ("好き", 0.86),
    ("嫌い", 0.18),
    ("きれい", 0.84),
]:
    add(
        make_entry(
            word=word,
            category="adjective_na",
            pos="adjective_na",
            sub_pos="state",
            entry_type="surface",
            can_start=False,
            can_end=False,
            vector=make_vector(valence=valence, abstractness=0.42, certainty=0.64),
            slots=[slot("state", semantic_hint=["state"]), slot("topic", semantic_hint=["state"] )],
            frequency=0.66,
        )
    )


# adjective-i surface words
for word, valence, arousal in [
    ("うれしい", 0.92, 0.62),
    ("かなしい", 0.18, 0.52),
    ("つらい", 0.16, 0.64),
    ("楽しい", 0.94, 0.72),
    ("忙しい", 0.34, 0.82),
    ("眠い", 0.42, 0.34),
    ("寒い", 0.28, 0.48),
    ("暑い", 0.32, 0.56),
    ("痛い", 0.12, 0.78),
    ("欲しい", 0.62, 0.56),
    ("難しい", 0.34, 0.52),
    ("やさしい", 0.84, 0.34),
]:
    add(
        make_entry(
            word=word,
            category="adjective_i",
            pos="adjective_i",
            sub_pos="state",
            entry_type="surface",
            can_start=False,
            can_end=True,
            vector=make_vector(valence=valence, arousal=arousal, abstractness=0.34, certainty=0.62),
            slots=[slot("state", semantic_hint=["state"]), slot("topic", semantic_hint=["state"])],
            frequency=0.66,
        )
    )


# adverbs
for word, sub_pos, vector in [
    ("すごく", "degree", make_vector(arousal=0.74, discourse_force=0.48)),
    ("ちょっと", "degree", make_vector(arousal=0.28, discourse_force=0.34)),
    ("かなり", "degree", make_vector(arousal=0.56, discourse_force=0.40)),
    ("ゆっくり", "manner", make_vector(arousal=0.18, certainty=0.54)),
    ("ちゃんと", "manner", make_vector(certainty=0.74, agency=0.44)),
    ("すぐ", "time", make_vector(temporality=0.84, arousal=0.52)),
    ("まだ", "time", make_vector(temporality=0.72, certainty=0.32)),
    ("もう", "time", make_vector(temporality=0.70, certainty=0.56)),
    ("よく", "manner", make_vector(certainty=0.58, sociality=0.24)),
    ("たぶん", "modal", make_vector(certainty=0.18, abstractness=0.44)),
]:
    add(
        make_entry(
            word=word,
            category="adverb",
            pos="adverb",
            sub_pos=sub_pos,
            entry_type="surface",
            can_start=True,
            vector=vector,
            slots=[slot("manner", semantic_hint=[sub_pos])],
            frequency=0.58,
        )
    )


# interjections
for word, valence, force in [
    ("こんにちは", 0.74, 0.82),
    ("こんばんは", 0.72, 0.80),
    ("おはよう", 0.82, 0.84),
    ("ありがとう", 0.96, 0.90),
    ("ごめん", 0.28, 0.72),
    ("はい", 0.62, 0.66),
    ("うん", 0.68, 0.64),
    ("了解", 0.58, 0.70),
]:
    add(
        make_entry(
            word=word,
            category="interjection",
            pos="interjection",
            sub_pos="formula",
            entry_type="surface",
            can_start=True,
            can_end=True,
            vector=make_vector(valence=valence, discourse_force=force, sociality=0.88),
            slots=[slot("predicate", semantic_hint=["formula"])],
            frequency=0.74,
        )
    )


# verb stems
verb_specs = [
    ("話", "話す", "hanas", "godan_su", "communication", 0.74, [rel("associated", "話", 0.42), rel("associated", "あなた", 0.24)]),
    ("行", "行く", "ik", "godan_ku", "move", 0.72, [rel("associated", "公園", 0.30), rel("associated", "駅", 0.30)]),
    ("来", "来る", "k", "irregular_kuru", "move", 0.70, [rel("associated", "家", 0.24)]),
    ("見", "見る", "mi", "ichidan", "perception", 0.72, [rel("associated", "映画", 0.34), rel("associated", "猫", 0.24)]),
    ("食べ", "食べる", "tabe", "ichidan", "consume", 0.82, [rel("associated", "ご飯", 0.40)]),
    ("飲", "飲む", "nom", "godan_mu", "consume", 0.72, [rel("associated", "水", 0.36)]),
    ("寝", "寝る", "ne", "ichidan", "rest", 0.64, [rel("associated", "夜", 0.20)]),
    ("起き", "起きる", "oki", "ichidan", "rest", 0.58, [rel("associated", "朝", 0.24)]),
    ("会", "会う", "a", "godan_u", "social", 0.84, [rel("associated", "友達", 0.32)]),
    ("思", "思う", "omo", "godan_u", "mental", 0.58, [rel("associated", "理由", 0.24), rel("associated", "意味", 0.24)]),
    ("分か", "分かる", "waka", "godan_ru", "mental", 0.62, [rel("associated", "意味", 0.34)]),
    ("作", "作る", "tsuku", "godan_ru", "create", 0.70, [rel("associated", "ご飯", 0.24), rel("associated", "方法", 0.22)]),
    ("読", "読む", "yom", "godan_mu", "consume", 0.68, [rel("associated", "本", 0.38)]),
    ("書", "書く", "kak", "godan_ku", "create", 0.64, [rel("associated", "話", 0.18)]),
    ("聞", "聞く", "kik", "godan_ku", "perception", 0.66, [rel("associated", "音楽", 0.36)]),
    ("遊", "遊ぶ", "asob", "godan_bu", "social", 0.88, [rel("associated", "ゲーム", 0.34)]),
    ("休", "休む", "yasum", "godan_mu", "rest", 0.72, [rel("associated", "仕事", 0.22)]),
    ("頑張", "頑張る", "ganba", "godan_ru", "effort", 0.82, [rel("associated", "仕事", 0.22)]),
    ("助け", "助ける", "tasuke", "ichidan", "social", 0.86, [rel("associated", "友達", 0.28)]),
    ("使", "使う", "tsuka", "godan_u", "action", 0.60, [rel("associated", "方法", 0.18)]),
    ("置", "置く", "ok", "godan_ku", "action", 0.52, [rel("associated", "場所", 0.24)]),
]


def build_surface_forms(stem: str, lemma: str, conj_class: str) -> List[Dict[str, Any]]:
    if conj_class == "ichidan":
        return [
            {"form": "plain", "surface": f"{stem}る", "tokens": [stem, "る"]},
            {"form": "past", "surface": f"{stem}た", "tokens": [stem, "た"]},
            {"form": "te", "surface": f"{stem}て", "tokens": [stem, "て"]},
            {"form": "polite", "surface": f"{stem}ます", "tokens": [stem, "ます"]},
            {"form": "negative", "surface": f"{stem}ない", "tokens": [stem, "ない"]},
            {"form": "desire", "surface": f"{stem}たい", "tokens": [stem, "たい"]},
        ]

    if conj_class == "irregular_kuru":
        return [
            {"form": "plain", "surface": "来る", "tokens": [stem, "る"]},
            {"form": "past", "surface": "来た", "tokens": [stem, "た"]},
            {"form": "te", "surface": "来て", "tokens": [stem, "て"]},
            {"form": "polite", "surface": "来ます", "tokens": [stem, "ます"]},
            {"form": "negative", "surface": "来ない", "tokens": [stem, "ない"]},
            {"form": "desire", "surface": "来たい", "tokens": [stem, "たい"]},
        ]

    rules = {
        "godan_u": {"plain": "う", "polite": "います", "negative": "わない", "te": "って", "past": "った"},
        "godan_tsu": {"plain": "つ", "polite": "ちます", "negative": "たない", "te": "って", "past": "った"},
        "godan_ru": {"plain": "る", "polite": "ります", "negative": "らない", "te": "って", "past": "った"},
        "godan_mu": {"plain": "む", "polite": "みます", "negative": "まない", "te": "んで", "past": "んだ"},
        "godan_nu": {"plain": "ぬ", "polite": "にます", "negative": "なない", "te": "んで", "past": "んだ"},
        "godan_bu": {"plain": "ぶ", "polite": "びます", "negative": "ばない", "te": "んで", "past": "んだ"},
        "godan_ku": {"plain": "く", "polite": "きます", "negative": "かない", "te": "いて", "past": "いた"},
        "godan_gu": {"plain": "ぐ", "polite": "ぎます", "negative": "がない", "te": "いで", "past": "いだ"},
        "godan_su": {"plain": "す", "polite": "します", "negative": "さない", "te": "して", "past": "した"},
    }
    rule = dict(rules.get(conj_class, {"plain": "る", "polite": "ます", "negative": "ない", "te": "て", "past": "た"}))

    if lemma == "行く":
        rule["te"] = "って"
        rule["past"] = "った"

    return [
        {"form": "plain", "surface": f"{stem}{rule['plain']}", "tokens": [stem, rule['plain']]},
        {"form": "past", "surface": f"{stem}{rule['past']}", "tokens": [stem, "た"]},
        {"form": "te", "surface": f"{stem}{rule['te']}", "tokens": [stem, "て"]},
        {"form": "polite", "surface": f"{stem}{rule['polite']}", "tokens": [stem, "ます"]},
        {"form": "negative", "surface": f"{stem}{rule['negative']}", "tokens": [stem, "ない"]},
        {"form": "desire", "surface": f"{stem}たい", "tokens": [stem, "たい"]},
    ]


for stem, lemma, stem_id, conj_class, semantic_hint, valence, relations in verb_specs:
    add(
        make_entry(
            word=stem,
            category="verb_stem",
            pos="verb_stem",
            sub_pos=semantic_hint,
            entry_type="stem",
            can_start=False,
            can_end=False,
            vector=make_vector(valence=valence, agency=0.82, causality=0.56, sociality=0.44, discourse_force=0.56),
            slots=[
                slot("predicate", required=True, allowed_pos=["verb_stem"], semantic_hint=[semantic_hint]),
                slot("actor", allowed_pos=["noun", "pronoun"], semantic_hint=["person"]),
                slot("target", allowed_pos=["noun", "pronoun"], semantic_hint=["object"]),
                slot("location", allowed_pos=["noun", "pronoun"], semantic_hint=["location"]),
                slot("time", allowed_pos=["noun", "adverb"], semantic_hint=["time"]),
            ],
            relations=relations,
            frequency=0.70,
            lemma=lemma,
            stem_id=stem_id,
            conj_class=conj_class,
            surface_forms=build_surface_forms(stem, lemma, conj_class),
        )
    )


# special irregular verbs kept as surface verbs for stability
for word, sub_pos, valence in [
    ("ある", "existence", 0.52),
    ("いる", "existence", 0.62),
]:
    add(
        make_entry(
            word=word,
            category="verb_stem",
            pos="verb_stem",
            sub_pos=sub_pos,
            entry_type="stem",
            can_start=False,
            can_end=False,
            vector=make_vector(valence=valence, agency=0.24, causality=0.42, discourse_force=0.46),
            slots=[slot("predicate", required=True, allowed_pos=["verb_stem"], semantic_hint=[sub_pos]), slot("location", allowed_pos=["noun", "pronoun"], semantic_hint=["location"])],
            relations=[rel("associated", "場所", 0.32)],
            frequency=0.84,
            lemma=word,
            stem_id=word,
            conj_class="irregular_exist",
            surface_forms=[
                {"form": "plain", "surface": word, "tokens": [word]},
                {"form": "past", "surface": f"{word}た", "tokens": [word, "た"]},
                {"form": "te", "surface": f"{word}て", "tokens": [word, "て"]},
                {"form": "polite", "surface": f"{word}ます", "tokens": [word, "ます"]},
                {"form": "negative", "surface": f"{word}ない", "tokens": [word, "ない"]},
            ],
        )
    )


# indexes
by_pos: Dict[str, List[str]] = {}
can_start: List[str] = []
can_end: List[str] = []
content_words: List[str] = []
function_words: List[str] = []
entry_path: Dict[str, List[str]] = {}

for word, entry in entries.items():
    pos = str(entry["grammar"].get("pos", "unknown"))
    by_pos.setdefault(pos, []).append(word)
    if entry["grammar"].get("can_start"):
        can_start.append(word)
    if entry["grammar"].get("can_end"):
        can_end.append(word)
    if entry["grammar"].get("content_word"):
        content_words.append(word)
    if entry["grammar"].get("function_word"):
        function_words.append(word)
    entry_path[word] = list(entry.get("hierarchy", []))

for values in by_pos.values():
    values.sort()

container = {
    "meta": {
        "version": "v3-split-minimal",
        "semantic_axes": list(SEMANTIC_AXES),
        "entry_count": len(entries),
        "design": "stem-inflection-particle-aware",
    },
    "entries": dict(sorted(entries.items())),
    "indexes": {
        "by_pos": by_pos,
        "can_start": sorted(can_start),
        "can_end": sorted(can_end),
        "content_words": sorted(content_words),
        "function_words": sorted(function_words),
        "entry_path": entry_path,
    },
}

OUTPUT_PATH.write_text(json.dumps(container, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] wrote {OUTPUT_PATH} entries={len(entries)}")
