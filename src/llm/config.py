from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml
from dotenv import load_dotenv

from .base import TeacherProfile


DEFAULT_LLM_ORDER = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gpt-5.4-mini",
    "gpt-5-mini",
]

DEFAULT_TEACHER_PROFILES: Dict[str, Dict[str, Any]] = {
    "evaluator": {
        "system_prompt": (
            "You are an external evaluator for a relation-first Japanese language system. "
            "Return only JSON. Score the final response from 0.0 to 1.0 and explain the main issue briefly."
        ),
        "user_prompt_template": (
            "Evaluate the following turn payload. Return JSON with keys: "
            "label, external_score, feedback_text, strengths, issues.\n\n{payload_json}"
        ),
        "max_output_tokens": 600,
        "temperature": 0.1,
    },
    "teacher": {
        "system_prompt": (
            "You are an external teacher for a relation-first Japanese language system. "
            "Return only JSON. Improve the response while preserving the user's apparent intent."
        ),
        "user_prompt_template": (
            "Teach from the following turn payload. Return JSON with keys: "
            "label, external_score, feedback_text, teacher_target, teacher_hints. "
            "teacher_hints must be an object with keys missing_slots, recommended_relation_types, issues.\n\n{payload_json}"
        ),
        "max_output_tokens": 900,
        "temperature": 0.2,
    },
    "lexicon_enricher": {
        "system_prompt": (
            "You enrich an auxiliary Japanese lexicon for a relation-first language system. "
            "Return JSON only. For each unknown term, provide a short safe lexical entry that can be stored separately from the base dictionary."
        ),
        "user_prompt_template": (
            "Generate auxiliary lexicon entries from the following payload. Return JSON with keys: "
            "entries, notes. entries must be an array of objects with keys: "
            "surface, reading, pos, category, short_definition, surface_forms, related_terms.\n\n{payload_json}"
        ),
        "max_output_tokens": 1200,
        "temperature": 0.2,
    },
    "input_generator": {
        "system_prompt": (
            "You generate diverse Japanese user utterances for a relation-first learning loop. "
            "Return JSON only. Prefer short, natural, varied prompts that are safe and useful for training."
        ),
        "user_prompt_template": (
            "Generate learning inputs from the following payload. Return JSON with keys: "
            "inputs, notes. inputs must be an array of short Japanese user utterances only.\n\n{payload_json}"
        ),
        "max_output_tokens": 1200,
        "temperature": 0.8,
    },
}


def load_environment() -> None:
    load_dotenv(override=False)



def _load_yaml(path: str | Path) -> Dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    with target.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}



def load_llm_order(path: str | Path = "settings/LLM_order.yaml") -> List[str]:
    payload = _load_yaml(path)
    candidates = payload.get("llm-api-order")
    if not isinstance(candidates, list):
        return list(DEFAULT_LLM_ORDER)
    order = [str(item).strip() for item in candidates if str(item).strip()]
    return order or list(DEFAULT_LLM_ORDER)



def load_teacher_profiles(path: str | Path = "settings/teacher_profiles.yaml") -> Dict[str, TeacherProfile]:
    payload = _load_yaml(path)
    raw_profiles = payload.get("profiles") if isinstance(payload.get("profiles"), Mapping) else payload
    profiles: Dict[str, TeacherProfile] = {}
    source = DEFAULT_TEACHER_PROFILES | (dict(raw_profiles) if isinstance(raw_profiles, Mapping) else {})
    for mode, profile_payload in source.items():
        if not isinstance(profile_payload, Mapping):
            continue
        profiles[str(mode)] = TeacherProfile.from_mapping(str(mode), profile_payload)
    for mode, profile_payload in DEFAULT_TEACHER_PROFILES.items():
        profiles.setdefault(mode, TeacherProfile.from_mapping(mode, profile_payload))
    return profiles
