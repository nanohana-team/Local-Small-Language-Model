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
    if not isinstance(raw_profiles, Mapping):
        return profiles
    for mode, profile_payload in raw_profiles.items():
        if not isinstance(profile_payload, Mapping):
            continue
        profiles[str(mode)] = TeacherProfile.from_mapping(str(mode), profile_payload)
    return profiles
