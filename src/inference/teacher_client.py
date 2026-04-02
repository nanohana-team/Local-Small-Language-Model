from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


def safe_get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


@dataclass
class TeacherConfig:
    name: str
    api_base: str
    api_key_env: str
    model: str
    system_prompt: str
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = 256
    timeout_sec: int = 120

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TeacherConfig":
        required = ["name", "api_base", "api_key_env", "model"]
        for key in required:
            if key not in d:
                raise ValueError(f"Teacher config missing required key: {key}")
        return TeacherConfig(
            name=str(d["name"]),
            api_base=str(d["api_base"]).rstrip("/"),
            api_key_env=str(d["api_key_env"]),
            model=str(d["model"]),
            system_prompt=str(
                d.get(
                    "system_prompt",
                    "あなたは日本語で自然に会話するAIです。簡潔で自然な返答をしてください。",
                )
            ),
            temperature=float(d.get("temperature", 0.8)),
            top_p=float(d.get("top_p", 1.0)),
            max_tokens=int(d.get("max_tokens", 256)),
            timeout_sec=int(d.get("timeout_sec", 120)),
        )


class OpenAICompatibleClient:
    def __init__(self, api_base: str, api_key: str, timeout_sec: int = 120):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> Dict[str, Any]:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        return response.json()


def generate_teacher_reply(
    teacher: TeacherConfig,
    messages: List[Dict[str, str]],
) -> str:
    api_key = safe_get_env(teacher.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable '{teacher.api_key_env}' is not set for teacher '{teacher.name}'."
        )

    client = OpenAICompatibleClient(
        api_base=teacher.api_base,
        api_key=api_key,
        timeout_sec=teacher.timeout_sec,
    )
    full_messages = [{"role": "system", "content": teacher.system_prompt}, *messages]
    result = client.chat_completion(
        model=teacher.model,
        messages=full_messages,
        temperature=teacher.temperature,
        max_tokens=teacher.max_tokens,
        top_p=teacher.top_p,
    )
    choices = result.get("choices", [])
    if not choices:
        raise RuntimeError("Teacher API response does not contain choices.")
    content = str((choices[0].get("message") or {}).get("content", "")).strip()
    return content
