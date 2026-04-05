# src/llm/llm_router.py
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def load_dotenv_file(dotenv_path: str | Path = ".env") -> None:
    path = Path(dotenv_path)
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_llm_order(config_path: str = "settings/config.yaml") -> List[str]:
    path = Path(config_path)
    if not path.exists():
        raise RuntimeError(f"config not found: {config_path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    order = data.get("llm-api-order", [])
    if not isinstance(order, list) or not order:
        raise RuntimeError("llm-api-order is empty")

    result: List[str] = []
    for x in order:
        s = str(x).strip()
        if s:
            result.append(s)
    return result


class LLMRouter:
    def __init__(
        self,
        config_path: str = "settings/config.yaml",
        dotenv_path: str = ".env",
        verbose: bool = False,
    ):
        self.verbose = verbose

        load_dotenv_file(dotenv_path)

        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        if self.verbose:
            print(f"[LLM] GEMINI_KEY={'set' if self.gemini_key else 'missing'}")
            print(f"[LLM] OPENAI_KEY={'set' if self.openai_key else 'missing'}")

        self.model_order = load_llm_order(config_path)

        if self.gemini_key and genai is not None:
            genai.configure(api_key=self.gemini_key)

        self.openai_client = None
        if self.openai_key and OpenAI is not None:
            self.openai_client = OpenAI(api_key=self.openai_key)

    # =========================================================
    # public
    # =========================================================

    def generate_json(self, prompt: Dict[str, Any], model_name: str | None = None) -> Dict[str, Any]:
        last_error: Optional[str] = None

        order = [model_name] if model_name else list(self.model_order)

        for current_model in order:
            try:
                if self.verbose:
                    print(f"[LLM TRY] {current_model}")

                if current_model.startswith("gemini"):
                    raw_text = self._call_gemini(current_model, prompt)
                else:
                    raw_text = self._call_openai(current_model, prompt)

                parsed = self._safe_parse_json(raw_text)

                if self.verbose:
                    print(f"[LLM OK] {current_model}")

                return parsed

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if self.verbose:
                    print(f"[LLM FAIL] {current_model} error={e}")
                time.sleep(0.5)

        raise RuntimeError(f"All LLM calls failed. last_error={last_error}")

    # =========================================================
    # gemini
    # =========================================================

    def _call_gemini(self, model_name: str, prompt: Dict[str, Any]) -> str:
        if not self.gemini_key or genai is None:
            raise RuntimeError("Gemini not available")

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            json.dumps(prompt, ensure_ascii=False),
            generation_config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )

        text = getattr(response, "text", "") or ""
        text = text.strip()
        if not text:
            raise RuntimeError(f"Gemini returned empty response: model={model_name}")
        return text

    # =========================================================
    # openai
    # =========================================================

    def _call_openai(self, model_name: str, prompt: Dict[str, Any]) -> str:
        if not self.openai_client:
            raise RuntimeError("OpenAI not available")

        response = self.openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You must output JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.2,
        )

        text = response.choices[0].message.content or ""
        text = text.strip()
        if not text:
            raise RuntimeError(f"OpenAI returned empty response: model={model_name}")
        return text

    # =========================================================
    # utils
    # =========================================================

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("empty response text")

        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj

        raise ValueError(f"response is not valid JSON object: {text[:200]!r}")