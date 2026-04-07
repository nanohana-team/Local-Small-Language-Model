from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    raw: dict[str, Any]


class LLMGateway:
    def __init__(
        self,
        *,
        env_path: str | Path = '.env',
        config_path: str | Path = 'settings/config.yaml',
    ) -> None:
        self.env_path = Path(env_path)
        self.config_path = Path(config_path)
        if self.env_path.exists():
            load_dotenv(self.env_path)
        self.model_order, self.higher_model_order = self._load_model_orders(self.config_path)
        self._gemini_client = None
        self._openai_client = None
        self._local_client = None

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        purpose: str,
        preferred_models: Optional[Iterable[str]] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 512,
    ) -> LLMResponse:
        model_candidates = self._resolve_model_candidates(
            purpose=purpose,
            preferred_models=preferred_models,
        )
        if not model_candidates:
            raise RuntimeError('No model candidates are configured for LLMGateway.')

        errors: list[str] = []
        for model_name in model_candidates:
            try:
                provider = self._detect_provider(model_name)
                LOGGER.info(
                    'llm_gateway.generate.try purpose=%s provider=%s model=%s',
                    purpose,
                    provider,
                    model_name,
                )
                if provider == 'gemini':
                    return self._generate_gemini(
                        model_name=model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                if provider == 'openai':
                    return self._generate_openai(
                        model_name=model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                if provider == 'local_openai':
                    return self._generate_local_openai(
                        model_name=model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
                raise RuntimeError(f'Unsupported provider for model: {model_name}')
            except Exception as exc:
                LOGGER.warning(
                    'llm_gateway.generate.failed purpose=%s model=%s error=%s',
                    purpose,
                    model_name,
                    exc,
                )
                errors.append(f'{model_name}: {exc}')

        raise RuntimeError('All configured LLM backends failed: ' + ' | '.join(errors))

    def _load_model_orders(self, config_path: Path) -> tuple[list[str], list[str]]:
        if not config_path.exists():
            return [], []
        data = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
        order = data.get('llm-api-order', [])
        higher_order = data.get('llm-api-higher-order', [])
        return (
            [str(x).strip() for x in order if str(x).strip()],
            [str(x).strip() for x in higher_order if str(x).strip()],
        )

    def _resolve_model_candidates(
        self,
        *,
        purpose: str,
        preferred_models: Optional[Iterable[str]] = None,
    ) -> list[str]:
        preferred = [str(x).strip() for x in (preferred_models or []) if str(x).strip()]
        if preferred:
            return preferred

        if self._purpose_requires_higher_order(purpose):
            return self._merge_model_orders(self.higher_model_order, self.model_order)
        return list(self.model_order)

    def _purpose_requires_higher_order(self, purpose: str) -> bool:
        normalized = str(purpose or '').strip().lower()
        if not normalized or not self.higher_model_order:
            return False
        return normalized in {
            'unknown_word_entry_relearn',
            'target_generation',
            'external_evaluation',
        }

    def _merge_model_orders(self, *orders: Iterable[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for order in orders:
            for model_name in order:
                normalized = str(model_name).strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(normalized)
        return merged

    def _detect_provider(self, model_name: str) -> str:
        normalized = str(model_name).strip()
        if normalized.startswith('local:'):
            return 'local_openai'
        if normalized.startswith('gemini') or normalized.startswith('gemma'):
            return 'gemini'
        return 'openai'

    def _generate_gemini(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> LLMResponse:
        from google import genai
        from google.genai import types

        api_key = os.getenv('GEMINI_API_KEY', '').strip()
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY is not set.')
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=api_key)

        response = self._gemini_client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        text = getattr(response, 'text', '') or self._extract_text_from_dump(response)
        if not text.strip():
            raise RuntimeError('Gemini returned empty text.')
        return LLMResponse(
            text=text.strip(),
            provider='gemini',
            model=model_name,
            raw=self._model_dump(response),
        )

    def _generate_openai(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> LLMResponse:
        from openai import OpenAI

        api_key = (
            os.getenv('OPENAI_API_KEY', '').strip()
            or os.getenv('OPENAI_KEY', '').strip()
        )
        if not api_key:
            raise RuntimeError('OPENAI_API_KEY is not set.')
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=api_key)

        response = self._openai_client.responses.create(
            model=model_name,
            input=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        text = getattr(response, 'output_text', '') or self._extract_text_from_dump(response)
        if not text.strip():
            raise RuntimeError('OpenAI returned empty text.')
        return LLMResponse(
            text=text.strip(),
            provider='openai',
            model=model_name,
            raw=self._model_dump(response),
        )

    def _generate_local_openai(
        self,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> LLMResponse:
        from openai import OpenAI

        base_url = os.getenv('LOCAL_OPENAI_BASE_URL', 'http://127.0.0.1:1234/v1').strip()
        api_key = os.getenv('LOCAL_OPENAI_API_KEY', 'local')
        if self._local_client is None:
            self._local_client = OpenAI(base_url=base_url, api_key=api_key)

        real_model = model_name.split(':', 1)[1] if ':' in model_name else model_name
        response = self._local_client.chat.completions.create(
            model=real_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
        choice = response.choices[0]
        text = (choice.message.content or '').strip()
        if not text:
            raise RuntimeError('Local OpenAI-compatible backend returned empty text.')
        return LLMResponse(
            text=text,
            provider='local_openai',
            model=real_model,
            raw=self._model_dump(response),
        )

    def _extract_text_from_dump(self, response: Any) -> str:
        dump = self._model_dump(response)
        chunks: list[str] = []

        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == 'text' and isinstance(item, str) and item.strip():
                        chunks.append(item.strip())
                    else:
                        walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

        walk(dump)
        return '\n'.join(dict.fromkeys(chunks))

    def _model_dump(self, response: Any) -> dict[str, Any]:
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        if hasattr(response, 'to_dict'):
            return response.to_dict()
        if isinstance(response, dict):
            return response
        try:
            return json.loads(json.dumps(response, default=str))
        except Exception:
            return {'repr': repr(response)}


_JSON_BLOCK_RE = re.compile(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', re.DOTALL)


def extract_json_block(text: str) -> str:
    raw = str(text or '').strip()
    if not raw:
        return '{}'
    match = _JSON_BLOCK_RE.search(raw)
    if match:
        return match.group(1).strip()
    start_obj = raw.find('{')
    end_obj = raw.rfind('}')
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return raw[start_obj : end_obj + 1]
    start_arr = raw.find('[')
    end_arr = raw.rfind(']')
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        return raw[start_arr : end_arr + 1]
    return raw
